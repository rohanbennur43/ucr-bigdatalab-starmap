from pathlib import Path
from typing import Optional, Set
import logging

import pyarrow.parquet as pq

from .datasource import DataSource, GeoParquetSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool

logger = logging.getLogger(__name__)


class RoundOrchestrator:
    """
    Runs bounded-writer tiling in rounds:
      - At most (max_parallel_files - 1) tile writers are open at a time; one overflow writer per round.
      - Rows for tiles that exceed the per-round capacity are diverted to an overflow file.
      - If overflow is non-empty at the end of a round, it becomes the input for the next round.
      - Process ends when a round produces no overflow (or empty overflow).
    """
    def __init__(
        self,
        source: DataSource,
        assigner: TileAssignerFromCSV,
        outdir: str,
        max_parallel_files: int,
        row_group_rows: int,
        compression: str = "zstd",
    ):
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.src_schema = source.schema()
        self.max_parallel_files = max_parallel_files
        self.row_group_rows = row_group_rows
        self.compression = compression

    def _run_one_round(self, ds: DataSource, round_id: int) -> Optional[Path]:
        logger.info("Starting round %d", round_id)

        # Pass bbox_resolver to inject per-tile bbox into GeoParquet metadata in writer_pool
        pool = WriterPool(
            self.outdir,
            self.src_schema,
            self.max_parallel_files,
            self.row_group_rows,
            self.compression,
            bbox_resolver=self.assigner.tile_bbox,
        )
        pool.begin_round(round_id)

        # Track which tiles we allow to open this round (<= max_parallel_files - 1)
        open_tiles: Set[str] = set()
        cap = self.max_parallel_files - 1
        # logger.info("DS size: %s", len(list(ds.iter_tables())))
        for batch_idx, batch in enumerate(ds.iter_tables()):
            logger.debug("Round %d: processing batch %d", round_id, batch_idx)
            parts = self.assigner.partition_by_tile(batch)
            logger.debug("Round %d: partitioner returned %d tiles for batch %d",
                         round_id, len(parts), batch_idx)

            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    open_tiles.add(tile_id)
                    pool.append_tile_rows(tile_id, sub)
                else:
                    pool.divert_to_overflow(sub)

        overflow_path = pool.end_round()

        # If overflow exists but is empty, remove and signal completion
        if overflow_path and overflow_path.exists():
            logger.info("Round %d: overflow file created at %s", round_id, overflow_path)
            pf = pq.ParquetFile(str(overflow_path))
            if pf.metadata.num_rows == 0:
                overflow_path.unlink(missing_ok=True)
                logger.info("Round %d: overflow file empty, removed", round_id)
                return None
            return overflow_path

        return None

    def run(self):
        round_id = 0
        ds: DataSource = self.source

        while True:
            overflow_path = self._run_one_round(ds, round_id)
            if overflow_path is None:
                break
            logger.info("Round %d produced overflow; continuing with overflow file %s",
                        round_id, overflow_path)
            ds = GeoParquetSource(str(overflow_path))
            round_id += 1

        # Cleanup any empty overflow files left around (extension fixed to .parquet)
        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                if pq.ParquetFile(str(p)).metadata.num_rows == 0:
                    p.unlink()
            except Exception:
                # Best effort cleanup
                pass
