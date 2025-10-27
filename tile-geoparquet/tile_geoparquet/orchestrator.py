from pathlib import Path
from typing import Optional, Set
import logging
import pyarrow.parquet as pq
from .datasource import DataSource, GeoParquetSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool

logger = logging.getLogger(__name__)

class RoundOrchestrator:
    def __init__(self, source: DataSource, assigner: TileAssignerFromCSV, outdir: str,
                 max_parallel_files: int, row_group_rows: int, compression: str = "zstd"):
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.src_schema = source.schema()
        self.max_parallel_files = max_parallel_files
        self.row_group_rows = row_group_rows
        self.compression = compression

    def _run_one_round(self, ds: DataSource, round_id: int) -> Optional[Path]:
        logger.info(f"Starting round {round_id}")
        pool = WriterPool(self.outdir, self.src_schema, self.max_parallel_files, self.row_group_rows, self.compression)
        pool.begin_round(round_id)
        open_tiles: Set[str] = set()
        cap = self.max_parallel_files - 1

        for batch_idx, batch in enumerate(ds.iter_tables()):
            logger.debug(f"Round {round_id}: processing batch {batch_idx}")
            parts = self.assigner.partition_by_tile(batch)
            logger.debug(f"Round {round_id}: partitioner returned {len(parts)} tiles for batch {batch_idx}")
            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    open_tiles.add(tile_id)
                    pool.append_tile_rows(tile_id, sub)
                else:
                    pool.divert_to_overflow(sub)

        overflow_path = pool.end_round()
        if overflow_path and overflow_path.exists():
            logger.info(f"Round {round_id}: overflow file created at {overflow_path}")
            pf = pq.ParquetFile(str(overflow_path))
            if pf.metadata.num_rows == 0:
                overflow_path.unlink(missing_ok=True)
                logger.info(f"Round {round_id}: overflow file empty, removed")
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
            logger.info(f"Round {round_id} produced overflow; continuing with overflow file {overflow_path}")
            ds = GeoParquetSource(str(overflow_path))
            round_id += 1
        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                if pq.ParquetFile(str(p)).metadata.num_rows == 0:
                    p.unlink()
            except Exception:
                pass
