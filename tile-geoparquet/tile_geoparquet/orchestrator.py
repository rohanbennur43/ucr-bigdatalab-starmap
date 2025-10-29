from __future__ import annotations
from pathlib import Path
from typing import Optional, Set, List
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from .datasource import DataSource, GeoParquetSource
from .utils_large import ensure_large_types
from .writer_pool import WriterPool, SortMode

logger = logging.getLogger(__name__)

class RoundOrchestrator:
    """
    Runs bounded-writer tiling in rounds:
      - At most (max_parallel_files - 1) tile writers are open at a time; one overflow writer per round.
      - Rows for tiles beyond the cap are diverted to a per-round overflow file.
      - End of round: tiles are written once (single write per tile); overflow becomes next round input.
      - Process stops when overflow is empty/nonexistent.
    """
    def __init__(
        self,
        source: DataSource,
        assigner,
        outdir: str,
        max_parallel_files: int,
        compression: str = "zstd",
        sort_mode: str = SortMode.ZORDER,
        sort_keys: Optional[list] = None,
        sfc_bits: int = 16,
    ):
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.src_schema = source.schema()
        self.max_parallel_files = max_parallel_files
        self.compression = compression
        self.sort_mode = sort_mode
        self.sort_keys = sort_keys
        self.sfc_bits = int(sfc_bits)
        # Try to discover geometry column name from assigner; fallback to 'geometry'
        self.geom_col = getattr(assigner, "geom_col", getattr(assigner, "_geom_col", "geometry"))

    def _write_overflow(self, overflow_batches: List[pa.Table], round_id: int) -> Optional[Path]:
        if not overflow_batches:
            return None
        upgraded = [ensure_large_types(b.combine_chunks(), self.geom_col) for b in overflow_batches]
        tbl = pa.concat_tables(upgraded, promote=True)
        tbl = ensure_large_types(tbl, self.geom_col)
        out = Path(self.outdir) / f"_overflow_round_{round_id}.parquet"
        pq.write_table(tbl, out, compression=self.compression)
        return out

    def _run_one_round(self, ds: DataSource, round_id: int) -> Optional[Path]:
        logger.info("Starting round %d", round_id)

        pool = WriterPool(
            outdir=self.outdir,
            compression=self.compression,
            geom_col=self.geom_col,
            max_parallel_files=self.max_parallel_files,  # used only at flush time
            sort_mode=self.sort_mode,
            sort_keys=self.sort_keys,
            sfc_bits=self.sfc_bits,
        )

        # Track which tiles we allow to open this round (<= max_parallel_files - 1)
        open_tiles: Set[str] = set()
        cap = max(1, self.max_parallel_files - 1)
        overflow_batches: List[pa.Table] = []

        for batch_idx, batch in enumerate(ds.iter_tables()):
            parts = self.assigner.partition_by_tile(batch)
            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    open_tiles.add(tile_id)
                    pool.append(tile_id, sub)
                else:
                    # divert entire sub-table to overflow
                    overflow_batches.append(sub)

        # Write tiles once (per-tile concat+sort+bbox+single write)
        pool.flush_all()

        # Write overflow (if any)
        overflow_path = self._write_overflow(overflow_batches, round_id)
        if overflow_path and overflow_path.exists():
            pf = pq.ParquetFile(str(overflow_path))
            if pf.metadata.num_rows == 0:
                overflow_path.unlink(missing_ok=True)
                logger.info("Round %d: overflow file empty, removed", round_id)
                return None
            logger.info("Round %d: overflow file created at %s (rows=%d)",
                        round_id, overflow_path, pf.metadata.num_rows)
            return overflow_path

        return None

    def run(self):
        round_id = 0
        ds: DataSource = self.source

        while True:
            overflow_path = self._run_one_round(ds, round_id)
            if overflow_path is None:
                break
            logger.info("Round %d produced overflow; continuing with %s", round_id, overflow_path)
            ds = GeoParquetSource(str(overflow_path))
            round_id += 1

        # Best-effort cleanup of any empty overflow files
        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                if pq.ParquetFile(str(p)).metadata.num_rows == 0:
                    p.unlink()
            except Exception:
                pass