from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class WriterPool:
    def __init__(
        self,
        outdir: str,
        src_schema: pa.Schema,
        max_parallel_files: int,
        row_group_rows: int,
        compression: str = "zstd",
        bbox_resolver=None,  # callable: tile_id -> (xmin, ymin, xmax, ymax)
    ):
        if max_parallel_files < 2:
            raise ValueError("max_parallel_files must be >= 2 (tiles + 1 overflow).")
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Keep an immutable reference to the incoming schema
        self.src_schema = src_schema

        # Capacity: K tile writers + 1 overflow
        self.K = max_parallel_files - 1
        self.row_group_rows = int(row_group_rows)
        self.compression = compression

        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._open_tiles: List[str] = []
        self._buffers: Dict[str, List[pa.Table]] = {}

        self._overflow_writer: Optional[pq.ParquetWriter] = None
        self._overflow_path: Optional[Path] = None

        self._bbox_resolver = bbox_resolver

        logger.info(
            "WriterPool created: outdir=%s, K=%d, row_group_rows=%d, compression=%s",
            self.outdir, self.K, self.row_group_rows, self.compression
        )

    def begin_round(self, round_id: int):
        """Create/replace the overflow writer for this round."""
        self._overflow_path = self.outdir / f"_overflow_round_{round_id}.parquet"
        if self._overflow_path.exists():
            self._overflow_path.unlink()
        # Overflow uses the unmodified source schema
        self._overflow_writer = pq.ParquetWriter(
            str(self._overflow_path),
            self.src_schema,
            compression=self.compression,
        )
        logger.info("Begin round %d: overflow path=%s", round_id, self._overflow_path)

    def _ensure_writer(self, tile_id: str):
        """Ensure a writer exists for tile_id; may evict an older writer to honor K."""
        if tile_id in self._writers:
            return

        # Evict oldest open tile if at capacity
        if len(self._open_tiles) >= self.K:
            victim = self._open_tiles.pop(0)
            self._flush_all(victim)
            self._writers[victim].close()
            del self._writers[victim]
            logger.info("Closed writer for victim tile %s to make room", victim)

        path = self.outdir / f"{tile_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Inject per-tile bbox into a cloned schema (if possible)
        schema_with_bbox = self._schema_with_tile_bbox(tile_id)

        self._writers[tile_id] = pq.ParquetWriter(
            str(path),
            schema_with_bbox,
            compression=self.compression,
        )
        self._open_tiles.append(tile_id)
        logger.info("Opened writer for tile %s", tile_id)
        logger.debug("Tile %s schema metadata: %s", tile_id, schema_with_bbox.metadata)

    def _schema_with_tile_bbox(self, tile_id: str) -> pa.Schema:
        """
        Return a copy of the source schema with GeoParquet 'geo' JSON updated to
        set columns[primary].bbox = (xmin, ymin, xmax, ymax) for this tile.
        Falls back to the unmodified schema if metadata/resolver is absent or errors occur.
        """
        try:
            md = dict(self.src_schema.metadata or {})
            geo_raw = md.get(b"geo")
            if not geo_raw or self._bbox_resolver is None:
                # Return a shallow copy so the ParquetWriter can't mutate the original
                return pa.schema(self.src_schema)

            geo = json.loads(geo_raw.decode("utf-8"))
            primary = geo.get("primary_column") or "geometry"
            cols = geo.setdefault("columns", {})
            col_md = cols.setdefault(primary, {})

            xmin, ymin, xmax, ymax = self._bbox_resolver(tile_id)
            col_md["bbox"] = [float(xmin), float(ymin), float(xmax), float(ymax)]

            md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
            return pa.schema(self.src_schema, metadata=md)
        except Exception as e:
            logger.warning("Failed to inject bbox metadata for tile %s: %s", tile_id, e)
            return pa.schema(self.src_schema)

    def append_tile_rows(self, tile_id: str, tbl: pa.Table):
        if tbl.num_rows == 0:
            return
        buf = self._buffers.setdefault(tile_id, [])
        buf.append(tbl)
        self._flush_row_groups(tile_id)
        logger.debug(
            "Appended %d rows to buffer for tile %s (buffered batches=%d)",
            tbl.num_rows, tile_id, len(self._buffers[tile_id])
        )

    def divert_to_overflow(self, tbl: pa.Table):
        """Write a sub-table to the round overflow file."""
        if tbl.num_rows == 0:
            return
        assert self._overflow_writer is not None
        # Respect configured row group size for overflow as well
        self._overflow_writer.write_table(tbl, row_group_size=self.row_group_rows)
        logger.info("Diverted %d rows to overflow", tbl.num_rows)

    def _flush_row_groups(self, tile_id: str):
        """Flush full row groups for the given tile from its buffered batches."""
        if tile_id not in self._buffers:
            return
        total = sum(t.num_rows for t in self._buffers[tile_id])
        logger.debug("Flushing row groups for tile %s: total buffered rows=%d", tile_id, total)

        while total >= self.row_group_rows:
            need = self.row_group_rows
            chunks: List[pa.Table] = []
            while need > 0:
                t = self._buffers[tile_id][0]
                if t.num_rows <= need:
                    chunks.append(t)
                    self._buffers[tile_id].pop(0)
                    need -= t.num_rows
                else:
                    chunks.append(t.slice(0, need))
                    self._buffers[tile_id][0] = t.slice(need)
                    need = 0

            out = pa.concat_tables(chunks) if len(chunks) > 1 else chunks[0]
            self._ensure_writer(tile_id)
            self._writers[tile_id].write_table(out, row_group_size=self.row_group_rows)
            total -= self.row_group_rows
            logger.info("Flushed row_group of %d rows for tile %s", self.row_group_rows, tile_id)

    def _flush_all(self, tile_id: str):
        """Flush all remaining buffered rows for a tile as a final (possibly short) row group."""
        buf = self._buffers.get(tile_id, [])
        if not buf:
            return
        out = pa.concat_tables(buf) if len(buf) > 1 else buf[0]
        self._buffers[tile_id].clear()
        self._ensure_writer(tile_id)
        self._writers[tile_id].write_table(out, row_group_size=self.row_group_rows)
        logger.info("Flushed ALL buffered rows for tile %s (%d rows)", tile_id, out.num_rows)

    def end_round(self) -> Path:
        """
        Flush and close all writers, clear buffers, and close the overflow writer.
        Returns the overflow file path (it may be empty; caller can check and remove).
        """
        # 1) Flush ALL buffered tiles (even those without an open writer yet)
        for tile_id in list(self._buffers.keys()):
            self._flush_all(tile_id)

        # 2) Close & drop any writers we opened
        for tile_id in list(self._writers.keys()):
            self._writers[tile_id].close()
            del self._writers[tile_id]
        self._open_tiles.clear()

        # 3) Clear buffers explicitly
        self._buffers.clear()

        # 4) Close overflow
        assert self._overflow_writer is not None
        self._overflow_writer.close()
        path = self._overflow_path  # may be None in theory, but begin_round() must be called
        self._overflow_writer = None
        self._overflow_path = None

        return path  # type: ignore
