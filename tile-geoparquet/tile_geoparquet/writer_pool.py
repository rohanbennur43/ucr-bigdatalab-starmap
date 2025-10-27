from pathlib import Path
from typing import Dict, List, Optional
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import json

logger = logging.getLogger(__name__)

class WriterPool:
    def __init__(self, outdir: str, src_schema: pa.Schema, max_parallel_files: int, row_group_rows: int, compression: str = "zstd"):
        if max_parallel_files < 2:
            raise ValueError("max_parallel_files must be >= 2 (tiles + 1 overflow).")
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.src_schema = src_schema
        self.K = max_parallel_files - 1
        self.row_group_rows = row_group_rows
        self.compression = compression
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._open_tiles: List[str] = []
        self._buffers: Dict[str, List[pa.Table]] = {}
        self._overflow_writer: Optional[pq.ParquetWriter] = None
        self._overflow_path: Optional[Path] = None
        logger.info(f"WriterPool created: outdir={self.outdir}, K={self.K}, row_group_rows={self.row_group_rows}, compression={self.compression}")

    def begin_round(self, round_id: int):
        self._overflow_path = self.outdir / f"_overflow_round_{round_id}.parquet"
        if self._overflow_path.exists(): self._overflow_path.unlink()
        self._overflow_writer = pq.ParquetWriter(str(self._overflow_path), self.src_schema, compression=self.compression)
        logger.info(f"Begin round {round_id}: overflow path={self._overflow_path}")

    def _ensure_writer(self, tile_id: str):
        if tile_id in self._writers: return
        if len(self._open_tiles) >= self.K:
            victim = self._open_tiles.pop(0)
            self._flush_all(victim)
            self._writers[victim].close()
            del self._writers[victim]
            logger.info(f"Closed writer for victim tile {victim} to make room")
        path = self.outdir / f"{tile_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        schema_with_bbox = self._schema_with_tile_bbox(tile_id)
        self._writers[tile_id] = pq.ParquetWriter(str(path), schema_with_bbox, compression=self.compression)
        logger.info(f"Source schema: {schema_with_bbox}")
        self._open_tiles.append(tile_id)
        logger.info(f"Opened writer for tile {tile_id}")

    def _schema_with_tile_bbox(self, tile_id: str) -> pa.Schema:
        md = dict(self.src_schema.metadata or {})
        geo_raw = md.get(b"geo")
        if not geo_raw or self._bbox_resolver is None:
            return self.src_schema
        geo = json.loads(geo_raw.decode("utf-8"))
        primary = geo.get("primary_column") or "geometry"
        col_md = geo.setdefault("columns", {}).setdefault(primary, {})
        xmin, ymin, xmax, ymax = self._bbox_resolver(tile_id)
        col_md["bbox"] = [float(xmin), float(ymin), float(xmax), float(ymax)]
        md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
        return pa.schema(self.src_schema, metadata=md)


    def append_tile_rows(self, tile_id: str, tbl: pa.Table):
        if tbl.num_rows == 0: return
        buf = self._buffers.setdefault(tile_id, [])
        buf.append(tbl)
        self._flush_row_groups(tile_id)
        logger.debug(f"Appended {tbl.num_rows} rows to buffer for tile {tile_id} (buffered batches={len(self._buffers[tile_id])})")

    def divert_to_overflow(self, tbl: pa.Table):
        if tbl.num_rows == 0: return
        assert self._overflow_writer is not None
        self._overflow_writer.write_table(tbl)
        logger.info(f"Diverted {tbl.num_rows} rows to overflow")

    def _flush_row_groups(self, tile_id: str):
        if tile_id not in self._buffers: return
        total = sum(t.num_rows for t in self._buffers[tile_id])
        logger.debug(f"Flushing row groups for tile {tile_id}: total buffered rows={total}")
        while total >= self.row_group_rows:
            need = self.row_group_rows; chunks: List[pa.Table] = []
            while need > 0:
                t = self._buffers[tile_id][0]
                if t.num_rows <= need:
                    chunks.append(t); self._buffers[tile_id].pop(0); need -= t.num_rows
                else:
                    chunks.append(t.slice(0, need)); self._buffers[tile_id][0] = t.slice(need); need = 0
            out = pa.concat_tables(chunks) if len(chunks) > 1 else chunks[0]
            self._ensure_writer(tile_id)
            self._writers[tile_id].write_table(out, row_group_size=self.row_group_rows)
            total -= self.row_group_rows
            logger.info(f"Flushed row_group of {self.row_group_rows} rows for tile {tile_id}")

    def _flush_all(self, tile_id: str):
        buf = self._buffers.get(tile_id, [])
        if not buf: return
        out = pa.concat_tables(buf) if len(buf) > 1 else buf[0]
        self._buffers[tile_id].clear()
        self._ensure_writer(tile_id)
        self._writers[tile_id].write_table(out, row_group_size=self.row_group_rows)
        logger.info(f"Flushed ALL buffered rows for tile {tile_id} ({out.num_rows} rows)")

    def end_round(self) -> Path:
        # 1) Flush ALL buffered tiles (even those without an open writer yet)
        for tile_id in list(self._buffers.keys()):
            # _flush_all will ensure a writer and write the remaining rows
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
        path = self._overflow_path
        self._overflow_writer = None
        self._overflow_path = None
        return path  # type: ignore
