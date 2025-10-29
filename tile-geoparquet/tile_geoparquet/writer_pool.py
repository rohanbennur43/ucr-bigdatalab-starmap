from __future__ import annotations

import json
import math
import os
import multiprocessing
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb

from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)

# ------------------------- Sorting configuration -------------------------

@dataclass
class SortKey:
    column: str
    ascending: bool = True

class SortMode:
    NONE = "none"
    COLUMNS = "columns"
    ZORDER = "zorder"
    HILBERT = "hilbert"

# ------------------------- Utility: Morton (Z-order) ----------------------

def _scale_to_uint(v: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(v, dtype=np.uint64)
    rng = vmax - vmin
    x = (v - vmin) / rng
    x = np.clip(x, 0.0, 1.0)
    return (x * ((1 << bits) - 1)).astype(np.uint64, copy=False)

def _interleave_bits_2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    x = x.astype(np.uint64, copy=False)
    y = y.astype(np.uint64, copy=False)

    def part1by1(n):
        n &= 0x00000000FFFFFFFF
        n = (n | (n << 16)) & 0x0000FFFF0000FFFF
        n = (n | (n << 8))  & 0x00FF00FF00FF00FF
        n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
        n = (n | (n << 2))  & 0x3333333333333333
        n = (n | (n << 1))  & 0x5555555555555555
        return n

    return (part1by1(y) << 1) | part1by1(x)

# ------------------------- Writer Pool (flush-once, MPF rounds) -------------------------

class WriterPool:
    """
    Buffer-everything writer:
      - append(tile_id, table): buffer Arrow Tables per tile (no IO)
      - flush_all(): writes once, in rounds of up to `max_parallel_files` concurrent files
    """

    def __init__(
        self,
        outdir: str,
        compression: str = "zstd",
        geom_col: str = "geometry",
        max_parallel_files: Optional[int] = None,
        sort_mode: str = SortMode.ZORDER,
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]] = None,
        sfc_bits: int = 16,
        parquet_writer_args: Optional[dict] = None,
        global_extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.outdir = outdir
        self.compression = compression
        self.geom_col = geom_col
        self.sort_mode = sort_mode
        self._sort_keys = self._normalize_sort_keys(sort_keys)
        self.sfc_bits = int(sfc_bits)
        self._pq_args = dict(parquet_writer_args or {})
        self.global_extent = global_extent

        if max_parallel_files is None:
            cpu = max(1, multiprocessing.cpu_count())
            self.max_parallel_files = max(2, cpu // 2)
        else:
            self.max_parallel_files = max(1, int(max_parallel_files))

        self._buffers: Dict[str, List[pa.Table]] = defaultdict(list)

    # --------------------------- Public API ---------------------------

    def append(self, tile_id: str, table: pa.Table) -> None:
        if table is None or table.num_rows == 0:
            return
        if self.geom_col not in table.column_names:
            raise ValueError(f"WriterPool.append: missing geometry column '{self.geom_col}'")
        table = table.combine_chunks()
        table = ensure_large_types(table, self.geom_col)
        self._buffers[tile_id].append(table)

    def flush_all(self) -> None:
        if not self._buffers:
            logger.info("WriterPool.flush_all(): no buffered tiles to flush.")
            return

        os.makedirs(self.outdir, exist_ok=True)
        items = list(self._buffers.items())
        self._buffers.clear()

        total = len(items)
        mpf = min(self.max_parallel_files, total)
        rounds = math.ceil(total / mpf)
        logger.info(f"WriterPool.flush_all(): {total} tiles buffered → flushing in {rounds} round(s), "
                    f"{mpf} parallel writes per round.")

        def _finalize_one_tile(tile_id: str, batches: List[pa.Table]) -> str:
            logger.debug(f"[{tile_id}] Concatenating {len(batches)} batches.")
            full = pa.concat_tables(batches, promote=True)
            full = ensure_large_types(full, self.geom_col)
            bbox, full = self._maybe_sort_and_bbox(full)
            full = self._with_updated_geo_metadata(full, bbox)

            out_path = os.path.join(self.outdir, f"{tile_id}.parquet")
            logger.info(f"[{tile_id}] Writing to disk → {out_path}")
            pq.write_table(full, out_path, compression=self.compression, **self._pq_args)
            logger.debug(f"[{tile_id}] Flush complete, rows={full.num_rows}")
            return out_path

        for r in range(rounds):
            start = r * mpf
            batch = items[start : start + mpf]
            logger.info(f"WriterPool: round {r+1}/{rounds} — writing {len(batch)} tiles to disk.")
            if len(batch) == 1:
                tid, b = batch[0]
                _finalize_one_tile(tid, b)
                continue
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futs = {ex.submit(_finalize_one_tile, tid, b): tid for tid, b in batch}
                for f in as_completed(futs):
                    try:
                        _ = f.result()
                    except Exception as e:
                        logger.error(f"Error writing tile {futs[f]}: {e}")

        logger.info("WriterPool.flush_all(): all tiles successfully flushed to disk.")

    def close(self) -> None:
        self.flush_all()

    def set_sort_keys(self, sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]]) -> None:
        self._sort_keys = self._normalize_sort_keys(sort_keys)

    # ------------------------- Internal helpers -----------------------

    @staticmethod
    def _normalize_sort_keys(
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]]
    ) -> List[SortKey]:
        out: List[SortKey] = []
        if not sort_keys:
            return out
        for k in sort_keys:
            if isinstance(k, SortKey):
                out.append(k)
            elif isinstance(k, tuple):
                name, asc = k
                out.append(SortKey(str(name), bool(asc)))
            elif isinstance(k, str):
                out.append(SortKey(k, True))
            else:
                raise TypeError(f"Unsupported sort key type: {type(k)}")
        return out

    def _maybe_sort_and_bbox(self, tbl: pa.Table) -> Tuple[Tuple[float, float, float, float], pa.Table]:
        geoms = from_wkb(tbl[self.geom_col].to_numpy(zero_copy_only=False))

        minx = np.inf; miny = np.inf; maxx = -np.inf; maxy = -np.inf
        has_geom = False
        centers_x = []; centers_y = []

        for g in geoms:
            if g is None or g.is_empty:
                continue
            bxmin, bymin, bxmax, bymax = g.bounds
            minx, miny = min(minx, bxmin), min(miny, bymin)
            maxx, maxy = max(maxx, bxmax), max(maxy, bymax)
            c = g.centroid
            centers_x.append(float(c.x))
            centers_y.append(float(c.y))
            has_geom = True

        if not has_geom:
            bbox = (np.inf, np.inf, -np.inf, -np.inf)
            return bbox, tbl

        bbox = (float(minx), float(miny), float(maxx), float(maxy))

        if self.sort_mode == SortMode.NONE:
            return bbox, tbl

        if self.sort_mode == SortMode.COLUMNS:
            if not self._sort_keys:
                return bbox, tbl
            spec = [{"column": sk.column, "order": "ascending" if sk.ascending else "descending"}
                    for sk in self._sort_keys]
            logger.debug("Sorting by columns: %s", spec)
            return bbox, tbl.sort_by(spec)

        if self.sort_mode in (SortMode.ZORDER, SortMode.HILBERT):
            cx = np.asarray(centers_x, dtype=np.float64)
            cy = np.asarray(centers_y, dtype=np.float64)
            gxmin, gymin, gxmax, gymax = self.global_extent or bbox

            X = _scale_to_uint(cx, gxmin, gxmax, self.sfc_bits)
            Y = _scale_to_uint(cy, gymin, gymax, self.sfc_bits)
            z = _interleave_bits_2d(X, Y, self.sfc_bits)

            N = tbl.num_rows
            max_code = np.uint64((1 << (2 * min(self.sfc_bits, 31))) - 1)
            zfull = np.full(N, max_code, dtype=np.uint64)
            valid_idx = [i for i, g in enumerate(geoms) if g and not g.is_empty]
            if valid_idx:
                zfull[np.asarray(valid_idx, dtype=np.int64)] = z
            order = np.argsort(zfull, kind="mergesort")
            logger.debug(f"Sorting {N} rows by Z-order (sfc_bits={self.sfc_bits})")
            return bbox, tbl.take(pa.array(order, type=pa.int64()))

        return bbox, tbl

    def _with_updated_geo_metadata(
        self,
        tbl: pa.Table,
        bbox: Tuple[float, float, float, float],
    ) -> pa.Table:
        schema = tbl.schema
        meta = dict(schema.metadata or {})

        geo_raw = meta.get(b"geo")
        geo = {}
        if geo_raw is not None:
            try:
                geo = json.loads(geo_raw.decode("utf-8"))
            except Exception:
                pass

        geo.setdefault("version", "1.0.0")
        geo.setdefault("primary_column", self.geom_col)
        geo.setdefault("columns", {})
        col = geo["columns"].get(self.geom_col, {})
        col.setdefault("encoding", "WKB")
        col["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        geo["columns"][self.geom_col] = col
        geo["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

        meta[b"geo"] = json.dumps(geo).encode("utf-8")
        new_schema = schema.with_metadata(meta)
        return tbl.replace_schema_metadata(new_schema.metadata)