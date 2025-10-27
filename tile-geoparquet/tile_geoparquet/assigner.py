# from typing import Dict, List, Optional
# import logging
# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.compute as pc
# from shapely import box, from_wkb
# from shapely.strtree import STRtree

# class TileAssignerFromCSV:
#     def __init__(self, index_csv_path: str, geom_col: str = "geometry", use_intersects: bool = False):
#         self.logger = logging.getLogger(__name__)
#         self.geom_col = geom_col
#         self.use_intersects = use_intersects

#         df = pd.read_csv(index_csv_path)
#         self.logger.info(f"Loaded index CSV '{index_csv_path}' with {len(df):,} rows")
#         required = {"xmin", "ymin", "xmax", "ymax"}
#         if not required.issubset(df.columns):
#             raise ValueError("index.csv must contain xmin,ymin,xmax,ymax columns")

#         if "File Name" in df.columns:
#             tile_ids = df["File Name"].astype(str).tolist()
#         elif "ID" in df.columns:
#             tile_ids = df["ID"].astype(str).tolist()
#         else:
#             tile_ids = df.index.astype(str).tolist()

#         # Keep the ORIGINAL lists (same length as CSV); STRtree indices align to this list.
#         polys = [box(xmin, ymin, xmax, ymax)
#                  for xmin, ymin, xmax, ymax in zip(df["xmin"], df["ymin"], df["xmax"], df["ymax"])]

#         self._polys = polys                 # list, length = len(df)
#         self._tile_ids = tile_ids           # list, same length
#         self._tree = STRtree(self._polys)   # may internally drop empties, but returned indices still refer to input order
#         self.logger.info(f"Built STRtree with {len(self._polys)} tile polygons")


#     def partition_by_tile(self, tbl: pa.Table) -> Dict[str, pa.Table]:
#         if self.geom_col not in tbl.column_names:
#             raise ValueError(f"Missing geometry column '{self.geom_col}'")
#         self.logger.debug(f"partition_by_tile called on table with {tbl.num_rows} rows")

#         # Normalize chunking; use index-based selection to avoid boolean mask issues
#         tbl = tbl.combine_chunks()

#         wkb_arr = tbl[self.geom_col].to_numpy(zero_copy_only=False)
#         geoms = from_wkb(wkb_arr)

#         # Candidate indices per geometry (indices refer to ORIGINAL input list passed to STRtree)
#         cands = self._tree.query(geoms, predicate="intersects")

#         chosen: List[Optional[str]] = [None] * len(geoms)
#         n_polys = len(self._polys)

#         for i, (g, idxs) in enumerate(zip(geoms, cands)):
#             if g is None or len(idxs) == 0:
#                 continue

#             picked: Optional[str] = None

#             # Helper to test candidate at position j
#             def try_pick(j: int, pred: str) -> Optional[str]:
#                 if j < 0 or j >= n_polys:
#                     return None
#                 poly = self._polys[j]
#                 if poly.is_empty:
#                     return None
#                 if pred == "contains":
#                     return self._tile_ids[j] if poly.contains(g) else None
#                 else:  # intersects
#                     return self._tile_ids[j] if poly.intersects(g) else None

#             if self.use_intersects:
#                 for j in idxs:
#                     picked = try_pick(int(j), "intersects")
#                     if picked is not None:
#                         break
#             else:
#                 # Prefer contains
#                 for j in idxs:
#                     picked = try_pick(int(j), "contains")
#                     if picked is not None:
#                         break
#                 # Fallback to intersects
#                 if picked is None:
#                     for j in idxs:
#                         picked = try_pick(int(j), "intersects")
#                         if picked is not None:
#                             break

#             chosen[i] = picked

#         # Build per-tile subtables using row indices (robust to chunking/nulls)
#         parts: Dict[str, pa.Table] = {}
#         chosen_np = np.asarray(chosen, dtype=object)
#         valid = chosen_np != None
#         if not valid.any():
#             self.logger.debug("No geometries assigned to tiles in this batch")
#             return parts

#         tiles = np.unique(chosen_np[valid])
#         row_indices = np.arange(len(chosen_np))
#         for t in tiles:
#             idx = row_indices[chosen_np == t]
#             if idx.size:
#                 sub = tbl.take(pa.array(idx, type=pa.int64()))
#                 if sub.num_rows:
#                     parts[str(t)] = sub
#         self.logger.debug(f"partition_by_tile returning {len(parts)} parts")
#         self.logger.debug(f"Tiles assigned in this batch: {list(parts.keys())}")
#         return parts

# tile_geoparquet/assigner.py
from typing import Dict, List, Optional
import json
import logging

import numpy as np
import pandas as pd
import pyarrow as pa

from shapely import box, from_wkb, from_wkt

logger = logging.getLogger(__name__)

def _detect_geom_in_schema(schema: pa.Schema, default: str = "geometry"):
    """Read GeoParquet metadata to get geometry column + encoding."""
    md = schema.metadata or {}
    geo = md.get(b"geo")
    if not geo:
        return default, "WKB"
    j = json.loads(geo.decode())
    col = j.get("primary_column") or default
    enc = j.get("columns", {}).get(col, {}).get("encoding", "WKB")
    return col, enc.upper()

class TileAssignerFromCSV:
    """
    Brute-force tile assigner (no STRtree):
      - For each feature geometry, iterate all tiles and pick first that matches.
      - Default policy: contains-first, then intersects fallback (toggle with use_intersects=True).
    Assumptions:
      - Same CRS for data and tiles.
      - Geometry column stores WKB or WKT; column name auto-detected from GeoParquet metadata if possible.
    """
    def __init__(self, index_csv_path: str, geom_col: str = "geometry", use_intersects: bool = False):
        self.use_intersects = use_intersects
        self.geom_col = geom_col  # may be overridden at first call when schema is known
        self._geom_encoding: Optional[str] = None  # "WKB" or "WKT" (set later)

        df = pd.read_csv(index_csv_path)
        required = {"xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(df.columns):
            raise ValueError("index.csv must contain xmin,ymin,xmax,ymax columns")

        if "File Name" in df.columns:
            tile_ids = df["File Name"].astype(str).tolist()
        elif "ID" in df.columns:
            tile_ids = df["ID"].astype(str).tolist()
        else:
            tile_ids = df.index.astype(str).tolist()

        # Keep tiles as simple parallel lists (same order as CSV)
        polys: List = []
        tids: List[str] = []
        bad = 0
        for xmin, ymin, xmax, ymax, tid in zip(df["xmin"], df["ymin"], df["xmax"], df["ymax"], tile_ids):
            if not np.isfinite([xmin, ymin, xmax, ymax]).all():
                bad += 1; continue
            if xmin >= xmax or ymin >= ymax:
                bad += 1; continue
            polys.append(box(float(xmin), float(ymin), float(xmax), float(ymax)))
            tids.append(str(tid))

        if not polys:
            raise ValueError("No valid tiles in index.csv after filtering invalid rows")

        self._polys = polys
        self._tile_ids = tids
        self._tile_bounds = {tid: p.bounds for tid, p in zip(self._tile_ids, self._polys)}

        logger.info("Loaded index CSV '%s' with %d tiles (%d invalid rows skipped)",
                    index_csv_path, len(self._polys), bad)

    def _ensure_geom_info(self, schema: pa.Schema):
        # Detect geometry column + encoding once from the source schema
        if self._geom_encoding is None:
            detected_col, enc = _detect_geom_in_schema(schema, default=self.geom_col)
            self.geom_col = detected_col
            self._geom_encoding = enc
            logger.info("Geometry column detected: '%s' (encoding=%s)", self.geom_col, self._geom_encoding)

    def tile_bbox(self, tile_id: str):
        return self._tile_bounds[str(tile_id)]  # (xmin, ymin, xmax, ymax)

    def partition_by_tile(self, tbl: pa.Table) -> Dict[str, pa.Table]:
        """Return {tile_id: pa.Table} for the given batch using brute-force matching."""
        # Normalize chunking first
        tbl = tbl.combine_chunks()

        # Detect geometry column + encoding from schema on first call
        self._ensure_geom_info(tbl.schema)

        if self.geom_col not in tbl.column_names:
            raise ValueError(f"Missing geometry column '{self.geom_col}'")

        col = tbl[self.geom_col]
        enc = self._geom_encoding or "WKB"

        # Decode geometries
        if enc == "WKT":
            arr = col.to_numpy(zero_copy_only=False)
            geoms = from_wkt(arr)
        else:  # WKB
            arr = col.to_numpy(zero_copy_only=False)
            geoms = from_wkb(arr)

        n = len(geoms)
        chosen: List[Optional[str]] = [None] * n
        n_tiles = len(self._polys)
        # Stats
        null_empty = 0
        no_match_contains = 0
        no_match_any = 0
        matched_contains = 0
        matched_intersects = 0

        # Precompute union bbox of all tiles for a fast outside check (optional)
        tiles_minx = min(p.bounds[0] for p in self._polys)
        tiles_miny = min(p.bounds[1] for p in self._polys)
        tiles_maxx = max(p.bounds[2] for p in self._polys)
        tiles_maxy = max(p.bounds[3] for p in self._polys)

        # Brute-force assignment
        # Policy: contains-first then intersects (unless use_intersects=True)
# Brute-force assignment (STRICT CONTAINS ONLY)
        for i in range(n):
            g = geoms[i]
            if g is None or g.is_empty:
                null_empty += 1
                continue

            # Fast reject if completely outside tiles’ union bbox
            gx0, gy0, gx1, gy1 = g.bounds
            if gx1 < tiles_minx or gx0 > tiles_maxx or gy1 < tiles_miny or gy0 > tiles_maxy:
                no_match_any += 1
                continue

            picked: Optional[str] = None
            for j in range(n_tiles):
                p = self._polys[j]
                if p.is_empty:
                    continue
                if p.contains(g):                # <- only contains
                    picked = self._tile_ids[j]
                    matched_contains += 1
                    break

            if picked is None:
                no_match_contains += 1
                no_match_any += 1

            chosen[i] = picked



        # Group rows by chosen tile using indices + take()
        parts: Dict[str, pa.Table] = {}
        chosen_np = np.asarray(chosen, dtype=object)
        valid = chosen_np != None
        if not valid.any():
            logger.info("batch rows=%d, assigned=0 (no matches)", n)
            return parts

        tiles = np.unique(chosen_np[valid])
        row_indices = np.arange(n, dtype=np.int64)
        assigned_total = 0
        for t in tiles:
            idx = row_indices[chosen_np == t]
            if idx.size:
                sub = tbl.take(pa.array(idx))
                if sub.num_rows:
                    parts[str(t)] = sub
                    assigned_total += sub.num_rows

        logger.info(
            "batch rows=%d, assigned=%d, null_empty=%d, no_match_contains=%d, no_match_any=%d, matched_contains=%d, matched_intersects=%d",
            n,
            int((np.asarray([c is not None for c in chosen])).sum()),
            null_empty, no_match_contains, no_match_any, matched_contains, matched_intersects
        )

        return parts
