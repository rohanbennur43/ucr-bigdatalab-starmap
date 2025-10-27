#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GeoParquet bbox from metadata vs bbox computed from all geometries.

Usage:
  python compare_bbox.py path/to/file.parquet
  python compare_bbox.py path/to/dir_with_tiles  # will scan *.parquet
"""
import sys, json
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb, from_wkt
import folium

GeomBBox = Tuple[float, float, float, float]

# ---------- helpers ----------
def get_geo_meta_bbox(schema: pa.Schema) -> Tuple[Optional[str], str, Optional[GeomBBox]]:
    """
    Returns (primary_col, enc, bbox) from the GeoParquet metadata if present.
    enc is 'WKB' or 'WKT'. bbox may be None if missing.
    """
    md = schema.metadata or {}
    raw = md.get(b"geo")
    if not raw:
        return "geometry", "WKB", None
    try:
        j = json.loads(raw.decode("utf-8"))
        col = j.get("primary_column") or "geometry"
        enc = (j.get("columns", {}).get(col, {}).get("encoding", "WKB") or "WKB").upper()
        bbox = j.get("columns", {}).get(col, {}).get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return col, enc, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        return col, enc, None
    except Exception:
        return "geometry", "WKB", None

def iter_geoms(pf: pq.ParquetFile, geom_col: str, enc: str) -> Iterator:
    """Yield Shapely geometries from the parquet file, reading only the geometry column."""
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[geom_col]).combine_chunks()
        arr = tbl[geom_col].to_numpy(zero_copy_only=False)
        if enc == "WKT":
            geoms = from_wkt(arr)
        else:
            geoms = from_wkb(arr)
        for g in geoms:
            yield g

def compute_bbox(geoms: Iterator) -> Optional[GeomBBox]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    any_geom = False
    for g in geoms:
        if g is None or g.is_empty:
            continue
        any_geom = True
        x0, y0, x1, y1 = g.bounds
        if x0 < minx: minx = x0
        if y0 < miny: miny = y0
        if x1 > maxx: maxx = x1
        if y1 > maxy: maxy = y1
    return (minx, miny, maxx, maxy) if any_geom else None

def ensure_geom_col(schema: pa.Schema, primary: Optional[str]) -> Optional[str]:
    """Return a geometry column name present in schema (try primary, then common fallbacks)."""
    names = schema.names
    cands = [primary, "geometry", "wkb_geometry", "geom", "GEOMETRY"]
    for c in cands:
        if c and c in names:
            return c
    return None

def add_rect(m: folium.Map, bbox: GeomBBox, tooltip: str, color: str, dash: bool = False):
    (xmin, ymin, xmax, ymax) = bbox
    folium.Rectangle(
        bounds=[[ymin, xmin], [ymax, xmax]],
        fill=False,
        color=color,
        weight=2,
        dash_array="6,4" if dash else None,
        tooltip=tooltip,
    ).add_to(m)

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_bbox.py <file.parquet | dir>")
        sys.exit(1)

    src = Path(sys.argv[1])
    files: List[Path] = []
    if src.is_dir():
        files = sorted([p for p in src.glob("*.parquet") if p.is_file()])
    elif src.suffix.lower() == ".parquet":
        files = [src]
    else:
        print("Provide a .parquet file or a directory containing .parquet tiles.")
        sys.exit(1)

    if not files:
        print("No parquet files found.")
        sys.exit(0)

    # Collect boxes for map fit
    global_minx = global_miny = float("inf")
    global_maxx = global_maxy = float("-inf")

    # Create map (will fit after we know bounds)
    m = folium.Map(tiles="CartoDB positron")

    for p in files:
        try:
            pf = pq.ParquetFile(p)
            schema = pf.schema_arrow
        except Exception as e:
            print(f"Skip {p.name}: cannot open ({e})")
            continue

        primary, enc, meta_bbox = get_geo_meta_bbox(schema)
        geom_col = ensure_geom_col(schema, primary)
        if not geom_col:
            print(f"Skip {p.name}: no geometry column found")
            continue

        comp_bbox = compute_bbox(iter_geoms(pf, geom_col, enc))

        # Update global bounds with whatever we have
        for b in [meta_bbox, comp_bbox]:
            if b is None:
                continue
            xmin, ymin, xmax, ymax = b
            global_minx = min(global_minx, xmin)
            global_miny = min(global_miny, ymin)
            global_maxx = max(global_maxx, xmax)
            global_maxy = max(global_maxy, ymax)

        # Visualize both boxes if available
        nrows = pf.metadata.num_rows if pf.metadata else None
        if meta_bbox:
            add_rect(
                m, meta_bbox,
                tooltip=f"{p.name} [meta bbox] rows={nrows}",
                color="blue",
                dash=True,
            )
        if comp_bbox:
            add_rect(
                m, comp_bbox,
                tooltip=f"{p.name} [computed bbox] rows={nrows}",
                color="red",
                dash=False,
            )

        # If both exist, add a small popup diff summary at center of computed bbox
        if meta_bbox and comp_bbox:
            mx = (comp_bbox[0] + comp_bbox[2]) / 2
            my = (comp_bbox[1] + comp_bbox[3]) / 2
        if meta_bbox and comp_bbox:
            mx = (comp_bbox[0] + comp_bbox[2]) / 2
            my = (comp_bbox[1] + comp_bbox[3]) / 2
            dx = tuple(abs(c - m) for c, m in zip(comp_bbox, meta_bbox))
            diff_html = (
                f"<b>{p.name}</b><br>"
                f"geom_col: {geom_col}, enc: {enc}<br>"
                f"meta bbox: {meta_bbox}<br>"
                f"comp bbox: {comp_bbox}<br>"
                f"deltas: {dx}"
            )
            folium.Marker([my, mx], tooltip=p.name, popup=diff_html).add_to(m)

    if global_minx == float("inf"):
        print("No valid bboxes found to display.")
        sys.exit(0)

    m.fit_bounds([[global_miny, global_minx], [global_maxy, global_maxx]])
    out = "compare_bboxes.html"
    m.save(out)
    print(f"Wrote {out} with {len(files)} file(s).")
    print("Legend: BLUE dashed = metadata bbox, RED solid = computed bbox")

if __name__ == "__main__":
    main()
