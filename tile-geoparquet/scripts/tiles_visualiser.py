#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import json

import pyarrow.parquet as pq
import pyarrow as pa
from shapely import from_wkb, from_wkt
import folium

TILES_DIR = "./tiles-out"
OUT_HTML = "tiles_bboxes.html"

def detect_geom(schema: pa.Schema, default="geometry"):
    md = schema.metadata or {}
    geo = md.get(b"geo")
    if not geo:
        return default, "WKB"
    j = json.loads(geo.decode())
    col = j.get("primary_column") or default
    enc = j.get("columns", {}).get(col, {}).get("encoding", "WKB")
    return col, enc.upper()

def file_bounds(geoms_iter):
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    any_geom = False
    for g in geoms_iter:
        if g is None or g.is_empty:
            continue
        any_geom = True
        x0, y0, x1, y1 = g.bounds
        if x0 < minx: minx = x0
        if y0 < miny: miny = y0
        if x1 > maxx: maxx = x1
        if y1 > maxy: maxy = y1
    if not any_geom:
        return None
    return (minx, miny, maxx, maxy)

def iter_file_geoms(pf: pq.ParquetFile, geom_col: str, encoding: str):
    # stream by row group, geometry column only
    for rg in range(pf.num_row_groups):
        col_tbl = pf.read_row_group(rg, columns=[geom_col]).combine_chunks()
        arr = col_tbl[geom_col].to_numpy(zero_copy_only=False)
        if encoding == "WKT":
            geoms = from_wkt(arr)
        else:
            geoms = from_wkb(arr)
        for g in geoms:
            yield g

def main():
    tiles = sorted(Path(TILES_DIR).glob("*.parquet"))
    if not tiles:
        print(f"No .parquet files found in {TILES_DIR}")
        return

    boxes = []  # (path, (xmin,ymin,xmax,ymax), nrows)
    global_min = [float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf")]

    for p in tiles:
        pf = pq.ParquetFile(p)
        schema = pf.schema_arrow
        geom_col, enc = detect_geom(schema)
        if geom_col not in schema.names:
            print(f"Skip {p.name}: geometry column '{geom_col}' not found")
            continue

        b = file_bounds(iter_file_geoms(pf, geom_col, enc))
        if b is None:
            print(f"Skip {p.name}: no valid geometries")
            continue

        xmin, ymin, xmax, ymax = b
        global_min[0] = min(global_min[0], xmin)
        global_min[1] = min(global_min[1], ymin)
        global_max[0] = max(global_max[0], xmax)
        global_max[1] = max(global_max[1], ymax)
        nrows = pf.metadata.num_rows if pf.metadata is not None else None
        boxes.append((p, b, nrows))

    if not boxes:
        print("No boxes to display.")
        return

    # Build folium map
    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[global_min[1], global_min[0]],[global_max[1], global_max[0]]])

    for p, (xmin, ymin, xmax, ymax), nrows in boxes:
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            fill=False,
            weight=2,
            tooltip=f"{p.name}",
            popup=f"{p.name}\nrows={nrows}",
        ).add_to(m)

    m.save(OUT_HTML)
    print(f"Wrote {OUT_HTML} with {len(boxes)} rectangles.")

if __name__ == "__main__":
    main()
