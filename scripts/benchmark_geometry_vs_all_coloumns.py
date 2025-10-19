#!/usr/bin/env python3
# benchmark_geometry_vs_all.py
import sys, time, os
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box

# EPSG:4326 (xmin, ymin, xmax, ymax)
REGIONS = {
    "ucr_core":        (-117.35, 33.95, -117.30, 34.00),
    "riverside_city":  (-117.50, 33.85, -117.25, 34.05),
    "riverside_county":(-117.80, 33.35, -114.40, 34.90),
    "socal_core":      (-119.20, 32.40, -116.00, 34.60),
    "california":      (-124.50, 32.50, -114.10, 42.05),
    "lower48":         (-125.00, 24.00,  -66.50, 49.50),
}

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB": return f"{n:.2f} {u}"
        n /= 1024

def dataset_insights(path):
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    rgs  = pf.metadata.num_row_groups
    size = os.path.getsize(path)

    # Per-column compressed size (approx): sum across row groups
    col_sizes = {}
    for i in range(rgs):
        rg = pf.metadata.row_group(i)
        for c in range(rg.num_columns):
            col = rg.column(c)
            name = col.path_in_schema
            # compressed size may be None on some writers; fall back to uncompressed
            comp = getattr(col, "total_compressed_size", None)
            if comp is None:
                comp = getattr(col, "total_uncompressed_size", None) or 0
            col_sizes[name] = col_sizes.get(name, 0) + int(comp)

    # Sort top contributors
    top = sorted(col_sizes.items(), key=lambda kv: kv[1], reverse=True)

    print("\n[INSIGHTS] File:", path)
    print(f"  rows={rows:,}  row_groups={rgs:,}  file_size={human(size)}")
    print("  columns:", ", ".join([pf.schema_arrow.names[i] for i in range(len(pf.schema_arrow.names))]))
    print("  ~Per-column size share (compressed if available):")
    for name, bytes_ in top[:12]:
        pct = (bytes_ / size) * 100 if size else 0
        print(f"    - {name:<24} {human(bytes_):>10}  ({pct:5.1f}%)")

def bench_read(path, bbox, columns=None):
    t0 = time.time()
    gdf = gpd.read_parquet(path, bbox=bbox, columns=columns)
    t1 = time.time()
    return {
        "rows": len(gdf),
        "time_s": round(t1 - t0, 3),
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_geometry_vs_all.py path/to/osm_US_lower48_bb.parquet")
        sys.exit(1)

    master = sys.argv[1]
    dataset_insights(master)

    print("\n[BENCH] Internal filtering (bbox) — ALL columns vs GEOMETRY only")
    print("Region           | Rows       | All cols | Geometry-only | Speedup")
    print("-----------------+------------+----------+---------------+--------")
    for name, bbox in REGIONS.items():
        r_all = bench_read(master, bbox=bbox, columns=None)
        r_geom = bench_read(master, bbox=bbox, columns=["geometry"])
        speed = (r_all["time_s"] / r_geom["time_s"]) if r_geom["time_s"] > 0 else float("inf")
        print(f"{name:<16} | {r_all['rows']:>10,} | {r_all['time_s']:>7.3f}s | {r_geom['time_s']:>12.3f}s | ×{speed:>5.1f}")
