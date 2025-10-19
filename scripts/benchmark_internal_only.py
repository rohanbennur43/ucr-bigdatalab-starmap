#!/usr/bin/env python3
# benchmark_internal_only.py
import sys, time, os
import geopandas as gpd
import pyarrow.parquet as pq

# EPSG:4326 (xmin, ymin, xmax, ymax)
REGIONS = {
    # 1) Tiny → local
    "ucr_core":        (-117.35, 33.95, -117.30, 34.00),
    "riverside_city":  (-117.50, 33.85, -117.25, 34.05),

    # 2) Metro → sub-regional
    "riverside_county":(-117.80, 33.35, -114.40, 34.90),
    "inland_empire":   (-117.90, 33.50, -116.40, 34.40),

    # 3) Regional → broader region
    "socal_core":      (-119.20, 32.40, -116.00, 34.60),
    "socal_wide":      (-121.00, 32.20, -114.00, 35.00),

    # 4) State
    "california":      (-124.50, 32.50, -114.10, 42.05),

    # 5) Multi-state
    "west_coast":      (-125.00, 32.00, -116.00, 49.50),
    "far_west":        (-125.00, 31.00, -108.00, 49.50),

    # 6) Half-country+
    "western_us":      (-125.00, 31.00, -102.00, 49.50),

    # 7) National
    "lower48":         (-125.00, 24.00,  -66.50, 49.50),
    "usa_all_bbox":    (-179.50, 17.50,  -65.00, 71.80),
}

def human_size(n):
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024 or u == "TB": return f"{n:.2f} {u}"
        n /= 1024

def count_rows_fast(path):
    pf = pq.ParquetFile(path)
    return sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))

def internal_pushdown(master_parquet, region, bbox, out_path, columns=None):
    # Read (with bbox pushdown + optional column projection)
    t0 = time.time()
    gdf = gpd.read_parquet(master_parquet, bbox=bbox, columns=columns)
    t1 = time.time()

    # Write (keep covering bbox so downstream pushdown still works)
    gdf.to_parquet(out_path, engine="pyarrow", index=False, write_covering_bbox=True)
    t2 = time.time()

    return dict(
        region=region,
        rows=len(gdf),
        read_s=round(t1 - t0, 3),
        write_s=round(t2 - t1, 3),
        total_s=round(t2 - t0, 3),
        out=out_path,
        out_size=human_size(os.path.getsize(out_path)),
    )

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python benchmark_internal_only.py master.parquet [geometry_only]")
        sys.exit(1)

    master = sys.argv[1]
    geometry_only = (len(sys.argv) == 3 and sys.argv[2] == "geometry_only")
    columns = ["geometry"] if geometry_only else None

    total_rows = count_rows_fast(master)
    print(f"[INFO] Master rows (fast): {total_rows:,}")
    print(f"[INFO] Columns: {'geometry only' if columns else 'all columns'}")

    results = []
    for name, bbox in REGIONS.items():
        out_int = f"{name}__internal.parquet"
        r = internal_pushdown(master, name, bbox, out_int, columns=columns)
        results.append(r)

    # Pretty print (one compact table)
    print("\nBenchmark results — Internal filtering only (read vs write):")
    print("Region           | Rows       | Read(s) | Write(s) | Total(s) | Output")
    print("-----------------+------------+---------+----------+----------+---------------------------")
    for r in results:
        print(f"{r['region']:<16} | {r['rows']:>10,} | {r['read_s']:>7.3f} | {r['write_s']:>8.3f} | {r['total_s']:>8.3f} | {r['out_size']:>9} → {r['out']}")
