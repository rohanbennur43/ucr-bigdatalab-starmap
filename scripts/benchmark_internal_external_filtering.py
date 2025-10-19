#!/usr/bin/env python3
import sys, time, os
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box

# EPSG:4326  (xmin, ymin, xmax, ymax)
REGIONS = {
    # 1) Tiny → local
    "ucr_core":          (-117.35, 33.95, -117.30, 34.00),  # UCR campus-ish
    "riverside_city":    (-117.50, 33.85, -117.25, 34.05),

    # 2) Metro → sub-regional
    "riverside_county":  (-117.80, 33.35, -114.40, 34.90),
    "inland_empire":     (-117.90, 33.50, -116.40, 34.40),

    # 3) Regional → broader region
    "socal_core":        (-119.20, 32.40, -116.00, 34.60),  # SD+OC+LA east + IE core
    "socal_wide":        (-121.00, 32.20, -114.00, 35.00),  # adds Ventura/SB edges

    # 4) State
    "california":        (-124.50, 32.50, -114.10, 42.05),

    # 5) Multi-state (strictly growing footprints)
    "west_coast":        (-125.00, 32.00, -116.00, 49.50),  # CA/OR/WA coastal belt
    "far_west":          (-125.00, 31.00, -108.00, 49.50),  # CA/OR/WA/NV/AZ/UT

    # 6) Half-country+
    "western_us":        (-125.00, 31.00, -102.00, 49.50),

    # 7) National (contiguous → then everything)
    "lower48":           (-125.00, 24.00,  -66.50, 49.50),
    "usa_all_bbox":      (-179.50, 17.50,  -65.00, 71.80),  # adds AK/HI/PR (ocean-heavy)
}

def human_size(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB": return f"{n:.2f} {u}"
        n /= 1024

def count_rows_fast(path):
    pf = pq.ParquetFile(path)
    return sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))

def internal_pushdown(master_parquet, region, bbox, out_path, columns=None):
    t0 = time.time()
    gdf = gpd.read_parquet(master_parquet, bbox=bbox, columns=columns)
    t1 = time.time()
    gdf.to_parquet(out_path, engine="pyarrow", index=False, write_covering_bbox=True)
    t2 = time.time()
    return dict(
        mode="internal",
        region=region,
        rows=len(gdf),
        load_s=round(t1-t0,3),
        write_s=round(t2-t1,3),
        total_s=round(t2-t0,3),
        out=out_path,
        out_size=human_size(os.path.getsize(out_path))
    )

def external_filter(master_parquet, region, bbox, out_path, columns=None):
    xmin, ymin, xmax, ymax = bbox
    window = box(xmin, ymin, xmax, ymax)

    # full read (NO bbox pushdown)
    t0 = time.time()
    gdf = gpd.read_parquet(master_parquet, columns=columns)
    t1 = time.time()

    # spatial filter (use sindex if present)
    if hasattr(gdf, "sindex") and gdf.sindex is not None:
        idx = gdf.sindex.query(window, predicate="intersects")
        subset = gdf.iloc[idx]
    else:
        subset = gdf[gdf.intersects(window)]
    t2 = time.time()

    subset.to_parquet(out_path, engine="pyarrow", index=False, write_covering_bbox=True)
    t3 = time.time()

    return dict(
        mode="external",
        region=region,
        rows=len(subset),
        load_s=round(t1-t0,3),          # full file I/O + decode
        filter_s=round(t2-t1,3),        # spatial intersects time
        write_s=round(t3-t2,3),
        total_s=round(t3-t0,3),
        out=out_path,
        out_size=human_size(os.path.getsize(out_path))
    )

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python benchmark_compare.py master.parquet [geometry_only]")
        sys.exit(1)

    master = sys.argv[1]
    geometry_only = (len(sys.argv) == 3 and sys.argv[2] == "geometry_only")
    columns = ["geometry"] if geometry_only else None

    total_rows = count_rows_fast(master)
    print(f"[INFO] Master rows (fast): {total_rows:,}")

    results = []
    for name, bbox in REGIONS.items():
        # Internal pushdown
        out_int = f"{name}__internal.parquet"
        r_int = internal_pushdown(master, name, bbox, out_int, columns=columns)
        r_int["rows_total"] = total_rows
        results.append(r_int)

        # External filter
        out_ext = f"{name}__external.parquet"
        r_ext = external_filter(master, name, bbox, out_ext, columns=columns)
        r_ext["rows_total"] = total_rows
        results.append(r_ext)

    # Pretty print
    print("\nBenchmark results (internal vs external):")
    for r in results:
        if r["mode"] == "internal":
            print(f"[IN ] {r['region']:<14} rows {r['rows']:<10} "
                  f"load={r['load_s']:.3f}s write={r['write_s']:.3f}s total={r['total_s']:.3f}s  "
                  f"→ {r['out']} [{r['out_size']}]")
        else:
            print(f"[OUT] {r['region']:<14} rows {r['rows']:<10} "
                  f"load={r['load_s']:.3f}s filt={r.get('filter_s',0):.3f}s "
                  f"write={r['write_s']:.3f}s total={r['total_s']:.3f}s  "
                  f"→ {r['out']} [{r['out_size']}]")
