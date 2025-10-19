# # benchmark_parquet_internal.py
# import sys
# import time
# import geopandas as gpd
# import pyarrow.parquet as pq

# # bounding boxes for regions (xmin, ymin, xmax, ymax)
# REGIONS = {
#     "ucr": (-117.35, 33.95, -117.30, 34.00),
#     "riverside": (-117.55, 33.85, -117.25, 34.05),
#     "county": (-118.00, 33.40, -116.40, 34.20),
#     "southern_ca": (-119.00, 32.50, -115.50, 34.50),
# }

# def count_rows_fast(parquet_path: str) -> int:
#     pf = pq.ParquetFile(parquet_path)
#     return sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))

# def extract_region_internal(master_parquet, region_name, bbox, out_file, columns=None):
#     t1 = time.time()
#     subset = gpd.read_parquet(master_parquet, bbox=bbox, columns=columns)
#     t2 = time.time()

#     subset.to_parquet(out_file, engine="pyarrow", index=False)
#     t3 = time.time()

#     stats = {
#         "region": region_name,
#         "rows_subset": len(subset),
#         "filter_time_s": round(t2 - t1, 3),  # includes IO + pushdown read
#         "write_time_s": round(t3 - t2, 3),
#         "out_file": out_file,
#     }
#     return stats

# if __name__ == "__main__":
#     if not (2 <= len(sys.argv) <= 3):
#         print("Usage: python benchmark_parquet_internal.py master.parquet [geometry_only]")
#         print("  Pass 'geometry_only' to project only the geometry column.")
#         sys.exit(1)

#     master_file = sys.argv[1]
#     geometry_only = (len(sys.argv) == 3 and sys.argv[2] == "geometry_only")
#     columns = ["geometry"] if geometry_only else None

#     # total rows without full scan
#     total_rows = count_rows_fast(master_file)

#     all_stats = []
#     for region, bbox in REGIONS.items():
#         out_file = f"{region}.parquet"
#         stats = extract_region_internal(master_file, region, bbox, out_file, columns=columns)

#         stats["rows_total"] = total_rows
#         stats["load_time_s"] = 0.0
#         stats["total_time_s"] = round(stats["load_time_s"] + stats["filter_time_s"] + stats["write_time_s"], 3)
#         all_stats.append(stats)

#     print("\nBenchmark results (internal bbox filter):")
#     for s in all_stats:
#         print(f"{s['region']}: rows {s['rows_subset']} / {s['rows_total']} "
#               f"(load={s['load_time_s']}s, filter={s['filter_time_s']}s, "
#               f"write={s['write_time_s']}s, total={s['total_time_s']}s)")

#!/usr/bin/env python3
# benchmark_parquet_internal.py
import sys, time, os
import geopandas as gpd
import pyarrow.parquet as pq

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


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(num_bytes)
    for u in units:
        if s < 1024.0 or u == units[-1]:
            return f"{s:.2f} {u}"
        s /= 1024.0

def count_rows_fast(parquet_path: str) -> int:
    pf = pq.ParquetFile(parquet_path)
    return sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))

def extract_region_internal(master_parquet, region_name, bbox, out_file, columns=None):
    t0 = time.time()
    # read with bbox -> triggers pushdown if covering bboxes exist
    gdf = gpd.read_parquet(master_parquet, bbox=bbox, columns=columns)
    t1 = time.time()

    # write subset; include covering bbox in output
    gdf.to_parquet(out_file, engine="pyarrow", write_covering_bbox=True, index=False)
    t2 = time.time()

    size_bytes = os.path.getsize(out_file)
    stats = {
        "region": region_name,
        "rows_subset": len(gdf),
        "load_time_s": round(t1 - t0, 3),    # I/O + pushdown + decode
        "write_time_s": round(t2 - t1, 3),
        "total_time_s": round(t2 - t0, 3),
        "out_file": out_file,
        "out_size": size_bytes,
        "out_size_h": human_size(size_bytes),
    }
    return stats

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python benchmark_parquet_internal.py master.parquet [geometry_only]")
        print("  Pass 'geometry_only' to project only the geometry column.")
        sys.exit(1)

    master_file = sys.argv[1]
    geometry_only = (len(sys.argv) == 3 and sys.argv[2] == "geometry_only")
    columns = ["geometry"] if geometry_only else None

    total_rows = count_rows_fast(master_file)

    all_stats = []
    for region, bbox in REGIONS.items():
        out_file = f"{region}.parquet"
        s = extract_region_internal(master_file, region, bbox, out_file, columns=columns)
        s["rows_total"] = total_rows
        all_stats.append(s)

    print("\nBenchmark results (internal bbox filter):")
    for s in all_stats:
        print(
            f"{s['region']}: rows {s['rows_subset']} / {s['rows_total']} "
            f"(load={s['load_time_s']}s, write={s['write_time_s']}s, total={s['total_time_s']}s)  "
            f"→ {s['out_file']} [{s['out_size_h']}]"
        )
