# benchmark_parquet_internal.py
import sys
import time
import geopandas as gpd
import pyarrow.parquet as pq

# bounding boxes for regions (xmin, ymin, xmax, ymax)
REGIONS = {
    "ucr": (-117.35, 33.95, -117.30, 34.00),
    "riverside": (-117.55, 33.85, -117.25, 34.05),
    "county": (-118.00, 33.40, -116.40, 34.20),
    "southern_ca": (-119.00, 32.50, -115.50, 34.50),
}

def count_rows_fast(parquet_path: str) -> int:
    pf = pq.ParquetFile(parquet_path)
    return sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))

def extract_region_internal(master_parquet, region_name, bbox, out_file, columns=None):
    t1 = time.time()
    subset = gpd.read_parquet(master_parquet, bbox=bbox, columns=columns)
    t2 = time.time()

    subset.to_parquet(out_file, engine="pyarrow", index=False)
    t3 = time.time()

    stats = {
        "region": region_name,
        "rows_subset": len(subset),
        "filter_time_s": round(t2 - t1, 3),  # includes IO + pushdown read
        "write_time_s": round(t3 - t2, 3),
        "out_file": out_file,
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

    # total rows without full scan
    total_rows = count_rows_fast(master_file)

    all_stats = []
    for region, bbox in REGIONS.items():
        out_file = f"{region}.parquet"
        stats = extract_region_internal(master_file, region, bbox, out_file, columns=columns)

        stats["rows_total"] = total_rows
        stats["load_time_s"] = 0.0
        stats["total_time_s"] = round(stats["load_time_s"] + stats["filter_time_s"] + stats["write_time_s"], 3)
        all_stats.append(stats)

    print("\nBenchmark results (internal bbox filter):")
    for s in all_stats:
        print(f"{s['region']}: rows {s['rows_subset']} / {s['rows_total']} "
              f"(load={s['load_time_s']}s, filter={s['filter_time_s']}s, "
              f"write={s['write_time_s']}s, total={s['total_time_s']}s)")
