import sys
import time
import geopandas as gpd
from shapely.geometry import box

# bounding boxes for regions (xmin, ymin, xmax, ymax)
REGIONS = {
    "ucr": (-117.35, 33.95, -117.30, 34.00),
    "riverside": (-117.55, 33.85, -117.25, 34.05),
    "county": (-118.00, 33.40, -116.40, 34.20),
    "southern_ca": (-119.00, 32.50, -115.50, 34.50)
}

def extract_region(gdf, region_name, bbox, out_file):
    xmin, ymin, xmax, ymax = bbox
    window = box(xmin, ymin, xmax, ymax)

    t1 = time.time()
    subset = gdf[gdf.intersects(window)]
    t2 = time.time()

    subset.to_parquet(out_file, engine="pyarrow", index=False)
    t3 = time.time()

    stats = {
        "region": region_name,
        "rows_total": len(gdf),
        "rows_subset": len(subset),
        "filter_time_s": round(t2 - t1, 3),
        "write_time_s": round(t3 - t2, 3),
        "out_file": out_file
    }
    return stats

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_parquet.py master.parquet")
        sys.exit(1)

    master_file = sys.argv[1]

    t0 = time.time()
    gdf = gpd.read_parquet(master_file)
    t1 = time.time()

    all_stats = []
    for region, bbox in REGIONS.items():
        out_file = f"{region}.parquet"
        stats = extract_region(gdf, region, bbox, out_file)
        stats["load_time_s"] = round(t1 - t0, 3)
        stats["total_time_s"] = round(stats["load_time_s"] +
                                      stats["filter_time_s"] +
                                      stats["write_time_s"], 3)
        all_stats.append(stats)

    print("\nBenchmark results:")
    for s in all_stats:
        print(f"{s['region']}: rows {s['rows_subset']} / {s['rows_total']} "
              f"(load={s['load_time_s']}s, filter={s['filter_time_s']}s, "
              f"write={s['write_time_s']}s, total={s['total_time_s']}s)")
