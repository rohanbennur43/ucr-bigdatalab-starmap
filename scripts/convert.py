#!/usr/bin/env python3
import sys
import time
import statistics
import geopandas as gpd

def geojson_to_geoparquet(in_geojson, out_parquet, bbox=True):
    gdf = gpd.read_file(in_geojson)              # reads GeoJSON (via Fiona/GDAL)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf.to_parquet(out_parquet, engine="pyarrow",
                   write_covering_bbox=bool(bbox), index=False)

def geoparquet_to_geojson(in_parquet, out_geojson):
    gdf = gpd.read_parquet(in_parquet)           # GeoPandas restores Shapely geometries
    gdf.to_file(out_geojson, driver="GeoJSON")   # Save as GeoJSON

def bench(fn, repeat, *args, **kwargs):
    times = []
    for i in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"run {i+1}/{repeat}: {dt:.4f}s")
    if repeat > 1:
        print(f"avg: {statistics.mean(times):.4f}s  "
              f"min: {min(times):.4f}s  max: {max(times):.4f}s")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert.py [mode] input output [--repeat=N] [--bbox=1|0]")
        print("Modes: gpq (GeoJSON→GeoParquet), gjs (GeoParquet→GeoJSON)")
        sys.exit(1)

    mode, infile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
    repeat = 1
    bbox = 1 

    for tok in sys.argv[4:]:
        if tok.startswith("--repeat="):
            repeat = max(1, int(tok.split("=", 1)[1]))
        elif tok.startswith("--bbox="):
            bbox = int(tok.split("=", 1)[1])

    if mode == "gpq":
        print(f"GeoJSON → GeoParquet  (bbox={bbox}, repeat={repeat})")
        bench(geojson_to_geoparquet, repeat, infile, outfile, bbox=bool(bbox))
    elif mode == "gjs":
        print(f"GeoParquet → GeoJSON  (repeat={repeat})")
        bench(geoparquet_to_geojson, repeat, infile, outfile)
    else:
        print("Invalid mode. Use 'gpq' or 'gjs'.")
        sys.exit(1)
