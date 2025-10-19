# #!/usr/bin/env python3
# import sys
# import time
# import statistics
# import geopandas as gpd

# def geojson_to_geoparquet(in_geojson, out_parquet, bbox=True):
#     print(f"[INFO] Reading GeoJSON: {in_geojson}")
#     t0 = time.perf_counter()
#     gdf = gpd.read_file(in_geojson)
#     t1 = time.perf_counter()
#     print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

#     if gdf.crs is None:
#         gdf.set_crs(epsg=4326, inplace=True)
#         print("[INFO] CRS not found — defaulted to EPSG:4326")

#     print(f"[INFO] Writing GeoParquet: {out_parquet} (write_covering_bbox={bbox})")
#     t2 = time.perf_counter()
#     gdf.to_parquet(out_parquet, engine="pyarrow",
#                    write_covering_bbox=bool(bbox), index=False)
#     t3 = time.perf_counter()
#     print(f"[INFO] Wrote GeoParquet in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")

# def geoparquet_to_geojson(in_parquet, out_geojson):
#     print(f"[INFO] Reading GeoParquet: {in_parquet}")
#     t0 = time.perf_counter()
#     gdf = gpd.read_parquet(in_parquet)
#     t1 = time.perf_counter()
#     print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

#     print(f"[INFO] Writing GeoJSON: {out_geojson}")
#     t2 = time.perf_counter()
#     gdf.to_file(out_geojson, driver="GeoJSON")
#     t3 = time.perf_counter()
#     print(f"[INFO] Wrote GeoJSON in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")

# def geoparquet_to_geoparquet(in_parquet, out_parquet, bbox=True, keep_cols=None):
#     import pandas as pd
#     print(f"[INFO] Reading GeoParquet: {in_parquet}")
#     t0 = time.perf_counter()
#     gdf = gpd.read_parquet(in_parquet)  # loads geometry properly
#     t1 = time.perf_counter()
#     print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

#     if keep_cols:
#         keep = [c for c in keep_cols if c in gdf.columns] + ["geometry"]
#         gdf = gdf[keep]
#         print(f"[INFO] Keeping columns: {keep}")

#     # preserve/ensure CRS just in case
#     if gdf.crs is None:
#         gdf.set_crs(epsg=4326, inplace=True)
#         print("[INFO] CRS missing — set to EPSG:4326")

#     print(f"[INFO] Writing GeoParquet: {out_parquet} (write_covering_bbox={bbox})")
#     t2 = time.perf_counter()
#     gdf.to_parquet(
#         out_parquet,
#         engine="pyarrow",
#         write_covering_bbox=bool(bbox),
#         index=False,
#     )
#     t3 = time.perf_counter()
#     print(f"[INFO] Wrote GeoParquet in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")


# def bench(fn, repeat, *args, **kwargs):
#     times = []
#     for i in range(repeat):
#         print(f"\n[RUN {i+1}/{repeat}] --------------------------")
#         t0 = time.perf_counter()
#         fn(*args, **kwargs)
#         t1 = time.perf_counter()
#         dt = t1 - t0
#         times.append(dt)
#         print(f"[INFO] Run {i+1}/{repeat} complete in {dt:.3f}s")

#     if repeat > 1:
#         avg = statistics.mean(times)
#         print(f"\n[SUMMARY] avg={avg:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s")

# if __name__ == "__main__":
#     if len(sys.argv) < 4:
#         print("Usage: python convert.py [mode] input output [--repeat=N] [--bbox=1|0]")
#         print("Modes: gpq (GeoJSON→GeoParquet), gjs (GeoParquet→GeoJSON)")
#         sys.exit(1)

#     mode, infile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
#     repeat = 1
#     bbox = 1 

#     for tok in sys.argv[4:]:
#         if tok.startswith("--repeat="):
#             repeat = max(1, int(tok.split("=", 1)[1]))
#         elif tok.startswith("--bbox="):
#             bbox = int(tok.split("=", 1)[1])

#     print(f"[START] Mode: {mode.upper()} | Input: {infile} | Output: {outfile} | Repeat: {repeat} | BBox: {bbox}\n")

#     if mode == "gpq":
#         print(f"[INFO] Starting GeoJSON → GeoParquet conversion (write_covering_bbox={bbox})")
#         bench(geojson_to_geoparquet, repeat, infile, outfile, bbox=bool(bbox))
#     elif mode == "gjs":
#         print(f"[INFO] Starting GeoParquet → GeoJSON conversion")
#         bench(geoparquet_to_geojson, repeat, infile, outfile)
#     elif mode == "pp":
#         print(f"[INFO] Starting GeoParquet → GeoParquet rewrite (write_covering_bbox={bbox})")
#         bench(geoparquet_to_geoparquet, repeat, infile, outfile, bbox=bool(bbox), keep_cols=None)
#     else:
#         print("Invalid mode. Use 'gpq' or 'gjs'.")
#         sys.exit(1)


#!/usr/bin/env python3
import sys, time, statistics, argparse
import numpy as np
import geopandas as gpd

# ---------------- Spatial keys ----------------

def _norm_lonlat_to_grid(lon, lat, bits):
    # Normalize lon∈[-180,180], lat∈[-90,90] to integers in [0, 2^bits - 1]
    n = (1 << bits) - 1
    x = ((lon + 180.0) / 360.0) * n
    y = ((lat + 90.0)  / 180.0) * n
    xi = np.clip(x.round().astype(np.uint64), 0, n)
    yi = np.clip(y.round().astype(np.uint64), 0, n)
    return xi, yi

def _morton_interleave_u32(x, y):
    # Interleave 32-bit x,y -> 64-bit Morton (Z-order). Works for bits<=32.
    def _split_by_1(v):
        v &= 0x00000000ffffffff
        v = (v | (v << 16)) & 0x0000ffff0000ffff
        v = (v | (v << 8))  & 0x00ff00ff00ff00ff
        v = (v | (v << 4))  & 0x0f0f0f0f0f0f0f0f
        v = (v | (v << 2))  & 0x3333333333333333
        v = (v | (v << 1))  & 0x5555555555555555
        return v
    return (_split_by_1(y) << 1) | _split_by_1(x)

def _hilbert_xy_to_index(x, y, bits):
    x = x.astype(np.uint64).copy()
    y = y.astype(np.uint64).copy()
    d = np.zeros_like(x, dtype=np.uint64)

    for i in range(bits - 1, -1, -1):
        rx = (x >> i) & 1
        ry = (y >> i) & 1

        d |= (((3 * rx) ^ ry) << (2 * i))

        lowmask = (np.uint64(1) << i) - 1
        xl = x & lowmask
        yl = y & lowmask

        swap = (ry == 0)
        reflect = swap & (rx == 1)

        xl = np.where(reflect, lowmask - xl, xl)
        yl = np.where(reflect, lowmask - yl, yl)

        xt = np.where(swap, yl, xl)
        yt = np.where(swap, xl, yl)

        x = (x & ~lowmask) | xt
        y = (y & ~lowmask) | yt

    return d

def _centroid_xy(g):
    # For points, use coords; for others, use representative_point (robust).
    geom = g.geometry
    is_point = geom.geom_type == "Point"
    x = np.empty(len(g), dtype=np.float64)
    y = np.empty(len(g), dtype=np.float64)
    if is_point.any():
        pts = geom[is_point]
        x[is_point] = pts.x.values
        y[is_point] = pts.y.values
    if (~is_point).any():
        rp = geom[~is_point].representative_point()
        x[~is_point] = rp.x.values
        y[~is_point] = rp.y.values
    return x, y

def add_spatial_key(gdf, order="z", bits=24, colname=None):
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326, allow_override=True)
    x, y = _centroid_xy(gdf)
    xi, yi = _norm_lonlat_to_grid(x, y, bits)
    if order == "z":
        key = _morton_interleave_u32(xi, yi)
        cname = colname or f"zkey{bits}"
    elif order == "hilbert":
        key = _hilbert_xy_to_index(xi, yi, bits)
        cname = colname or f"hkey{bits}"
    else:
        raise ValueError("order must be 'z' or 'hilbert'")
    gdf[cname] = key
    return gdf, cname

# ---------------- Converters ----------------

def geojson_to_geoparquet(in_geojson, out_parquet, bbox=True):
    print(f"[INFO] Reading GeoJSON: {in_geojson}")
    t0 = time.perf_counter()
    gdf = gpd.read_file(in_geojson)
    t1 = time.perf_counter()
    print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
        print("[INFO] CRS not found — defaulted to EPSG:4326")

    print(f"[INFO] Writing GeoParquet: {out_parquet} (write_covering_bbox={bbox})")
    t2 = time.perf_counter()
    gdf.to_parquet(out_parquet, engine="pyarrow",
                   write_covering_bbox=bool(bbox), index=False)
    t3 = time.perf_counter()
    print(f"[INFO] Wrote GeoParquet in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")

def geoparquet_to_geojson(in_parquet, out_geojson):
    print(f"[INFO] Reading GeoParquet: {in_parquet}")
    t0 = time.perf_counter()
    gdf = gpd.read_parquet(in_parquet)
    t1 = time.perf_counter()
    print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

    print(f"[INFO] Writing GeoJSON: {out_geojson}")
    t2 = time.perf_counter()
    gdf.to_file(out_geojson, driver="GeoJSON")
    t3 = time.perf_counter()
    print(f"[INFO] Wrote GeoJSON in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")

def geoparquet_to_geoparquet(in_parquet, out_parquet, bbox=True, keep_cols=None,
                             order=None, bits=24, row_group_size=None):
    import pyarrow.parquet as pq

    print(f"[INFO] Reading GeoParquet: {in_parquet}")
    t0 = time.perf_counter()
    gdf = gpd.read_parquet(in_parquet)
    t1 = time.perf_counter()
    print(f"[INFO] Loaded {len(gdf):,} rows in {t1 - t0:.2f}s")

    if keep_cols:
        keep = [c for c in keep_cols if c in gdf.columns] + ["geometry"]
        gdf = gdf[keep]
        print(f"[INFO] Keeping columns: {keep}")

    if order:
        print(f"[INFO] Adding spatial key and sorting by {order} (bits={bits}) …")
        gdf, keycol = add_spatial_key(gdf, order=order, bits=bits)
        gdf = gdf.sort_values(keycol, kind="stable")
        # Optionally drop the key column after sorting:
        # gdf = gdf.drop(columns=[keycol])
        print(f"[INFO] Sorted by {keycol}")

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
        print("[INFO] CRS missing — set to EPSG:4326")

    print(f"[INFO] Writing GeoParquet: {out_parquet} (write_covering_bbox={bbox})")
    t2 = time.perf_counter()
    to_kw = dict(engine="pyarrow", write_covering_bbox=bool(bbox), index=False)
    if row_group_size:
        # Use small-ish groups to preserve locality; e.g., 128k–512k
        to_kw["row_group_size"] = int(row_group_size)
    gdf.to_parquet(out_parquet, **to_kw)
    t3 = time.perf_counter()
    print(f"[INFO] Wrote GeoParquet in {t3 - t2:.2f}s (total {t3 - t0:.2f}s)\n")

# ---------------- Bench wrapper ----------------

def bench(fn, repeat, *args, **kwargs):
    times = []
    for i in range(repeat):
        print(f"\n[RUN {i+1}/{repeat}] --------------------------")
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"[INFO] Run {i+1}/{repeat} complete in {dt:.3f}s")
    if repeat > 1:
        avg = statistics.mean(times)
        print(f"\n[SUMMARY] avg={avg:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s")

# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GeoJSON/GeoParquet converters with spatial ordering.")
    ap.add_argument("mode", choices=["gpq","gjs","pp"])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--bbox", type=int, default=1, help="write_covering_bbox (1/0)")
    ap.add_argument("--keep", help="comma-separated columns to keep (pp mode)")
    ap.add_argument("--order", choices=["z","hilbert"], help="spatial sort key (pp mode)")
    ap.add_argument("--bits", type=int, default=24, help="key bit-depth (<=32 recommended)")
    ap.add_argument("--row-group-size", type=int, help="Parquet row group size (e.g., 262144)")

    args = ap.parse_args()
    keep_cols = args.keep.split(",") if args.keep else None

    print(f"[START] Mode: {args.mode.upper()} | Input: {args.input} | Output: {args.output} | "
          f"Repeat: {args.repeat} | BBox: {args.bbox} | Order: {args.order} | Bits: {args.bits} | "
          f"RowGroup: {args.row_group_size}\n")

    if args.mode == "gpq":
        bench(geojson_to_geoparquet, args.repeat, args.input, args.output, bbox=bool(args.bbox))
    elif args.mode == "gjs":
        bench(geoparquet_to_geojson, args.repeat, args.input, args.output)
    elif args.mode == "pp":
        bench(geoparquet_to_geoparquet, args.repeat, args.input, args.output,
              bbox=bool(args.bbox), keep_cols=keep_cols,
              order=args.order, bits=args.bits, row_group_size=args.row_group_size)
