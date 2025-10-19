#!/usr/bin/env python3
import argparse, os, sys, json
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely import wkt as shapely_wkt

def load_region(args) -> gpd.GeoSeries:
    if args.bbox:
        minx, miny, maxx, maxy = map(float, args.bbox.split(","))
        geom = box(minx, miny, maxx, maxy)
        return gpd.GeoSeries([geom], crs=args.crs or "EPSG:4326")
    elif args.region_file:
        rg = gpd.read_file(args.region_file)
        if args.region_name_field and args.region_name_value is not None:
            rg = rg[rg[args.region_name_field] == args.region_name_value]
            if rg.empty:
                raise SystemExit(f"Region '{args.region_name_value}' not found in {args.region_name_field}")
        if args.crs:
            rg = rg.to_crs(args.crs)
        return rg.geometry
    elif args.region_wkt:
        geom = shapely_wkt.loads(args.region_wkt)
        return gpd.GeoSeries([geom], crs=args.crs or "EPSG:4326")
    else:
        raise SystemExit("Provide one of --bbox, --region-file, or --region-wkt")

def tiles_overlapping_region(index_csv: str, region_gs: gpd.GeoSeries) -> pd.DataFrame:
    idx = pd.read_csv(index_csv)
    for col in ["ID","xmin","ymin","xmax","ymax"]:
        if col not in idx.columns: raise SystemExit(f"Index CSV missing '{col}'")
    # region bbox (union of all provided geoms)
    minx, miny, maxx, maxy = gpd.GeoSeries(region_gs.unary_union).total_bounds
    # fast bbox overlap test
    sep = (idx["xmax"] <= minx) | (maxx <= idx["xmin"]) | (idx["ymax"] <= miny) | (maxy <= idx["ymin"])
    return idx.loc[~sep].copy()

def build_tile_path(row, out_dir: str, suffix: str) -> str:
    if "File Name" in row and isinstance(row["File Name"], str) and row["File Name"]:
        base = os.path.splitext(os.path.basename(row["File Name"]))[0]
        return os.path.join(out_dir, base + suffix)
    # fallback to part-00000 style
    return os.path.join(out_dir, f"part-{int(row['ID']):05d}{suffix}")

def main():
    ap = argparse.ArgumentParser(description="Query: all points inside a region from tiled GeoParquet outputs.")
    ap.add_argument("out_dir", help="Directory containing per-tile GeoParquet files and tiles_index.csv")
    ap.add_argument("--index", default=None, help="Path to tiles_index.csv (default: out_dir/tiles_index.csv)")
    ap.add_argument("--geometry", default="geometry", help="Geometry column name (default: geometry)")
    ap.add_argument("--suffix", default=".parquet", help="Tile file suffix (default: .parquet)")
    ap.add_argument("--crs", default="EPSG:4326", help="Target CRS for region & data (default: EPSG:4326)")

    # One of these to define the region:
    ap.add_argument("--bbox", help="Region bbox as 'minx,miny,maxx,maxy'")
    ap.add_argument("--region-file", help="Vector file with the region geometry (GeoJSON/GeoPackage/GeoParquet)")
    ap.add_argument("--region-name-field", help="Attribute/column name to select a single region by value")
    ap.add_argument("--region-name-value", help="Attribute value to select a region")
    ap.add_argument("--region-wkt", help="Region geometry as WKT")

    ap.add_argument("--out", default="query_result.parquet", help="Output GeoParquet path")
    ap.add_argument("--columns", nargs="*", default=None, help="Optional list of non-geometry columns to keep")

    args = ap.parse_args()
    index_csv = args.index or os.path.join(args.out_dir, "tiles_index.csv")

    # 1) Load region geometry (to args.crs)
    region = load_region(args)
    region = region.to_crs(args.crs)  # normalize

    # 2) Pick overlapping tiles from index
    tiles = tiles_overlapping_region(index_csv, region)
    if tiles.empty:
        print("No tiles overlap region. Exiting.")
        return

    # 3) Read only those tiles; filter to points within region geometry precisely
    region_union = region.unary_union  # shapely geometry
    keep_cols = None
    if args.columns:
        keep_cols = list(dict.fromkeys(args.columns + [args.geometry]))  # ensure geometry present

    parts = []
    for _, row in tiles.iterrows():
        fp = build_tile_path(row, args.out_dir, args.suffix)
        print(f"Reading tile {fp} ...")
        if not os.path.exists(fp):
            # allow missing tiles gracefully
            continue
        gdf = gpd.read_parquet(fp, columns=keep_cols)
        # ensure CRS
        if gdf.crs and str(gdf.crs) != args.crs:
            gdf = gdf.to_crs(args.crs)
        # keep only points (safeguard) and then spatial filter
        geom = gdf.geometry
        is_point = geom.geom_type == "Point"
        gdf = gdf[is_point]
        if gdf.empty: 
            continue
        # fast bbox prefilter, then precise within()
        minx, miny, maxx, maxy = region_union.bounds
        bbox_mask = ~((geom.bounds.maxx < minx) | (geom.bounds.maxy < miny) |
                      (geom.bounds.minx > maxx) | (geom.bounds.miny > maxy))
        gdf = gdf[bbox_mask]
        if gdf.empty:
            continue
        gdf = gdf[gdf.geometry.within(region_union)]
        if not gdf.empty:
            parts.append(gdf)

    if not parts:
        print("No matching points found inside the region.")
        return

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry=args.geometry, crs=args.crs)

    # 4) Write GeoParquet result
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out):,} points to {args.out}")

if __name__ == "__main__":
    main()
