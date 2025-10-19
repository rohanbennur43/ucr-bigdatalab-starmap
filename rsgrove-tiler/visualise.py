#!/usr/bin/env python3
import argparse, os, glob
import geopandas as gpd
from shapely.geometry import box
import folium
import pyarrow.parquet as pq

def list_parquets(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.parquet"))
        else:
            files += glob.glob(p)  # supports globs like: out_dir/part-*.parquet
    return sorted(set(files))

def row_count_fast(path: str) -> int:
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser(description="Interactive map of MBRs for GeoParquet tiles (Leaflet/Folium).")
    ap.add_argument("--inputs", nargs="+", help="Parquet files, globs, or directories")
    ap.add_argument("--out", default="tiles_mbr_map.html", help="Output HTML file")
    ap.add_argument("--assume-crs", default="EPSG:4326", help="CRS to assume if missing (default EPSG:4326)")
    ap.add_argument("--opacity", type=float, default=0.15, help="Fill opacity of boxes (0–1)")
    ap.add_argument("--color", default="#e74c3c", help="Stroke color (hex)")
    args = ap.parse_args()

    files = list_parquets(args.inputs)
    if not files:
        raise SystemExit("No Parquet files found.")

    rects = []
    names = []
    counts = []
    crs = None

    for fp in files:
        try:
            gdf = gpd.read_parquet(fp)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue
        if gdf.empty or gdf.geometry.isna().all():
            continue

        # choose/normalize CRS
        file_crs = gdf.crs or args.assume_crs
        if crs is None:
            crs = file_crs
        if str(file_crs) != "EPSG:4326":
            gdf = gdf.to_crs(4326)

        minx, miny, maxx, maxy = gdf.total_bounds
        rects.append(box(minx, miny, maxx, maxy))
        names.append(os.path.basename(fp))
        counts.append(row_count_fast(fp))

    if not rects:
        raise SystemExit("No geometries found to compute MBRs.")

    tiles = gpd.GeoDataFrame({"file": names, "rows": counts}, geometry=rects, crs="EPSG:4326")

    # Create Leaflet map centered on all rectangles
    m = folium.Map(location=[20,0], zoom_start=2, tiles="OpenStreetMap")
    minx, miny, maxx, maxy = tiles.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Add each rectangle as a clickable layer with tooltip
    for _, row in tiles.iterrows():
        geom = row.geometry  # polygon bbox
        coords = [(y, x) for x, y in geom.exterior.coords]
        tooltip = f"{row['file']} — {row['rows']:,} rows" if row['rows'] >= 0 else row['file']
        folium.Polygon(
            locations=coords,
            color=args.color,
            weight=2,
            fill=True,
            fill_color=args.color,
            fill_opacity=args.opacity,
            tooltip=tooltip,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(args.out)
    print(f"✔ Interactive map written to {args.out}")

if __name__ == "__main__":
    main()
