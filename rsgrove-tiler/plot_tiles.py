#!/usr/bin/env python3
import sys, os, glob
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as ctx  # pip install contextily

root = sys.argv[1] if len(sys.argv) > 1 else "out_dir"
files = sorted(glob.glob(os.path.join(root, "*.parquet")))
if not files:
    raise SystemExit(f"No Parquet files in {root}")

geoms, names = [], []
crs = "EPSG:4326"

for fp in files:
    gdf = gpd.read_parquet(fp)
    if gdf.empty or gdf.geometry.isna().all():
        continue
    minx, miny, maxx, maxy = gdf.total_bounds
    geoms.append(box(minx, miny, maxx, maxy))
    names.append(os.path.basename(fp))

tiles = gpd.GeoDataFrame({"file": names}, geometry=geoms, crs=crs)

# Reproject to Web Mercator for contextily basemap
tiles_web = tiles.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
tiles_web.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=1)

ctx.add_basemap(ax, crs=tiles_web.crs, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title("Tile MBRs over OpenStreetMap")
ax.set_axis_off()
plt.tight_layout()
plt.show()
