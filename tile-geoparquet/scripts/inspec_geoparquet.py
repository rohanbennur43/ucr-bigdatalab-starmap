# compute_bbox.py
import sys, json
import pyarrow.parquet as pq
import geopandas as gpd

path = sys.argv[1]

# read geo metadata (to detect primary geometry column if set)
pf = pq.ParquetFile(path)
md = (pf.schema_arrow.metadata or {})
geo = json.loads(md[b"geo"].decode("utf-8")) if b"geo" in md else {}
primary = geo.get("primary_column")

# load and set geometry if needed
gdf = gpd.read_parquet(path)
if primary and primary in gdf.columns and gdf.geometry.name != primary:
    gdf = gdf.set_geometry(primary, crs=gdf.crs)

bbox = gdf.total_bounds.tolist()  # [minx, miny, maxx, maxy]
print(bbox)
