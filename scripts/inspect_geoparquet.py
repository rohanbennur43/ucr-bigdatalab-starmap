#!/usr/bin/env python3
import sys
import json
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

def main(in_geojson, out_parquet):
    # Read GeoJSON
    pq.read_table(in_geojson)  # Ensure pyarrow can read it
    gdf = gpd.read_file(in_geojson)

    # Ensure CRS (set or convert to WGS84 if needed)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Write GeoParquet (GeoPandas defaults to WKB geometry encoding)
    # Try zstd, fall back to snappy if not available
    compression = "zstd"
    try:
        gdf.to_parquet(out_parquet, index=False)
    except Exception:
        compression = "snappy"
        gdf.to_parquet(out_parquet, index=False, compression=compression)

    print(f"Wrote {out_parquet} with compression={compression}")

    # Read back the GeoParquet
    gdf2 = gpd.read_parquet(out_parquet)

    # Print schema
    print("\n=== Schema ===")
    print(gdf2.dtypes)
    print(gdf2.head(1).geometry)
    print(gdf2.geometry)
    print(gdf.info())
    print(gdf.crs)
    # GeoParquet metadata if exposed by GeoPandas (may be None for older writers)
    meta = getattr(gdf2, "attrs", {}).get("geo", None)
    print("\n=== GeoParquet Metadata ===")
    if meta:
        print(json.dumps(meta, indent=2))
    else:
        print("(No embedded 'geo' metadata exposed by GeoPandas)")

    # # Show first 10 rows
    # print("\n=== First 10 rows ===")
    # with pd.option_context("display.max_columns", None, "display.max_colwidth", 120):
    #     print(gdf2.head(10))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python geojson_to_geoparquet_and_inspect.py <input.geojson> <output.parquet>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

# import pyarrow.parquet as pq
# import json

# # Path to your GeoParquet file
# file_path = "../original_datasets/out_ca.parquet"

# # Read the Parquet table
# table = pq.read_table(file_path)

# # Extract the metadata dictionary
# metadata = table.schema.metadata

# # Decode and pretty-print the GeoParquet metadata
# if metadata and b'geo' in metadata:
#     geo_meta = json.loads(metadata[b'geo'].decode('utf-8'))
#     print(json.dumps(geo_meta, indent=2))
# else:
#     print("No GeoParquet metadata found.")
