import pyarrow.parquet as pq
import json

path = "./out_dir/part-00000.parquet"
meta = pq.read_metadata(path).metadata or {}

if b"geo" in meta:
    print("✅ GeoParquet detected!")
    geo_meta = json.loads(meta[b"geo"].decode("utf-8"))
    print(json.dumps(geo_meta, indent=2))
else:
    print("❌ Not a GeoParquet file.")
