# tile-geoparquet
Split a GeoParquet into per-tile GeoParquet files using round-based, bounded writers.

## Usage
```bash
tile-geoparquet --index index.csv --input input.geoparquet --outdir out --max-parallel-files 64 --row-group-rows 100000
