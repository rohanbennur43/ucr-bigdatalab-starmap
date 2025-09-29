# StarMap – Scripts

Utilities to convert, slice, and benchmark OSM21 datasets (e.g., Lakes/Buildings) using GeoJSON and GeoParquet.

## Contents

- `convert.py` – Convert **GeoJSON ⇄ GeoParquet** (with optional per-row bbox on write) + simple timing.
- `benchmark_geopandas.py` – Benchmark **post-filter** workflow: load GeoJSON master → bbox filter in memory → write subsets.
- `benchmark_geoparquet.py` – Benchmark **post-filter** workflow: load GeoParquet master once → bbox filter in memory → write subsets.
- `benchmark_geoparquet_internal_filtering.py` – Benchmark **internal bbox filtering**: `gpd.read_parquet(..., bbox=...)` (predicate pushdown) and write subsets.

> Regions used (lon/lat, EPSG:4326):  
> `ucr` (-117.35, 33.95, -117.30, 34.00) • `riverside` (-117.55, 33.85, -117.25, 34.05) •  
> `county` (-118.00, 33.40, -116.40, 34.20) • `southern_ca` (-119.00, 32.50, -115.50, 34.50)

---

## Requirements

- Python 3.9+ recommended  
- `geopandas`, `pyarrow`, `shapely`  

Install:
```bash
pip install geopandas pyarrow shapely 
```

## Usage

### 1. Convert GeoJSON ⇄ GeoParquet

#### GeoJSON → GeoParquet (with per-row bbox)
Writes GeoParquet 1.1 with per-row bbox (`struct{xmin,ymin,xmax,ymax}`) when `--bbox=1`. Prints per-run times and avg/min/max.

```bash
python convert.py gpq input.geojson output.parquet --repeat=3 --bbox=1
```

#### GeoParquet → GeoJSON

```bash
python convert.py gjs input.parquet output.geojson --repeat=2
```

---

### 2. Benchmark (Post-filter, load once → filter in memory)

Reads the GeoParquet master once into memory, applies bbox with intersects, and writes subsets.

```bash
python benchmark_geoparquet.py master.parquet
```

**Sample output:**
```
Benchmark results:
ucr: rows 28 / 918376 (load=0.662s, filter=0.026s, write=0.002s, total=0.690s)
...
```

---

### 3. Benchmark (Internal bbox filter / predicate pushdown)

Reads each region using `bbox=` so Parquet/Arrow can prune row groups (requires the file was written with per-row bbox).

```bash
python benchmark_geoparquet_internal_filtering.py master_with_bbox.parquet
# or project geometry only:
python benchmark_geoparquet_internal_filtering.py master_with_bbox.parquet geometry_only
```

**Sample output:**
```
Benchmark results (internal bbox filter):
ucr: rows 30 / 918376 (load=0.0s, filter=0.206s, write=0.003s, total=0.209s)
...
```

---

#### Important notes

- **Pushdown is working** (row-group pruning confirmed).
- But internal filtering did **not outperform post-filtering**.

**Reasons:**
- Row groups are large → small bboxes still load most groups.
- Data not spatially clustered → many groups overlap every bbox.
- Overhead of Arrow filtering dominates small queries.
- Filesystem caching makes later/larger reads appear faster.

**To see clear wins:**
- Write with smaller row groups (e.g., `row_group_size=50_000`).
- Spatially sort rows before writing (cluster by tile/Hilbert key).

---

### 4. Benchmark (GeoJSON → subsets)

To show the GeoJSON load penalty:

```bash
python benchmark_geopandas.py master.geojson
```

Output shows ~20s load time for large files, with tiny filter/write times per region.

---

## Dataset

Download OSM21 datasets (e.g., Lakes/Buildings) for California from UCR Star:  
https://star.cs.ucr.edu/?osm21/buildings#center=37.39,-118.17&zoom=6

Save as `*.geojson`, then convert with `convert.py`.

---

## Tips & Troubleshooting

- **Confirm per-row bbox exists:** schema should show  
  `bbox: struct<xmin: double, ymin: double, xmax: double, ymax: double>`.
- **Verify pushdown:** use PyArrow Dataset to count row groups selected with vs without bbox.
- **For clearer benchmarks:**
  - Use smaller row groups.
  - Spatially sort rows.
  - Run multiple trials
