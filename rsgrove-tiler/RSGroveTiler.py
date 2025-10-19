#!/usr/bin/env python3
import argparse, json, numpy as np, math
from venv import logger
import geopandas as gpd
from shapely.geometry import Point, Polygon, mapping
from RSGrove import RSGrovePartitioner, BeastOptions, EnvelopeNDLite
import os
import pandas as pd
# assumes RSGrovePartitioner, BeastOptions, EnvelopeNDLite are already imported

class Summary2D:
    def __init__(self, mins, maxs):
        self._mins = np.array(mins, float)
        self._maxs = np.array(maxs, float)
    def getCoordinateDimension(self): return 2
    def getMinCoord(self, d): return float(self._mins[d])
    def getMaxCoord(self, d): return float(self._maxs[d])

def _clip_inf_to_summary(mins, maxs, summary):
    smin = np.array([summary.getMinCoord(0), summary.getMinCoord(1)], float)
    smax = np.array([summary.getMaxCoord(0), summary.getMaxCoord(1)], float)
    mins = np.where(np.isneginf(mins), smin, mins)
    maxs = np.where(np.isposinf(maxs),  smax, maxs)
    return mins, maxs

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("RSGroveTiler")

    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="GeoParquet file path")
    ap.add_argument("--sample-pct", type=float, required=True)
    ap.add_argument("--num-partitions", type=int, required=True)
    ap.add_argument("--out", default="tiles.geojson")
    ap.add_argument("--expand-to-inf", action="store_true")
    args = ap.parse_args()

    logger.info(f"Reading GeoParquet: {args.input}")
    gdf = gpd.read_parquet(args.input)
    logger.info(f"Loaded {len(gdf):,} rows.")
    if gdf.empty:
        logger.error("GeoParquet has no rows.")
        raise SystemExit("GeoParquet has no rows.")

    # Detect CRS; if geographic, project to EPSG:3857 for centroid computation
    if gdf.crs and gdf.crs.is_geographic:
        logger.info("Projecting to EPSG:3857 for centroid computation.")
        gdf_proj = gdf.to_crs(epsg=3857)
        gdf_proj["centroid"] = gdf_proj.geometry.centroid
        # Bring centroids back to original CRS for output consistency
        gdf["centroid"] = gdf_proj["centroid"].to_crs(gdf.crs)
    else:
        logger.info("Computing centroids in native CRS.")
        gdf["centroid"] = gdf.geometry.centroid

    gdf["x"] = gdf["centroid"].x
    gdf["y"] = gdf["centroid"].y

    # Randomly sample rows
    frac = args.sample_pct / 100.0
    logger.info(f"Sampling {frac*100:.2f}% of rows.")
    gdf_sample = gdf.sample(frac=frac, random_state=42)
    logger.info(f"Sampled {len(gdf_sample):,} rows.")
    xs = gdf_sample["x"].to_numpy()
    ys = gdf_sample["y"].to_numpy()

    # Global MBR
    xmin, ymin, xmax, ymax = gdf.total_bounds
    logger.info(f"Global bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    summary = Summary2D([xmin, ymin], [xmax, ymax])

    logger.info("Configuring partitioner options.")
    conf = BeastOptions({
        RSGrovePartitioner.MMRatio: 0.95,
        RSGrovePartitioner.MinSplitRatio: 0.0,
        RSGrovePartitioner.ExpandToInfinity: bool(args.expand_to_inf),
    })
    r = RSGrovePartitioner()
    r.setup(conf, disjoint=True)
    sample_coords = np.vstack([xs, ys])
    logger.info(f"Constructing partitions (numPartitions={args.num_partitions})…")
    r.construct(summary=summary, sample=sample_coords, histogram=None,
                numPartitions=args.num_partitions)
    logger.info(f"Constructed {r.numPartitions()} partitions.")

    # # Convert boxes → polygons → GeoDataFrame
    # logger.info("Converting partition boxes to polygons.")
    # polys, ids = [], []
    # for pid in range(r.numPartitions()):
    #     env = EnvelopeNDLite(np.zeros(2), np.zeros(2))
    #     r.getPartitionMBR(pid, env)
    #     mins, maxs = _clip_inf_to_summary(env.mins, env.maxs, summary)
    #     minx, miny = mins; maxx, maxy = maxs
    #     poly = Polygon([(minx, miny), (maxx, miny),
    #                     (maxx, maxy), (minx, maxy)])
    #     polys.append(poly)
    #     ids.append(pid)
    #     logger.info(f"Partition {pid}: ({minx}, {miny}) to ({maxx}, {maxy})")

    # out_gdf = gpd.GeoDataFrame({"id": ids, "geometry": polys}, crs=gdf.crs)
    # out_gdf.to_file(args.out, driver="GeoJSON")
    # logger.info(f"Wrote {len(out_gdf)} tiles to {args.out}")
    logger.info("Converting partition boxes to polygons and writing CSV index.")
    polys, ids, records = [], [], []

    for pid in range(r.numPartitions()):
        env = EnvelopeNDLite(np.zeros(2), np.zeros(2))
        r.getPartitionMBR(pid, env)
        mins, maxs = _clip_inf_to_summary(env.mins, env.maxs, summary)
        minx, miny = mins
        maxx, maxy = maxs
        poly = Polygon([(minx, miny), (maxx, miny),
                        (maxx, maxy), (minx, maxy)])
        polys.append(poly)
        ids.append(pid)

        fname = f"part-{pid:05d}.rtree"

        # placeholder fields; to be filled when dataset is actually written per tile
        record = {
            "ID": pid,
            "File Name": fname,
            "Record Count": 0,
            "NonEmpty Count": 0,
            "NumPoints": 0,
            "Data Size": 0,
            "Sum_x": 0.0,
            "Sum_y": 0.0,
            "Geometry": poly.wkt,
            "xmin": minx,
            "ymin": miny,
            "xmax": maxx,
            "ymax": maxy,
        }
        records.append(record)

        logger.info(f"Partition {pid}: ({minx}, {miny}) to ({maxx}, {maxy})")

        # --- save tiles.geojson ---
        out_gdf = gpd.GeoDataFrame({"id": ids, "geometry": polys}, crs=gdf.crs)
        out_gdf.to_file(args.out, driver="GeoJSON")
        logger.info(f"Wrote {len(out_gdf)} tiles to {args.out}")

        # --- save index.csv ---
        index_path = os.path.splitext(args.out)[0] + "_index.csv"
        pd.DataFrame(records).to_csv(index_path, index=False)
        logger.info(f"Wrote index CSV with {len(records)} rows to {index_path}")

    

if __name__ == "__main__":
    main()
