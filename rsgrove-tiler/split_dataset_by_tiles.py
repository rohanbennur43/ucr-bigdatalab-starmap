#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, math, logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from shapely import from_wkb
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("tile-writer")

def load_index(index_csv: str):
    df = pd.read_csv(index_csv)
    required = ["ID","File Name","xmin","ymin","xmax","ymax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Index CSV missing columns: {missing}")
    # tiles arrays
    mins = df[["xmin","ymin"]].to_numpy(float)  # (P,2)
    maxs = df[["xmax","ymax"]].to_numpy(float)  # (P,2)
    names = df["File Name"].astype(str).tolist()
    ids = df["ID"].tolist()
    return ids, names, mins, maxs, df

def point_in_rect(x, y, mins, maxs):
    # returns index of the first rect containing the point or -1
    # mins/maxs: (P,2)
    ge_minx = x >= mins[:,0]
    lt_maxx = x <  maxs[:,0]
    ge_miny = y >= mins[:,1]
    lt_maxy = y <  maxs[:,1]
    hit = ge_minx & lt_maxx & ge_miny & lt_maxy  # (P,)
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else -1

def rects_overlap(a_minx, a_miny, a_maxx, a_maxy, mins, maxs):
    # vectorized: which tiles overlap this bbox? returns boolean mask (P,)
    sep = (maxs[:,0] <= a_minx) | (a_maxx <= mins[:,0]) | \
          (maxs[:,1] <= a_miny) | (a_maxy <= mins[:,1])
    return ~sep

def main():
    ap = argparse.ArgumentParser(
        description="Split a GeoParquet into per-tile Parquet files using tiles_index.csv"
    )
    ap.add_argument("input", help="Path to GeoParquet (dataset or file)")
    ap.add_argument("index_csv", help="tiles_index.csv produced earlier")
    ap.add_argument("out_dir", help="Output directory for per-tile Parquet files")
    ap.add_argument("--geometry", default="geometry", help="Geometry column name (default: geometry)")
    ap.add_argument("--mode", choices=["disjoint","covering"], default="disjoint",
                    help="Assignment mode: disjoint=by centroid to 1 tile; covering=write to all overlapping tiles")
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--suffix", default=".parquet",
                    help="Output file suffix/extension (default .parquet)")
    ap.add_argument("--use-index-filenames", action="store_true",
                    help="Use 'File Name' from index CSV (with --suffix replacing its extension)")
    args = ap.parse_args()

    logger.info(f"Creating output directory: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Loading tile index from: {args.index_csv}")
    tile_ids, tile_names, mins, maxs, index_df = load_index(args.index_csv)
    P = len(tile_ids)
    logger.info(f"Loaded {P} tiles from index CSV")

    logger.info(f"Preparing input dataset: {args.input}")
    dataset = ds.dataset(args.input, format="parquet")
    schema: pa.Schema = dataset.schema
    logger.info(f"Dataset schema columns: {schema.names}")
    if args.geometry not in schema.names:
        logger.error(f"Geometry column '{args.geometry}' not in dataset. Columns: {schema.names}")
        raise SystemExit(f"Geometry column '{args.geometry}' not in dataset. Columns: {schema.names}")

    # Prepare writers per tile (lazy open)
    writers: list[pq.ParquetWriter | None] = [None] * P
    counts = np.zeros(P, dtype=np.int64)

    def out_path_for(i: int) -> str:
        if args.use_index_filenames:
            base = os.path.splitext(tile_names[i])[0]
            return os.path.join(args.out_dir, base + args.suffix)
        else:
            return os.path.join(args.out_dir, f"part-{tile_ids[i]:05d}{args.suffix}")

    def ensure_writer(i: int):
        if writers[i] is None:
            # preserve original schema & metadata
            writers[i] = pq.ParquetWriter(out_path_for(i), schema=schema, version="2.6",
                                          compression="zstd", write_statistics=True)
        return writers[i]

    # Streaming over batches
    scanner = dataset.scanner(batch_size=args.batch_size)  # all columns
    total = 0
    assigned = 0
    logger.info(f"Starting batch scan with batch_size={args.batch_size}")
    batch_num = 0
    for batch in scanner.to_batches():
        batch_num += 1
        n = batch.num_rows
        logger.info(f"Processing batch {batch_num} with {n} rows.")
        if n == 0:
            logger.info(f"Batch {batch_num} is empty, skipping.")
            continue
        total += n

        # geometry column as numpy object array of bytes (or None)
        wkb_arr: pa.Array = batch.column(args.geometry)
        wkb_np = wkb_arr.to_numpy(zero_copy_only=False)  # dtype=object

        # shapely vectorized parse; compute centroids & bboxes
        geoms = from_wkb(wkb_np)
        valid_mask = np.array([g is not None and not g.is_empty for g in geoms], dtype=bool)
        logger.info(f"Batch {batch_num}: {valid_mask.sum()} valid geometries out of {n}.")
        if not np.any(valid_mask):
            logger.info(f"Batch {batch_num} has no valid geometries, skipping.")
            continue

        geoms = geoms[valid_mask]
        sub_batch = batch.filter(pa.array(valid_mask))  # keep rows in sync

        if args.mode == "disjoint":
            logger.info(f"Batch {batch_num}: Assigning features to tiles by centroid.")
            xs = np.array([g.x if isinstance(g, Point) else g.centroid.x for g in geoms], dtype=float)
            ys = np.array([g.y if isinstance(g, Point) else g.centroid.y for g in geoms], dtype=float)

            # Assign per feature
            tile_idx = np.full(xs.shape[0], -1, dtype=int)
            for j in range(xs.shape[0]):
                tile_idx[j] = point_in_rect(xs[j], ys[j], mins, maxs)
            logger.info(f"Batch {batch_num}: Assigned {np.count_nonzero(tile_idx >= 0)} features to tiles.")

            # write per tile
            for i in range(P):
                sel = tile_idx == i
                if not np.any(sel):
                    continue
                writer = ensure_writer(i)
                taken = sub_batch.filter(pa.array(sel.tolist()))
                writer.write_table(pa.Table.from_batches([taken]))
                c = int(sel.sum())
                counts[i] += c
                assigned += c
                logger.info(f"Batch {batch_num}: Wrote {c} rows to tile {i}.")

        else:  # covering mode
            logger.info(f"Batch {batch_num}: Assigning features to all overlapping tiles by bbox.")
            bxs = np.array([g.bounds[0] for g in geoms], dtype=float)
            bys = np.array([g.bounds[1] for g in geoms], dtype=float)
            bxe = np.array([g.bounds[2] for g in geoms], dtype=float)
            bye = np.array([g.bounds[3] for g in geoms], dtype=float)

            # For each tile, select overlapping rows and write
            for i in range(P):
                sep = (maxs[i,0] <= bxs) | (bxe <= mins[i,0]) | (maxs[i,1] <= bys) | (bye <= mins[i,1])
                sel = ~sep
                if not np.any(sel):
                    continue
                writer = ensure_writer(i)
                taken = sub_batch.filter(pa.array(sel.tolist()))
                writer.write_table(pa.Table.from_batches([taken]))
                c = int(sel.sum())
                counts[i] += c
                assigned += c
                logger.info(f"Batch {batch_num}: Wrote {c} rows to tile {i}.")

    # Close writers
    logger.info("Closing tile writers and finalizing output files.")
    for i, w in enumerate(writers):
        if w is not None:
            w.close()
            logger.info(f"Wrote tile {i} → {out_path_for(i)} ({counts[i]} rows)")

    logger.info(f"Done. Total rows seen: {total}, rows written: {assigned} "
                f"(mode={args.mode}, tiles={P})")

if __name__ == "__main__":
    main()