#!/usr/bin/env python3
import os, glob
import pyarrow.parquet as pq

def count_rows_in_parquet(file_path: str) -> int:
    """Return the total number of rows in a Parquet file."""
    try:
        meta = pq.read_metadata(file_path)
        return meta.num_rows
    except Exception as e:
        print(f"⚠️  Skipping {file_path}: {e}")
        return 0

def main():
    root = "out_dir"   # change if needed
    files = sorted(glob.glob(os.path.join(root, "part-*.parquet")))
    if not files:
        raise SystemExit(f"No Parquet files found in {root}")

    total_rows = 0
    for f in files:
        n = count_rows_in_parquet(f)
        print(f"{os.path.basename(f):25s} → {n:,} rows")
        total_rows += n

    print("-" * 50)
    print(f"Total rows across all tiles: {total_rows:,}")

if __name__ == "__main__":
    main()
