#!/usr/bin/env python3
import argparse, os, sys, math
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("ERROR: This script requires 'pyarrow'. Install with: pip install pyarrow", file=sys.stderr)
    sys.exit(2)


X_CANDIDATES = ["x", "lon", "longitude"]
Y_CANDIDATES = ["y", "lat", "latitude"]
GEOM_CANDIDATES = ["geometry", "geom", "wkb", "wkt"]


def find_first(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
        # try case-insensitive match
        for name in colnames:
            if name.lower() == c.lower():
                return name
    return None


def safe_getsize(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def summarize_parquet(path):
    """
    Returns a dict:
      record_count, nonempty_count, num_points, data_size, sum_x, sum_y
    - nonempty_count: non-null count of geometry-like column if available, else record_count
    - num_points: rows where both x and y are non-null if available; else falls back to nonempty_count
    - sum_x/sum_y: sum over x/y columns if available; else 0.0
    """
    out = {
        "record_count": 0,
        "nonempty_count": 0,
        "num_points": 0,
        "data_size": safe_getsize(path),
        "sum_x": 0.0,
        "sum_y": 0.0,
    }

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    n_rows = meta.num_rows if meta is not None else 0
    out["record_count"] = n_rows

    # Determine available columns once from schema
    schema = pf.schema_arrow
    colnames = [schema.names[i] for i in range(len(schema.names))]

    xcol = find_first(colnames, X_CANDIDATES)
    ycol = find_first(colnames, Y_CANDIDATES)
    gcol = find_first(colnames, GEOM_CANDIDATES)

    has_xy = xcol is not None and ycol is not None

    # Count non-empty geometry if possible without loading everything
    if gcol is not None:
        nonempty = 0
        for batch in pf.iter_batches(columns=[gcol], batch_size=131072):
            arr = batch.column(0)
            # pyarrow arrays support null_count; but per-batch count_nonzero isn't available for nulls
            # We'll compute valid count as len - null_count
            nonempty += len(arr) - arr.null_count
        out["nonempty_count"] = int(nonempty)
    else:
        out["nonempty_count"] = int(n_rows)

    # Aggregate sums for x/y and compute num_points = rows where both x and y are non-null
    if has_xy:
        sum_x = 0.0
        sum_y = 0.0
        npairs = 0
        for batch in pf.iter_batches(columns=[xcol, ycol], batch_size=131072):
            bx = batch.column(0).cast(pa.float64(), safe=True)
            by = batch.column(1).cast(pa.float64(), safe=True)

            # Convert to numpy with mask for speed
            nx = bx.to_numpy(zero_copy_only=False)
            ny = by.to_numpy(zero_copy_only=False)

            # Valid where both not NaN
            # (Arrow nulls become np.nan on to_numpy)
            import numpy as np
            mask = (~np.isnan(nx)) & (~np.isnan(ny))

            if mask.any():
                sum_x += float(np.nansum(nx[mask]))
                sum_y += float(np.nansum(ny[mask]))
                npairs += int(mask.sum())

        out["sum_x"] = float(sum_x)
        out["sum_y"] = float(sum_y)
        out["num_points"] = int(npairs)
    else:
        # Fallbacks
        out["sum_x"] = 0.0
        out["sum_y"] = 0.0
        out["num_points"] = int(out["nonempty_count"] or out["record_count"])

    return out


def main():
    ap = argparse.ArgumentParser(description="Update tiles index CSV by scanning Parquet files for counts/sums.")
    ap.add_argument("index_csv", help="Path to tiles_index.csv")
    ap.add_argument("out_dir", help="Directory containing the Parquet parts (e.g., out_dir)")
    ap.add_argument("--out", default=None, help="Path to write the updated CSV (default: <index_csv basename>_filled.csv)")
    ap.add_argument("--strict", action="store_true", help="Fail if a parquet file is missing or unreadable (default: skip with zeros)")
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)

    # Normalize column names used in the index CSV
    # Expect columns: 'File Name', 'Record Count', 'NonEmpty Count', 'NumPoints', 'Data Size', 'Sum_x', 'Sum_y'
    required_cols = ["File Name", "Record Count", "NonEmpty Count", "NumPoints", "Data Size", "Sum_x", "Sum_y"]
    for c in required_cols:
        if c not in df.columns:
            print(f"ERROR: Required column missing in index CSV: '{c}'", file=sys.stderr)
            sys.exit(3)

    updated_rows = 0
    for i, row in df.iterrows():
        fname = str(row["File Name"]).strip()
        fpath = os.path.join(args.out_dir, fname)

        if not os.path.isfile(fpath):
            msg = f"WARNING: File not found: {fpath}"
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(4)
            else:
                print(msg, file=sys.stderr)
                continue

        try:
            s = summarize_parquet(fpath)
        except Exception as e:
            msg = f"WARNING: Failed to read '{fpath}': {e}"
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(5)
            else:
                print(msg, file=sys.stderr)
                continue

        df.at[i, "Record Count"] = s["record_count"]
        df.at[i, "NonEmpty Count"] = s["nonempty_count"]
        df.at[i, "NumPoints"] = s["num_points"]
        df.at[i, "Data Size"] = s["data_size"]
        df.at[i, "Sum_x"] = s["sum_x"]
        df.at[i, "Sum_y"] = s["sum_y"]
        updated_rows += 1

    out_path = args.out or (os.path.splitext(args.index_csv)[0] + "_filled.csv")
    df.to_csv(out_path, index=False)
    print(f"Updated {updated_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
