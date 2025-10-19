#!/usr/bin/env python3
"""
run_all_rowgroup_benchmarks.py
--------------------------------
Runs benchmark_geoparquet_internal_filtering.py (or benchmark_parquet_internal.py)
on all Parquet files inside a directory to compare the effect of row-group size,
spatial ordering, and bbox flag.

Usage:
  python run_all_rowgroup_benchmarks.py ../dataset/bench_writes results_rowgroups.csv
"""

import os, sys, subprocess, csv, re

# regex to extract parameters from filename
PATTERN = re.compile(r'(?P<order>none|z|h|hilbert)?(\d+)?_?rg(?P<rg>\d+)_?bb(?P<bb>[01])', re.I)

def parse_filename(fname):
    """Parse ordering, row-group, bbox info from filename."""
    base = os.path.basename(fname)
    m = PATTERN.search(base)
    if not m:
        return {"order": "none", "rg": "", "bb": ""}
    order = (m.group("order") or "none").lower()
    if order == "h": order = "hilbert"
    return {
        "order": order,
        "rg": m.group("rg") or "",
        "bb": m.group("bb") or ""
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_all_rowgroup_benchmarks.py <dir_with_parquets> <output_csv>")
        sys.exit(1)

    parquet_dir = sys.argv[1]
    out_csv = sys.argv[2]

    if not os.path.isdir(parquet_dir):
        print(f"[ERROR] Directory not found: {parquet_dir}")
        sys.exit(1)

    parquet_files = [os.path.join(parquet_dir, f)
                     for f in os.listdir(parquet_dir)
                     if f.endswith(".parquet")]

    if not parquet_files:
        print(f"[ERROR] No .parquet files found in {parquet_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(parquet_files)} files to benchmark.")
    print(f"[INFO] Results will be saved to {out_csv}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "order", "row_group", "bbox", "region",
                         "rows_subset", "rows_total",
                         "load_s", "write_s", "total_s", "out_size_h"])

        for path in parquet_files:
            info = parse_filename(path)
            print(f"\n=== Running benchmark on: {os.path.basename(path)} "
                  f"(order={info['order']}, rg={info['rg']}, bb={info['bb']}) ===")

            # run your benchmark script as subprocess
            cmd = ["python3", "benchmark_geoparquet_internal_filtering.py", path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout.strip().splitlines()

            # parse benchmark output
            for line in output:
                if line.startswith("Benchmark") or not line or "→" not in line:
                    continue
                # sample line:
                # inland_empire: rows 413817 / 24819173 (load=0.161s, write=0.196s, total=0.357s)  → inland_empire__internal.parquet [32.43 MB]
                try:
                    region = line.split(":")[0].strip()
                    rows_part = line.split("rows")[1].split("(")[0].strip()
                    subset, total = rows_part.split("/")
                    load = line.split("load=")[1].split("s")[0]
                    write = line.split("write=")[1].split("s")[0]
                    total_time = line.split("total=")[1].split("s")[0]
                    size_h = line.split("[")[-1].strip("]")
                    writer.writerow([
                        path, info["order"], info["rg"], info["bb"], region,
                        subset.strip(), total.strip(),
                        load.strip(), write.strip(), total_time.strip(), size_h
                    ])
                except Exception:
                    pass

    print(f"\n[OK] All benchmarks completed. Results written to {out_csv}")

if __name__ == "__main__":
    main()
