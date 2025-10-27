#!/usr/bin/env python3
"""
Sweep --max-parallel-files and measure peak memory usage.

Requirements:
  pip install psutil

Example:
  python sweep_mpf.py \
    --cmd tile-geoparquet \
    --index tiles_index.csv \
    --input input.geoparquet \
    --outdir-base out/tiles_run \
    --start 5 --stop 25 --step 5 \
    --row-group-rows 100000 \
    --geom-col geometry \
    --use-intersects \
    --fresh-outdir \
    --csv results.csv
"""
import argparse, subprocess, time, sys, os, psutil, csv
from datetime import datetime
from pathlib import Path

# --- replace your monitor_peak_rss with this ---
import psutil, time

# 1) Replace your monitor with this version (expects Popen)
def monitor_peak_rss(popen, poll_s=0.5) -> int:
    import psutil, time
    peak = 0
    try:
        root = psutil.Process(popen.pid)
    except psutil.Error:
        return 0

    post_exit_cycles = 2
    while True:
        rss = 0
        try:
            procs = [root] + root.children(recursive=True)
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass

        peak = max(peak, rss)

        if popen.poll() is not None:
            if post_exit_cycles <= 0:
                break
            post_exit_cycles -= 1

        time.sleep(poll_s)

    return peak



def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def main():
    ap = argparse.ArgumentParser(description="Sweep --max-parallel-files and measure peak memory.")
    ap.add_argument("--cmd", default="tile-geoparquet",
                    help="Command or module to execute (default: tile-geoparquet).")
    ap.add_argument("--index", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir-base", required=True,
                    help="Base directory; each run writes to <base>_mpf_<N>/")
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--stop", type=int, default=25)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--row-group-rows", type=int, default=100_000)
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--use-intersects", action="store_true")
    ap.add_argument("--fresh-outdir", action="store_true")
    ap.add_argument("--csv", default=None, help="Optional CSV output file for results.")
    ap.add_argument("--module", action="store_true",
                    help="If set, run as `python -m <cmd>` instead of invoking the binary.")
    args = ap.parse_args()

    sweep_vals = list(range(args.start, args.stop + 1, args.step))
    results = []

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting sweep: {sweep_vals}")

    for mpf in sweep_vals:
        outdir = f"{args.outdir_base}_mpf_{mpf}"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        cmd = []
        if args.module:
            cmd = [sys.executable, "-m", args.cmd]
        else:
            cmd = [args.cmd]

        cmd += [
            "--index", args.index,
            "--input", args.input,
            "--outdir", outdir,
            "--max-parallel-files", str(mpf),
            "--row-group-rows", str(args.row_group_rows),
            "--geom-col", args.geom_col,
        ]
        if args.use_intersects:
            cmd.append("--use-intersects")
        if args.fresh_outdir:
            cmd.append("--fresh-outdir")

        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Running: {cmd}")
        t0 = time.perf_counter()
        # Launch and wrap with psutil for monitoring
        popen = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        peak_rss = monitor_peak_rss(popen)
        rc = popen.wait()
        dt = time.perf_counter() - t0

        print(f"[{datetime.now().isoformat(timespec='seconds')}] Exit code: {rc} | "
              f"Time: {dt:.1f}s | Peak RSS: {human_bytes(peak_rss)}")

        results.append({
            "max_parallel_files": mpf,
            "exit_code": rc,
            "wall_time_sec": round(dt, 3),
            "peak_rss_bytes": peak_rss,
            "peak_rss_human": human_bytes(peak_rss),
            "outdir": outdir,
        })

        if rc != 0:
            print(f"Warning: run with --max-parallel-files {mpf} exited with code {rc}", file=sys.stderr)

    # Print a compact table
    print("\n=== Results ===")
    print(f"{'MPF':>4}  {'Time(s)':>8}  {'Peak RSS':>12}  {'Exit':>4}  Outdir")
    for r in results:
        print(f"{r['max_parallel_files']:>4}  {r['wall_time_sec']:>8.1f}  {r['peak_rss_human']:>12}  "
              f"{r['exit_code']:>4}  {r['outdir']}")

    # Optional CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved CSV: {args.csv}")

if __name__ == "__main__":
    main()
