#!/usr/bin/env python3
# bench_write_variations.py
import os, sys, time, subprocess, shlex
from datetime import datetime

# Matrix of variations to test
# You can comment/uncomment row-group sizes to narrow the sweep.
ROW_GROUP_SIZES = [262144, 131072, 65536]  # 256k, 128k, 64k

VARIATIONS = []
for rg in ROW_GROUP_SIZES:
    # Unordered
    VARIATIONS += [
        dict(name=f"unordered_rg{rg}_bb1", order=None, bits=None, bbox=1, row_group=rg),
        dict(name=f"unordered_rg{rg}_bb0", order=None, bits=None, bbox=0, row_group=rg),
    ]
    # Z-order
    VARIATIONS += [
        dict(name=f"z26_rg{rg}_bb1", order="z", bits=26, bbox=1, row_group=rg),
        dict(name=f"z26_rg{rg}_bb0", order="z", bits=26, bbox=0, row_group=rg),
    ]
    # Hilbert
    VARIATIONS += [
        dict(name=f"h26_rg{rg}_bb1", order="hilbert", bits=26, bbox=1, row_group=rg),
        dict(name=f"h26_rg{rg}_bb0", order="hilbert", bits=26, bbox=0, row_group=rg),
    ]

def human(n):
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024 or u == "TB": return f"{n:.2f} {u}"
        n /= 1024

def run(cmd):
    # Run command, capture live output (so you still see convert.py logs),
    # measure wall-clock write time (total command time).
    t0 = time.time()
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="")  # stream to console
        lines.append(line)
    proc.wait()
    dt = time.time() - t0
    return proc.returncode, dt, "".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bench_write_variations.py <input_parquet> <output_dir>")
        sys.exit(1)

    inp = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    # print header
    print(f"[INFO] Input: {inp}")
    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Start: {datetime.now().isoformat()}")
    print()

    rows = []
    for v in VARIATIONS:
        out = os.path.join(outdir, f"pois_{v['name']}.parquet")

        # Build command
        # We call your convert.py in pp mode; omit --order/--bits when unordered.
        base = f"python3 convert.py pp {inp} {out} --row-group-size={v['row_group']} --bbox={v['bbox']}"
        if v["order"] is not None:
            base += f" --order={v['order']} --bits={v['bits']}"

        print(f"\n=== Running: {base}")
        rc, wall, log = run(base)

        if rc != 0:
            print(f"[ERROR] Command failed with rc={rc}")
            size_str = "-"
        else:
            size = os.path.getsize(out) if os.path.exists(out) else 0
            size_str = human(size)

        rows.append(dict(
            variant=v["name"],
            order=v["order"] or "none",
            bits=v["bits"] or "-",
            row_group=v["row_group"],
            bbox=v["bbox"],
            wall_s=round(wall, 2),
            size=size_str,
            path=out,
        ))

    # Pretty table
    print("\n=== Write-time & size summary ===")
    print("Variant                   | Order    | Bits | RG     | BB | Write(s) | Size     | Output path")
    print("--------------------------+----------+------+--------+----+----------+----------+---------------------------------------")
    for r in rows:
        print(f"{r['variant']:<26} | {r['order']:<8} | {r['bits']!s:>4} | {r['row_group']:<6} | {r['bbox']:<2} | "
              f"{r['wall_s']:>8.2f} | {r['size']:>8} | {r['path']}")
