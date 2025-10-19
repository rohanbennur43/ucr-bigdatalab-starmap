#!/usr/bin/env python3
# slice_geojson_bbox.py
# Stream-filter a large GeoJSON by bounding box overlap, with progress logs and CRS debugging.

import argparse, gzip, json, os, sys, math, time
from typing import Dict, Any, Tuple

try:
    import ijson
except ImportError:
    sys.stderr.write("Requires 'ijson': pip install ijson\n")
    sys.exit(1)

# ---------- IO helpers ----------
def open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        if "r" in mode:
            return gzip.open(path, "rt", encoding="utf-8", newline="")
        else:
            return gzip.open(path, "wb")
    return open(path, "rt", encoding="utf-8", newline="") if "r" in mode else open(path, "wb")

# ---------- BBox math ----------
def bbox_intersects(a: Tuple[float,float,float,float],
                    b: Tuple[float,float,float,float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def coords_bounds(coords, order: str) -> Tuple[float,float,float,float]:
    if not coords:
        return (math.inf, math.inf, -math.inf, -math.inf)
    if isinstance(coords[0], (float, int)):
        if order == "xy":
            x = float(coords[0]); y = float(coords[1])
        else:  # yx
            y = float(coords[0]); x = float(coords[1])
        return (x, y, x, y)
    minx=miny= math.inf
    maxx=maxy=-math.inf
    for c in coords:
        bx1, by1, bx2, by2 = coords_bounds(c, order)
        if bx1 < minx: minx = bx1
        if by1 < miny: miny = by1
        if bx2 > maxx: maxx = bx2
        if by2 > maxy: maxy = by2
    return (minx, miny, maxx, maxy)

def geom_bounds(geom: Dict[str, Any], order: str) -> Tuple[float,float,float,float]:
    if not geom:
        return (math.inf, math.inf, -math.inf, -math.inf)
    if "bbox" in geom and geom["bbox"] and len(geom["bbox"]) >= 4:
        b = geom["bbox"]
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    t = geom.get("type")
    if t == "GeometryCollection":
        minx=miny= math.inf
        maxx=maxy=-math.inf
        for g in geom.get("geometries", []):
            bx1, by1, bx2, by2 = geom_bounds(g, order)
            if bx1 < minx: minx = bx1
            if by1 < miny: miny = by1
            if bx2 > maxx: maxx = bx2
            if by2 > maxy: maxy = by2
        return (minx, miny, maxx, maxy)
    return coords_bounds(geom.get("coordinates"), order)

# ---------- JSON write helpers ----------
def write_header(out_bin, indent: int) -> int:
    if indent > 0:
        nl = "\n"; sp = " " * indent
        txt = "{"+nl+sp+'"type": "FeatureCollection",'+nl+sp+'"features": ['+nl
    else:
        txt = '{"type":"FeatureCollection","features":['
    data = txt.encode("utf-8")
    out_bin.write(data)
    return len(data)

def write_footer(out_bin, indent: int, wrote_any: bool) -> int:
    txt = "\n]\n}\n" if indent > 0 else "]}"
    data = txt.encode("utf-8")
    out_bin.write(data)
    return len(data)

def feature_bytes(feature: Dict[str, Any], indent: int, first: bool) -> bytes:
    if indent > 0:
        prefix = "" if first else ",\n"
        return (prefix + json.dumps(feature, ensure_ascii=False, indent=indent)).encode("utf-8")
    else:
        prefix = b"" if first else b","
        return prefix + json.dumps(feature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def fmt_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

# ---------- Stream filter with progress ----------
def stream_filter_bbox(input_path: str, output_path: str, bbox: Tuple[float,float,float,float],
                       indent: int, log_interval_sec: float, job_name: str,
                       coord_order: str, peek: int):
    start = time.time()
    last_log = start
    seen = kept = 0
    bytes_out = 0

    sys.stderr.write(f"[{job_name}] Start\n")
    sys.stderr.write(f"[{job_name}] Input:  {input_path}\n")
    sys.stderr.write(f"[{job_name}] Output: {output_path}\n")
    sys.stderr.write(f"[{job_name}] BBox:   {bbox}\n")
    sys.stderr.write(f"[{job_name}] Coord order: {coord_order}\n")

    with open_maybe_gzip(input_path, "r") as inf, open_maybe_gzip(output_path, "wb") as outf:
        bytes_out += write_header(outf, indent)
        first = True

        for feat in ijson.items(inf, "features.item", use_float=True):
            seen += 1
            if "bbox" in feat and feat["bbox"] and len(feat["bbox"]) >= 4:
                fb = feat["bbox"]
                fbbox = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
            else:
                fbbox = geom_bounds(feat.get("geometry"), coord_order)

            if peek and seen <= peek:
                sys.stderr.write(f"[{job_name}] peek#{seen} fbbox={fbbox}\n")

            if not math.isfinite(fbbox[0]):
                pass
            elif bbox_intersects(fbbox, bbox):
                chunk = feature_bytes(feat, indent, first)
                outf.write(chunk)
                bytes_out += len(chunk)
                kept += 1
                first = False

            now = time.time()
            if now - last_log >= log_interval_sec:
                elapsed = now - start
                rate = seen / elapsed if elapsed > 0 else 0.0
                keep_ratio = (kept / seen * 100.0) if seen else 0.0
                sys.stderr.write(
                    f"[{job_name}] seen={seen:,} kept={kept:,} "
                    f"keep={keep_ratio:.2f}% out={fmt_bytes(bytes_out)} "
                    f"elapsed={elapsed:,.1f}s rate={rate:,.1f} feats/s\n"
                )
                last_log = now

        bytes_out += write_footer(outf, indent, kept > 0)

    total = time.time() - start
    rate = seen / total if total > 0 else 0.0
    keep_ratio = (kept / seen * 100.0) if seen else 0.0
    sys.stderr.write(
        f"[{job_name}] Done. seen={seen:,} kept={kept:,} "
        f"keep={keep_ratio:.2f}% out={fmt_bytes(bytes_out)} "
        f"elapsed={total:,.1f}s rate={rate:,.1f} feats/s\n"
    )

# ---------- CLI ----------
def parse_bbox(s: str) -> Tuple[float,float,float,float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox must be 'minx,miny,maxx,maxy' (lon,lat).")
    x1, y1, x2, y2 = map(float, parts)
    if x2 < x1 or y2 < y1:
        raise ValueError("BBox must have max >= min on both axes.")
    return (x1, y1, x2, y2)

def main():
    ap = argparse.ArgumentParser(description="Stream-filter a GeoJSON by bbox (lon/lat).")
    ap.add_argument("-i","--input", required=True, help="Input .geojson or .geojson.gz")
    ap.add_argument("-o","--output", required=True, help="Output .geojson or .geojson.gz")
    ap.add_argument("--bbox", required=True,
                    help="minx,miny,maxx,maxy in lon,lat (EPSG:4326). Example: --bbox=-124.5,32.3,-113.9,42.1")
    ap.add_argument("--indent", type=int, default=0, help="Pretty indent (0 compact).")
    ap.add_argument("--log-interval-sec", type=float, default=5.0,
                    help="Seconds between progress logs (default 5).")
    ap.add_argument("--name", default="bbox-filter", help="Short name for progress logs.")
    ap.add_argument("--coord-order", choices=["xy","yx"], default="xy",
                    help="Interpret coordinates as 'xy' (lon,lat) or 'yx' (lat,lon). Default xy.")
    ap.add_argument("--peek", type=int, default=0,
                    help="Log the first N feature bbox values for debugging.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    bbox = parse_bbox(args.bbox)
    stream_filter_bbox(args.input, args.output, bbox, args.indent,
                       args.log_interval_sec, args.name,
                       args.coord_order, args.peek)

if __name__ == "__main__":
    main()
