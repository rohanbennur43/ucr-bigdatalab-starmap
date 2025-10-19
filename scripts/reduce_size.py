#!/usr/bin/env python3
# slice_geojson_bbox.py
# Stream-filter a large GeoJSON by bounding box overlap.

import argparse, gzip, json, os, sys, math, random
from typing import Dict, Any, Iterable, Tuple

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
    # inclusive overlap
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def coords_bounds(coords) -> Tuple[float,float,float,float]:
    # Recursively compute bounds from "coordinates"
    if not coords:
        return (math.inf, math.inf, -math.inf, -math.inf)
    if isinstance(coords[0], (float, int)):  # a single [x,y(,z)] position
        x = float(coords[0]); y = float(coords[1])
        return (x, y, x, y)
    # list of positions or nested lists
    minx=miny= math.inf
    maxx=maxy=-math.inf
    for c in coords:
        bx1, by1, bx2, by2 = coords_bounds(c)
        if bx1 < minx: minx = bx1
        if by1 < miny: miny = by1
        if bx2 > maxx: maxx = bx2
        if by2 > maxy: maxy = by2
    return (minx, miny, maxx, maxy)

def geom_bounds(geom: Dict[str, Any]) -> Tuple[float,float,float,float]:
    if not geom:  # null geometry
        return (math.inf, math.inf, -math.inf, -math.inf)
    # If GeoJSON has a 'bbox' field, prefer that (faster)
    if "bbox" in geom and geom["bbox"] and len(geom["bbox"]) >= 4:
        b = geom["bbox"]
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    t = geom.get("type")
    if t == "GeometryCollection":
        minx=miny= math.inf
        maxx=maxy=-math.inf
        for g in geom.get("geometries", []):
            bx1, by1, bx2, by2 = geom_bounds(g)
            if bx1 < minx: minx = bx1
            if by1 < miny: miny = by1
            if bx2 > maxx: maxx = bx2
            if by2 > maxy: maxy = by2
        return (minx, miny, maxx, maxy)
    # All other geometry types rely on "coordinates"
    return coords_bounds(geom.get("coordinates"))

# ---------- JSON write helpers ----------
def write_header(out_bin, indent: int):
    if indent > 0:
        nl = "\n"; sp = " " * indent
        txt = "{"+nl+sp+'"type": "FeatureCollection",'+nl+sp+'"features": ['+nl
    else:
        txt = '{"type":"FeatureCollection","features":['
    out_bin.write(txt.encode("utf-8"))

def write_footer(out_bin, indent: int, wrote_any: bool):
    if indent > 0:
        txt = "\n]\n}\n"
    else:
        txt = "]}"
    out_bin.write(txt.encode("utf-8"))

def feature_bytes(feature: Dict[str, Any], indent: int, first: bool) -> bytes:
    # Ensure any Decimal from ijson is float by decoding with use_float=True
    if indent > 0:
        prefix = "" if first else ",\n"
        return (prefix + json.dumps(feature, ensure_ascii=False, indent=indent)).encode("utf-8")
    else:
        prefix = b"" if first else b","
        return prefix + json.dumps(feature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

# ---------- Stream filter ----------
def stream_filter_bbox(input_path: str, output_path: str, bbox: Tuple[float,float,float,float],
                       indent: int):
    with open_maybe_gzip(input_path, "r") as inf, open_maybe_gzip(output_path, "w") as outf:
        # Write header (binary)
        if hasattr(outf, "write") and "b" not in getattr(outf, "mode", "wb"):
            # gzip.open(...,"wb") returns binary; open(...,"wb") too.
            # If we accidentally got text, reopen as binary.
            outf.close()
            outf = open_maybe_gzip(output_path, "wb")
        write_header(outf, indent)
        first = True
        kept = 0
        for feat in ijson.items(inf, "features.item", use_float=True):
            # Try feature.bbox first
            if "bbox" in feat and feat["bbox"] and len(feat["bbox"]) >= 4:
                fb = feat["bbox"]
                fbbox = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
            else:
                fbbox = geom_bounds(feat.get("geometry"))
            if not math.isfinite(fbbox[0]):
                # empty/invalid geometry -> skip
                continue
            if bbox_intersects(fbbox, bbox):
                outf.write(feature_bytes(feat, indent, first))
                first = False
                kept += 1
        write_footer(outf, indent, kept > 0)
    sys.stderr.write(f"Done. Kept {kept} features that intersect bbox {bbox}\n")

def parse_bbox(s: str) -> Tuple[float,float,float,float]:
    # Expect "minx,miny,maxx,maxy" in lon,lat (EPSG:4326)
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
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")
    bbox = parse_bbox(args.bbox)
    stream_filter_bbox(args.input, args.output, bbox, args.indent)

if __name__ == "__main__":
    main()
