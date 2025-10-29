from __future__ import annotations
import argparse
import logging

from .datasource import GeoParquetSource, GeoJSONSource, is_geojson_path
from .assigner import TileAssignerFromCSV, RSGroveAssigner
from .orchestrator import RoundOrchestrator
from .writer_pool import SortMode, SortKey

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_source(input_path: str, geom_col: str):
    if is_geojson_path(input_path):
        logger.info("Using GeoJSONSource for %s", input_path)
        return GeoJSONSource(input_path, geom_col=geom_col)
    else:
        logger.info("Using GeoParquetSource for %s", input_path)
        return GeoParquetSource(input_path)


def _parse_sort_mode(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("", "none"): return SortMode.NONE
    if s in ("columns", "cols", "column"): return SortMode.COLUMNS
    if s in ("z", "zorder", "z-order", "morton"): return SortMode.ZORDER
    if s in ("hilbert", "h"): return SortMode.HILBERT
    raise argparse.ArgumentTypeError(f"Unsupported --sort-mode: {s}")


def _parse_sort_keys(keys: str | None):
    # format examples:
    #   "colA,colB" (both ascending)
    #   "colA:asc,colB:desc"
    if not keys:
        return None
    out = []
    for tok in keys.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, order = tok.split(":", 1)
            out.append(SortKey(name.strip(), ascending=(order.strip().lower() != "desc")))
        else:
            out.append(SortKey(tok, True))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="GeoJSON/GeoParquet → tiled GeoParquet (round-based, bounded writers, single final writes)."
    )
    # Source
    ap.add_argument("--input", required=True, help="Path to input GeoJSON or GeoParquet.")
    ap.add_argument("--geom-col", default="geometry", help="Geometry column name (default: geometry).")

    # Output / run
    ap.add_argument("--outdir", required=True, help="Output directory for tiles.")
    ap.add_argument("--compression", default="zstd", help="Parquet compression codec (default: zstd).")
    ap.add_argument("--max-parallel-files", type=int, default=64,
                    help="Max files to write concurrently each round.")
    ap.add_argument("--sort-mode", type=_parse_sort_mode, default=SortMode.ZORDER,
                    help="none|columns|zorder|hilbert (hilbert currently = zorder).")
    ap.add_argument("--sort-keys", default=None,
                    help='Only for --sort-mode=columns. Example: "colA:asc,colB:desc".')
    ap.add_argument("--sfc-bits", type=int, default=16,
                    help="Bits per axis for Z-order/Hilbert key (typical: 16–20).")

    # Mode
    ap.add_argument("--index", help="CSV index with columns: id,minx,miny,maxx,maxy (legacy mode).")
    ap.add_argument("--num-tiles", type=int, help="Number of tiles to build via RSGrove (preferred).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for RSGrove (if --num-tiles is used).")

    # Sampling (RSGrove)
    ap.add_argument("--sample-ratio", type=float, default=1.0,
                    help="Bernoulli sampling probability for centroids (0<r<=1).")
    ap.add_argument("--sample-cap", type=int, default=None,
                    help="Reservoir sampling cap K (wins over ratio if provided).")

    args = ap.parse_args()

    source = build_source(args.input, geom_col=args.geom_col)

    if args.index:
        logger.info("Using TileAssignerFromCSV with index=%s", args.index)
        assigner = TileAssignerFromCSV(args.index, geom_col=args.geom_col)
    else:
        if not args.num_tiles:
            ap.error("You must supply either --index (CSV) or --num-tiles (RSGrove).")
        logger.info(
            "Building RSGroveAssigner from source with num_tiles=%d seed=%d ratio=%.4f cap=%s",
            args.num_tiles, args.seed, args.sample_ratio, str(args.sample_cap),
        )
        assigner = RSGroveAssigner.from_source(
            tables=source.iter_tables(),  # streaming
            num_partitions=args.num_tiles,
            geom_col=args.geom_col,
            seed=args.seed,
            sample_ratio=args.sample_ratio,
            sample_cap=args.sample_cap,
        )

    orchestrator = RoundOrchestrator(
        source=source,
        assigner=assigner,
        outdir=args.outdir,
        max_parallel_files=args.max_parallel_files,
        compression=args.compression,
        sort_mode=args.sort_mode,
        sort_keys=_parse_sort_keys(args.sort_keys),
        sfc_bits=args.sfc_bits,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()