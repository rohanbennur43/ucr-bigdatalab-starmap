import argparse, os, shutil, logging
from .datasource import GeoParquetSource,GeoJSONSource, is_geojson_path
from .assigner import TileAssignerFromCSV
from .orchestrator import RoundOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser(description="GeoJSON/GeoParquet → tiled GeoParquet (round-based, bounded writers).")
    ap.add_argument("--index", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-parallel-files", type=int, default=64)
    ap.add_argument("--row-group-rows", type=int, default=100_000)
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--use-intersects", action="store_true")
    ap.add_argument("--fresh-outdir", action="store_true")

    ap.add_argument("--geojson-batch-rows", type=int, default=50_000,
                    help="Rows per read batch when input is GeoJSON (default: 50k)")
    ap.add_argument("--src-crs", default="EPSG:4326",
                    help="Assumed source CRS for GeoJSON (default: EPSG:4326)")
    ap.add_argument("--target-crs", default=None,
                    help="Optional target CRS to reproject to during read (e.g., EPSG:3857)")
    ap.add_argument("--keep-null-geoms", action="store_true",
                    help="Keep rows with null/empty geometries (default: drop)")

    args = ap.parse_args()

    if args.fresh_outdir and os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)

    logger.info(f"CLI invoked with input={args.input}, index={args.index}, outdir={args.outdir}")
    # Select data source: GeoJSON vs GeoParquet
    if is_geojson_path(args.input):
        try:
            source = GeoJSONSource(
                path=args.input,
                batch_rows=args.geojson_batch_rows,
                src_crs=args.src_crs,
                target_crs=args.target_crs,
                keep_null_geoms=args.keep_null_geoms,
            )
            logger.info("Using GeoJSONSource with pyogrio")
        except ImportError as e:
            raise SystemExit(
                "pyogrio is required to read GeoJSON as Arrow. "
                "Install with: pip install pyogrio"
            ) from e
    else:
        source = GeoParquetSource(args.input)


    assigner = TileAssignerFromCSV(args.index, geom_col=args.geom_col, use_intersects=args.use_intersects)
    orchestrator = RoundOrchestrator(
        source=source,
        assigner=assigner,
        outdir=args.outdir,
        max_parallel_files=args.max_parallel_files,
        row_group_rows=args.row_group_rows,
        compression="zstd",
    )
    orchestrator.run()
