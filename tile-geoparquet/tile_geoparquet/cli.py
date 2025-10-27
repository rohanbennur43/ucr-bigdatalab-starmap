import argparse, os, shutil, logging
from .datasource import GeoParquetSource
from .assigner import TileAssignerFromCSV
from .orchestrator import RoundOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser(description="GeoParquet → tiled GeoParquet (round-based, bounded writers).")
    ap.add_argument("--index", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-parallel-files", type=int, default=64)
    ap.add_argument("--row-group-rows", type=int, default=100_000)
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--use-intersects", action="store_true")
    ap.add_argument("--fresh-outdir", action="store_true")
    args = ap.parse_args()

    if args.fresh_outdir and os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)

    logger.info(f"CLI invoked with input={args.input}, index={args.index}, outdir={args.outdir}")

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
