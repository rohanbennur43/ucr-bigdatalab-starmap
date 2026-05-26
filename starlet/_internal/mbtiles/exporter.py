"""Export MVT tiles to MBTiles archive format."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_mbtiles(
    mvt_dir: str,
    output_path: str,
    dataset_dir: str | None = None,
) -> str:
    """Export MVT tiles directory to MBTiles SQLite archive.

    MBTiles is a specification for storing tiled map data in SQLite databases.
    Spec: https://github.com/mapbox/mbtiles-spec/blob/master/1.3/spec.md

    Parameters
    ----------
    mvt_dir : str
        Directory containing MVT tiles in z/x/y.mvt structure.
    output_path : str
        Path to output .mbtiles file.
    dataset_dir : str | None
        Dataset directory containing stats/ for metadata extraction.
        If None, uses default metadata.

    Returns
    -------
    str
        Path to created MBTiles file.

    Notes
    -----
    MBTiles uses TMS tile coordinates (origin bottom-left), while MVT uses
    XYZ coordinates (origin top-left). This function handles the conversion:
    tms_y = (2^zoom - 1) - xyz_y

    Examples
    --------
    >>> export_to_mbtiles(
    ...     mvt_dir="datasets/mydata/mvt",
    ...     output_path="datasets/mydata/mydata.mbtiles",
    ...     dataset_dir="datasets/mydata"
    ... )
    'datasets/mydata/mydata.mbtiles'
    """
    mvt_path = Path(mvt_dir)
    if not mvt_path.exists():
        raise FileNotFoundError(f"MVT directory not found: {mvt_dir}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file if present
    if output_file.exists():
        logger.info(f"Removing existing MBTiles file: {output_path}")
        output_file.unlink()

    logger.info(f"Exporting MVT tiles from {mvt_dir} to {output_path}")

    # Create SQLite database
    conn = sqlite3.connect(str(output_file))
    cursor = conn.cursor()

    # Create MBTiles schema
    cursor.execute("""
        CREATE TABLE tiles (
            zoom_level INTEGER NOT NULL,
            tile_column INTEGER NOT NULL,
            tile_row INTEGER NOT NULL,
            tile_data BLOB NOT NULL,
            PRIMARY KEY (zoom_level, tile_column, tile_row)
        )
    """)

    cursor.execute("""
        CREATE TABLE metadata (
            name TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Collect all tiles and convert XYZ to TMS
    tiles = []
    min_zoom, max_zoom = float('inf'), 0

    logger.info("Scanning MVT tiles...")
    for tile_file in mvt_path.rglob("*.mvt"):
        rel_path = tile_file.relative_to(mvt_path)
        parts = rel_path.parts

        if len(parts) != 3:
            logger.warning(f"Skipping invalid tile path: {tile_file}")
            continue

        try:
            z = int(parts[0])
            x = int(parts[1])
            y_str = parts[2]
            if not y_str.endswith(".mvt"):
                logger.warning(f"Skipping non-MVT file: {tile_file}")
                continue
            y = int(y_str.replace(".mvt", ""))
        except ValueError:
            logger.warning(f"Skipping non-numeric tile path: {tile_file}")
            continue

        # Convert XYZ to TMS (flip Y axis)
        # MBTiles spec uses TMS: origin bottom-left
        # MVT uses XYZ: origin top-left
        tms_y = (2 ** z - 1) - y

        tile_data = tile_file.read_bytes()
        tiles.append((z, x, tms_y, tile_data))

        min_zoom = min(min_zoom, z)
        max_zoom = max(max_zoom, z)

    if not tiles:
        conn.close()
        output_file.unlink()
        raise ValueError(f"No MVT tiles found in {mvt_dir}")

    logger.info(f"Found {len(tiles)} tiles (zoom {min_zoom}-{max_zoom})")

    # Insert tiles in a transaction for performance
    logger.info("Inserting tiles into MBTiles database...")
    cursor.execute("BEGIN TRANSACTION")
    cursor.executemany(
        "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
        tiles
    )
    cursor.execute("COMMIT")

    # Extract metadata from dataset stats
    bounds = "-180,-85,180,85"  # Default Web Mercator bounds
    center = "0,0,4"
    name = mvt_path.parent.name if mvt_path.parent.name != "." else "dataset"
    description = f"Starlet MVT tiles: {name}"

    if dataset_dir:
        stats_file = Path(dataset_dir) / "stats" / "attributes.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    stats = json.load(f)

                # Extract bounds from geometry stats
                for attr in stats.get("attributes", []):
                    if attr.get("name") == "geometry":
                        mbr = attr.get("stats", {}).get("mbr")
                        if mbr and len(mbr) == 4:
                            bounds = f"{mbr[0]},{mbr[1]},{mbr[2]},{mbr[3]}"
                            center_lon = (mbr[0] + mbr[2]) / 2
                            center_lat = (mbr[1] + mbr[3]) / 2
                            center_zoom = min_zoom + (max_zoom - min_zoom) // 2
                            center = f"{center_lon},{center_lat},{center_zoom}"
                        break
            except Exception as e:
                logger.warning(f"Could not read metadata from {stats_file}: {e}")

    # Insert metadata according to MBTiles spec 1.3
    metadata = {
        "name": name,
        "type": "overlay",  # or "baselayer"
        "version": "1.0.0",
        "description": description,
        "format": "pbf",  # Protocol Buffer Format (MVT)
        "bounds": bounds,
        "center": center,
        "minzoom": str(min_zoom),
        "maxzoom": str(max_zoom),
    }

    logger.info("Writing metadata...")
    cursor.executemany(
        "INSERT INTO metadata (name, value) VALUES (?, ?)",
        metadata.items()
    )

    # Create index for tile lookups
    logger.info("Creating spatial index...")
    cursor.execute("""
        CREATE UNIQUE INDEX tile_index
        ON tiles (zoom_level, tile_column, tile_row)
    """)

    conn.commit()

    # Vacuum to minimize file size
    logger.info("Optimizing database (VACUUM)...")
    cursor.execute("VACUUM")

    conn.close()

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"MBTiles export complete: {output_path} ({file_size_mb:.2f} MB)")
    logger.info(f"  Tiles: {len(tiles)}")
    logger.info(f"  Zoom range: {min_zoom}-{max_zoom}")
    logger.info(f"  Bounds: {bounds}")

    return str(output_file)
