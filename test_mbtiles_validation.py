#!/usr/bin/env python3
"""Validation script for MBTiles implementation."""
import sqlite3
import sys
from pathlib import Path

def validate_mbtiles(mbtiles_path):
    """Validate MBTiles file structure and content."""
    print(f"Validating MBTiles file: {mbtiles_path}")

    if not Path(mbtiles_path).exists():
        print(f"❌ ERROR: File does not exist: {mbtiles_path}")
        return False

    # Connect to database
    try:
        conn = sqlite3.connect(mbtiles_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ ERROR: Cannot open SQLite database: {e}")
        return False

    # Check required tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    if "tiles" not in tables:
        print("❌ ERROR: Missing 'tiles' table")
        return False
    print("✅ Table 'tiles' exists")

    if "metadata" not in tables:
        print("❌ ERROR: Missing 'metadata' table")
        return False
    print("✅ Table 'metadata' exists")

    # Check tiles schema
    cursor.execute("PRAGMA table_info(tiles)")
    tiles_columns = {row[1]: row[2] for row in cursor.fetchall()}

    required_columns = {
        "zoom_level": "INTEGER",
        "tile_column": "INTEGER",
        "tile_row": "INTEGER",
        "tile_data": "BLOB"
    }

    for col, dtype in required_columns.items():
        if col not in tiles_columns:
            print(f"❌ ERROR: Missing column '{col}' in tiles table")
            return False
        if tiles_columns[col] != dtype:
            print(f"❌ ERROR: Column '{col}' has type '{tiles_columns[col]}' but expected '{dtype}'")
            return False
    print("✅ Tiles table schema correct")

    # Check metadata schema
    cursor.execute("PRAGMA table_info(metadata)")
    metadata_columns = {row[1]: row[2] for row in cursor.fetchall()}

    if "name" not in metadata_columns or "value" not in metadata_columns:
        print("❌ ERROR: Metadata table missing required columns")
        return False
    print("✅ Metadata table schema correct")

    # Check required metadata fields
    cursor.execute("SELECT name, value FROM metadata")
    metadata = dict(cursor.fetchall())

    required_metadata = ["name", "type", "version", "description", "format", "bounds", "minzoom", "maxzoom"]
    for field in required_metadata:
        if field not in metadata:
            print(f"❌ ERROR: Missing required metadata field: {field}")
            return False
    print(f"✅ All required metadata fields present: {list(metadata.keys())}")

    # Verify format is pbf for MVT
    if metadata["format"] != "pbf":
        print(f"❌ ERROR: Expected format='pbf', got '{metadata['format']}'")
        return False
    print("✅ Format is 'pbf' (MVT)")

    # Check tile count
    cursor.execute("SELECT COUNT(*) FROM tiles")
    tile_count = cursor.fetchone()[0]
    if tile_count == 0:
        print("❌ ERROR: No tiles in database")
        return False
    print(f"✅ Found {tile_count} tiles")

    # Check zoom distribution
    cursor.execute("SELECT zoom_level, COUNT(*) FROM tiles GROUP BY zoom_level ORDER BY zoom_level")
    zoom_dist = cursor.fetchall()
    print(f"✅ Zoom distribution:")
    for zoom, count in zoom_dist:
        print(f"   Zoom {zoom}: {count} tiles")

    # Verify zoom range matches metadata
    min_zoom = int(metadata["minzoom"])
    max_zoom = int(metadata["maxzoom"])
    actual_min = min(z for z, _ in zoom_dist)
    actual_max = max(z for z, _ in zoom_dist)

    if actual_min != min_zoom:
        print(f"⚠️  WARNING: Metadata minzoom={min_zoom} but actual min zoom={actual_min}")
    if actual_max != max_zoom:
        print(f"⚠️  WARNING: Metadata maxzoom={max_zoom} but actual max zoom={actual_max}")

    print(f"✅ Zoom range: {actual_min}-{actual_max}")

    # Check tile data is not empty
    cursor.execute("SELECT AVG(LENGTH(tile_data)) FROM tiles")
    avg_size = cursor.fetchone()[0]
    if avg_size == 0:
        print("❌ ERROR: Tiles have zero size")
        return False
    print(f"✅ Average tile size: {avg_size:.0f} bytes")

    # Verify TMS coordinate conversion (check a sample)
    cursor.execute("SELECT zoom_level, tile_column, tile_row FROM tiles WHERE zoom_level=1 LIMIT 1")
    sample = cursor.fetchone()
    if sample:
        z, x, y = sample
        # For TMS, at zoom 1, valid y values are 0 or 1
        if y not in (0, 1):
            print(f"⚠️  WARNING: Unusual TMS coordinate at zoom 1: x={x}, y={y}")
        print(f"✅ Sample tile coordinate (TMS): zoom={z}, x={x}, y={y}")

    # Check for unique index
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tiles'")
    indices = [row[0] for row in cursor.fetchall()]
    if not indices:
        print("⚠️  WARNING: No index on tiles table (performance may be slow)")
    else:
        print(f"✅ Indices present: {indices}")

    # File size
    file_size_mb = Path(mbtiles_path).stat().st_size / (1024 * 1024)
    print(f"✅ File size: {file_size_mb:.2f} MB")

    conn.close()

    print("\n✅ VALIDATION PASSED: MBTiles file is valid!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_mbtiles_validation.py <path-to-mbtiles-file>")
        sys.exit(1)

    mbtiles_path = sys.argv[1]
    success = validate_mbtiles(mbtiles_path)
    sys.exit(0 if success else 1)
