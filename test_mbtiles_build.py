#!/usr/bin/env python3
"""End-to-end test for MBTiles build functionality."""
import sqlite3
import tempfile
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

def create_test_dataset():
    """Create a small test GeoParquet file."""
    # Create simple point data
    points = []
    for i in range(100):
        lon = -120 + (i % 10) * 0.1
        lat = 35 + (i // 10) * 0.1
        points.append({
            'geometry': Point(lon, lat),
            'id': i,
            'name': f'Point {i}'
        })

    gdf = gpd.GeoDataFrame(points, crs='EPSG:4326')
    return gdf


def test_build_with_mbtiles():
    """Test full build pipeline with MBTiles export."""
    print("Creating test dataset...")
    gdf = create_test_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save test data
        input_file = tmpdir / "test_data.parquet"
        gdf.to_parquet(input_file)
        print(f"✅ Created test data: {input_file} ({len(gdf)} features)")

        # Test with Python API
        print("\n=== Testing Python API ===")
        import starlet

        output_dir = tmpdir / "test_output"

        tile_result, mvt_result = starlet.build(
            input=str(input_file),
            outdir=str(output_dir),
            num_tiles=5,
            zoom=4,
            threshold=0,
            mbtiles=True,
        )

        print(f"✅ Tiling: {tile_result.num_files} tiles, {tile_result.total_rows} rows")
        print(f"✅ MVT: {mvt_result.tile_count} tiles, zoom {mvt_result.zoom_levels}")
        print(f"✅ MBTiles: {mvt_result.mbtiles_path}")

        # Verify MBTiles file
        assert mvt_result.mbtiles_path is not None, "MBTiles path should not be None"
        mbtiles_path = Path(mvt_result.mbtiles_path)
        assert mbtiles_path.exists(), f"MBTiles file not found: {mbtiles_path}"

        # Verify contents
        conn = sqlite3.connect(str(mbtiles_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM tiles")
        tile_count = cursor.fetchone()[0]
        assert tile_count > 0, "MBTiles should contain tiles"
        print(f"✅ MBTiles contains {tile_count} tiles")

        cursor.execute("SELECT value FROM metadata WHERE name='format'")
        format_value = cursor.fetchone()[0]
        assert format_value == 'pbf', f"Expected format='pbf', got '{format_value}'"
        print("✅ Format is 'pbf'")

        cursor.execute("SELECT value FROM metadata WHERE name='name'")
        name_value = cursor.fetchone()[0]
        print(f"✅ Dataset name: {name_value}")

        conn.close()

        print("\n✅ ALL TESTS PASSED!")
        return True


if __name__ == "__main__":
    try:
        test_build_with_mbtiles()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
