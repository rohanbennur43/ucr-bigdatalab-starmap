"""Public result types and dataset introspection for starlet."""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class TileResult:
    """Result returned by :func:`starlet.tile`."""
    outdir: str
    num_files: int
    total_rows: int
    bbox: Tuple[float, float, float, float]
    histogram_path: str


@dataclasses.dataclass(frozen=True)
class MVTResult:
    """Result returned by :func:`starlet.generate_mvt`."""
    outdir: str
    zoom_levels: List[int]
    tile_count: int
    mbtiles_path: Optional[str] = None


class Dataset:
    """Read-only introspection object for a starlet dataset directory.

    A dataset directory is expected to contain at least ``parquet_tiles/``
    and optionally ``histograms/``, ``mvt/``, and ``stats/``.

    Parameters
    ----------
    path : str
        Path to the dataset root directory.
    """

    def __init__(self, path: str) -> None:
        self._root = Path(path)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")

    @property
    def path(self) -> str:
        return str(self._root)

    @property
    def num_tiles(self) -> int:
        tiles_dir = self._root / "parquet_tiles"
        if not tiles_dir.exists():
            return 0
        return len(list(tiles_dir.glob("*.parquet")))

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        stats_path = self._root / "stats" / "attributes.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                for attr in stats.get("attributes", []):
                    if attr["name"] == "geometry":
                        mbr = attr["stats"].get("mbr")
                        if mbr and len(mbr) == 4:
                            return tuple(mbr)
            except Exception:
                pass
        return None

    @property
    def zoom_levels(self) -> List[int]:
        mvt_dir = self._root / "mvt"
        if not mvt_dir.exists():
            return []
        levels = []
        for child in mvt_dir.iterdir():
            if child.is_dir():
                try:
                    levels.append(int(child.name))
                except ValueError:
                    pass
        return sorted(levels)

    @property
    def has_histograms(self) -> bool:
        return (self._root / "histograms" / "global_prefix.npy").exists()

    @property
    def has_mvt(self) -> bool:
        return (self._root / "mvt").is_dir()

    @property
    def has_stats(self) -> bool:
        return (self._root / "stats" / "attributes.json").exists()

    def __repr__(self) -> str:
        return f"Dataset({self._root!s}, tiles={self.num_tiles})"
