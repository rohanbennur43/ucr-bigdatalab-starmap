from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol
import numpy as np
import math
import random


# ------------------------------- Geometry ------------------------------------

@dataclass
class EnvelopeNDLite:
    """Lightweight N-D axis-aligned rectangle (min/max per dimension)."""
    mins: np.ndarray  # shape (D,)
    maxs: np.ndarray  # shape (D,)

    def __post_init__(self):
        self.mins = np.asarray(self.mins, dtype=float)
        self.maxs = np.asarray(self.maxs, dtype=float)
        assert self.mins.shape == self.maxs.shape
        assert self.mins.ndim == 1
        # normalize
        bad = self.mins > self.maxs
        if np.any(bad):
            a = self.mins.copy(); b = self.maxs.copy()
            self.mins = np.minimum(a, b); self.maxs = np.maximum(a, b)

    def getCoordinateDimension(self) -> int:
        return int(self.mins.size)

    def isEmpty(self) -> bool:
        return bool(np.any(self.maxs <= self.mins))

    def getMinCoord(self, d: int) -> float:
        return float(self.mins[d])

    def getMaxCoord(self, d: int) -> float:
        return float(self.maxs[d])

    def setCoordinateDimension(self, d: int):
        if self.mins.size != d:
            self.mins = np.full(d, np.inf)
            self.maxs = np.full(d, -np.inf)

    def setEmpty(self):
        self.mins[:] = np.inf
        self.maxs[:] = -np.inf

    def merge_point(self, coord: np.ndarray):
        self.mins = np.minimum(self.mins, coord)
        self.maxs = np.maximum(self.maxs, coord)

    def merge_box(self, other: "EnvelopeNDLite"):
        self.mins = np.minimum(self.mins, other.mins)
        self.maxs = np.maximum(self.maxs, other.maxs)

    def copy(self) -> "EnvelopeNDLite":
        return EnvelopeNDLite(self.mins.copy(), self.maxs.copy())

    def area(self) -> float:
        side = np.maximum(0.0, self.maxs - self.mins)
        vol = float(np.prod(side))
        return vol

    def margin(self) -> float:
        # R*-tree uses sum of edge lengths (perimeter in 2D, "surface area" proxy in N-D)
        side = np.maximum(0.0, self.maxs - self.mins)
        if side.size == 0:
            return 0.0
        # generalized "Manhattan perimeter": 2 * sum(side) in 2D; here use sum(side) as a proxy
        return float(np.sum(side))

    @staticmethod
    def from_points(coords: np.ndarray) -> "EnvelopeNDLite":
        # coords: (D, N) or (N, D)
        if coords.ndim != 2:
            raise ValueError("coords must be 2-D")
        if coords.shape[0] < coords.shape[1]:  # assume (D,N)
            mins = np.min(coords, axis=1)
            maxs = np.max(coords, axis=1)
        else:  # (N,D)
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
        return EnvelopeNDLite(mins, maxs)


# ----------------------------- Aux Search ------------------------------------

class IntArray(list):
    """Simple stand-in for the Java IntArray."""
    pass


class AuxiliarySearchStructure:
    """
    Linear-scan overlap search over partitions.
    Implemented with NumPy arrays for speed; replaceable by an interval tree/R-tree.
    Covers entire space if expand_to_inf=True was used.
    """
    def __init__(self):
        self.mins: Optional[np.ndarray] = None  # shape (P, D)
        self.maxs: Optional[np.ndarray] = None  # shape (P, D)

    def build(self, boxes: List[EnvelopeNDLite]):
        if not boxes:
            self.mins = self.maxs = None
            return
        D = boxes[0].getCoordinateDimension()
        P = len(boxes)
        self.mins = np.vstack([b.mins for b in boxes])  # (P, D)
        self.maxs = np.vstack([b.maxs for b in boxes])  # (P, D)

    def search(self, mbr: EnvelopeNDLite, out: Optional[IntArray] = None) -> IntArray:
        if out is None:
            out = IntArray()
        out.clear()
        if self.mins is None:
            return out
        # Overlap: not (max <= qmin or qmax <= min) along any dimension
        qmin = mbr.mins[None, :]  # (1, D)
        qmax = mbr.maxs[None, :]  # (1, D)
        sep = (self.maxs <= qmin) | (qmax <= self.mins)   # (P, D)
        disjoint = np.any(sep, axis=1)                    # (P,)
        hits = np.nonzero(~disjoint)[0]
        out.extend(map(int, hits.tolist()))
        return out


# ----------------------------- Histogram API ---------------------------------

class AbstractHistogram(Protocol):
    """Minimal protocol to interoperate with computePointWeights()."""
    def getCoordinateDimension(self) -> int: ...
    def getNumBins(self) -> int: ...
    def getBinID(self, coords: np.ndarray) -> int: ...
    def getBinValue(self, bin_id: int) -> int: ...


# ----------------------------- R*-like splitter -------------------------------

def _bbox_from_indices(coords: np.ndarray, idx: np.ndarray) -> EnvelopeNDLite:
    # coords (D, N); idx (K,)
    mins = np.min(coords[:, idx], axis=1)
    maxs = np.max(coords[:, idx], axis=1)
    return EnvelopeNDLite(mins, maxs)


def _overlap_volume(a: EnvelopeNDLite, b: EnvelopeNDLite) -> float:
    lo = np.maximum(a.mins, b.mins)
    hi = np.minimum(a.maxs, b.maxs)
    side = np.maximum(0.0, hi - lo)
    return float(np.prod(side))


def _choose_split(coords: np.ndarray,
                  idx: np.ndarray,
                  w: Optional[np.ndarray],
                  m: float,
                  M: float,
                  fraction_min_split: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    R*-style split selection on the given subset (indices idx).
    Returns (left_idx, right_idx) as index arrays into 'idx'.
    - Examine each axis
    - For each axis, sort by coordinate, consider candidate split positions
      respecting m (min on each side) and (M) capacity target (soft). We allow
      full range but optionally thin with fraction_min_split in (0..0.5].
    - First criterion: minimal sum of margins of the two groups
    - Tie-breaker: minimal overlap of the two MBRs
    - Tie-breaker: minimal total area
    """
    D, _ = coords.shape
    n = idx.size
    assert n >= 2, "Need at least 2 points to split"

    # Build weights (default = 1)
    if w is None:
        ww = np.ones(n, dtype=float)
    else:
        ww = w[idx].astype(float, copy=False)

    # Candidate split positions must leave >= m on each side (in weight terms).
    # Convert m (min) to weight floor if using weights; otherwise count floor.
    total_w = float(np.sum(ww))
    min_side_w = float(m) if w is None else float(m)  # already weight for weighted mode
    # Boundaries in index space: we'll use cumulative weights to honor m and M
    # Build a helper over a sorted order per axis, so thresholds translate via prefix sums.

    best = None  # (score_margin, score_overlap, score_area, axis, k_split, order)
    best_lr = None

    for axis in range(D):
        order = idx[np.argsort(coords[axis, idx], kind="mergesort")]
        vals = coords[axis, order]
        w_sorted = ww[np.argsort(coords[axis, idx], kind="mergesort")]

        prefix = np.cumsum(w_sorted)  # cum weights
        # valid split positions are between elements: at k means left=[0:k], right=[k:n]
        # require both sides >= min_side_w
        left_ok = prefix >= min_side_w
        right_ok = (total_w - prefix) >= min_side_w
        valid = left_ok & right_ok

        if not np.any(valid):
            # fall back to a median split if nothing valid
            k_candidates = [n // 2]
        else:
            k_valid = np.nonzero(valid)[0]  # positions 0..n-1 (split after k)
            # thin candidates per fraction_min_split (like Java's fraction)
            if fraction_min_split > 0.0:
                # keep a band around mid (e.g., 0.0=all, 0.25=middle 50%)
                lo = int((1.0 - fraction_min_split) * 0.5 * len(k_valid))
                hi = int((1.0 + fraction_min_split) * 0.5 * len(k_valid))
                if hi <= lo:  # ensure at least one candidate
                    lo = 0; hi = len(k_valid)
                k_valid = k_valid[lo:hi]
            k_candidates = k_valid.tolist()

        # Evaluate candidates on R*-criteria
        best_axis = None
        best_axis_lr = None
        for k in k_candidates:
            left_ids = order[:k+1]   # include element k on the left
            right_ids = order[k+1:]
            if left_ids.size == 0 or right_ids.size == 0:
                continue

            mbr_l = _bbox_from_indices(coords, left_ids)
            mbr_r = _bbox_from_indices(coords, right_ids)

            score_margin = mbr_l.margin() + mbr_r.margin()
            score_overlap = _overlap_volume(mbr_l, mbr_r)
            score_area = mbr_l.area() + mbr_r.area()

            cand = (score_margin, score_overlap, score_area)
            if (best_axis is None) or (cand < best_axis):
                best_axis = cand
                best_axis_lr = (left_ids, right_ids)

        if best_axis is None:
            continue

        if (best is None) or (best_axis < best[:3]):
            best = (*best_axis, axis)
            best_lr = best_axis_lr

    if best_lr is None:
        # ultimate fallback: split by first axis median
        order = idx[np.argsort(coords[0, idx], kind="mergesort")]
        k = order.size // 2
        return order[:k], order[k:]
    return best_lr


def _rstar_partition_recursive(coords: np.ndarray,
                               idx: np.ndarray,
                               w: Optional[np.ndarray],
                               min_cap: float,
                               max_cap: float,
                               fraction_min_split: float,
                               out_boxes: List[EnvelopeNDLite],
                               depth: int = 0):
    """
    Recursively partition indices into boxes with capacity in [min_cap, max_cap]
    (capacity = count if w is None, else sum(weights) when w provided).
    """
    import logging
    logger = logging.getLogger("RSGrovePartitioner._rstar_partition_recursive")
    logger.debug(f"Recursion depth {depth}, idx.size={idx.size}")
    if idx.size == 0:
        logger.debug(f"Recursion depth {depth}: idx.size == 0, returning.")
        return

    if w is None:
        cap_here = float(idx.size)
    else:
        cap_here = float(np.sum(w[idx]))

    logger.debug(f"Recursion depth {depth}: cap_here={cap_here}, max_cap={max_cap}")
    if cap_here <= max_cap:
        logger.debug(f"Recursion depth {depth}: cap_here <= max_cap, creating box.")
        out_boxes.append(_bbox_from_indices(coords, idx))
        return

    left, right = _choose_split(coords, idx, w, min_cap, max_cap, fraction_min_split)
    logger.debug(f"Recursion depth {depth}: left.size={left.size}, right.size={right.size}")

    # Ensure both sides make progress (avoid pathological empty splits)
    if left.size == 0 or right.size == 0:
        logger.warning(f"Recursion depth {depth}: Pathological split detected, splitting by median.")
        k = idx.size // 2
        left, right = idx[:k], idx[k:]

    _rstar_partition_recursive(coords, left, w, min_cap, max_cap, fraction_min_split, out_boxes, depth + 1)
    _rstar_partition_recursive(coords, right, w, min_cap, max_cap, fraction_min_split, out_boxes, depth + 1)


def partition_points(coords: np.ndarray,
                     min_cap: int,
                     max_cap: int,
                     expand_to_inf: bool,
                     fraction_min_split: float) -> list:
    import logging
    logger = logging.getLogger("RSGrovePartitioner.partition_points")
    boxes = []
    logger.info(f"Starting _rstar_partition_recursive with {coords.shape[1]} points.")
    _rstar_partition_recursive(coords, np.arange(coords.shape[1], dtype=np.int64), None, float(min_cap), float(max_cap), fraction_min_split, boxes)
    logger.info(f"_rstar_partition_recursive completed. {len(boxes)} boxes created.")
    if expand_to_inf and boxes:
        logger.info("Expanding boxes to infinity.")
        boxes = _expand_to_infinity(boxes)
    return boxes


def partition_weighted_points(coords: np.ndarray,
                              weights: np.ndarray,
                              min_cap_w: float,
                              max_cap_w: float,
                              expand_to_inf: bool,
                              fraction_min_split: float) -> List[EnvelopeNDLite]:
    """Weighted partitioning (capacities based on data sizes)."""
    D, N = coords.shape
    idx = np.arange(N, dtype=np.int64)
    boxes: List[EnvelopeNDLite] = []
    _rstar_partition_recursive(coords, idx, weights.astype(float), float(min_cap_w), float(max_cap_w),
                               fraction_min_split, boxes)
    if expand_to_inf and boxes:
        boxes = _expand_to_infinity(boxes)
    return boxes


def _expand_to_infinity(boxes: List[EnvelopeNDLite]) -> List[EnvelopeNDLite]:
    """Expand outermost boxes to (-inf,+inf) per dimension to guarantee full coverage."""
    if not boxes:
        return boxes
    D = boxes[0].getCoordinateDimension()
    mins = np.vstack([b.mins for b in boxes])   # (P,D)
    maxs = np.vstack([b.maxs for b in boxes])   # (P,D)
    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)

    out: List[EnvelopeNDLite] = []
    for i, b in enumerate(boxes):
        mm = b.mins.copy()
        xx = b.maxs.copy()
        # if equals global min on a dim, extend to -inf; if equals global max, extend to +inf
        on_min = (np.isclose(mm, global_min) | (mm <= global_min))
        on_max = (np.isclose(xx, global_max) | (xx >= global_max))
        mm = np.where(on_min, -np.inf, mm)
        xx = np.where(on_max, +np.inf, xx)
        out.append(EnvelopeNDLite(mm, xx))
    return out


# --------------------------- Partitioner (public) -----------------------------

class BeastOptions(dict):
    """Very small substitute to provide .getDouble/.getBoolean like the Java conf."""
    def getDouble(self, key: str, default: float) -> float:
        return float(self.get(key, default))
    def getBoolean(self, key: str, default: bool) -> bool:
        return bool(self.get(key, default))


class RSGrovePartitioner:
    """
    Python implementation inspired by Beast's RSGrovePartitioner with R*-style splitting.

    Methods:
      - setup(conf: BeastOptions, disjoint: bool)
      - construct(summary, sample, histogram, numPartitions)
      - overlapPartitions(mbr: EnvelopeNDLite, out: Optional[IntArray]) -> IntArray
      - overlapPartition(mbr: EnvelopeNDLite) -> int
      - numPartitions() -> int
      - isDisjoint() -> bool
      - getCoordinateDimension() -> int
      - getPartitionMBR(partitionID, mbr_out: EnvelopeNDLite)
      - getEnvelope() -> EnvelopeNDLite

    Notes:
      * `summary` must expose:
          - getCoordinateDimension()
          - getMinCoord(d), getMaxCoord(d)
          - getSideLength(d)
          - (or) mins/maxs arrays; we only need the global MBR.
      * If `histogram` is provided (AbstractHistogram), weighted partitioning is used.
    """

    # Config keys (mirror Java constants)
    MMRatio = "mmratio"
    MinSplitRatio = "RSGrove.MinSplitRatio"
    ExpandToInfinity = "RSGrove.ExpandToInf"

    def __init__(self):
        import logging
        self.logger = logging.getLogger("RSGrovePartitioner")
        # Config / state
        self.disjointPartitions: bool = True
        self.mMRatio: float = 0.95
        self.fractionMinSplitSize: float = 0.0
        self.expandToInf: bool = True

        # Geometry / partitions
        self.mbrPoints: EnvelopeNDLite = EnvelopeNDLite(np.array([np.inf]), np.array([-np.inf]))
        self.minCoord: Optional[np.ndarray] = None  # (D, P)
        self.maxCoord: Optional[np.ndarray] = None  # (D, P)

        self.aux: AuxiliarySearchStructure = AuxiliarySearchStructure()
        self._rng = random.Random()

    # ---------- API parity ----------

    def setup(self, conf: BeastOptions, disjoint: bool):
        self.logger.info(f"Setting up partitioner: disjoint={disjoint}")
        self.disjointPartitions = bool(disjoint)
        self.mMRatio = conf.getDouble(self.MMRatio, 0.95)
        self.fractionMinSplitSize = conf.getDouble(self.MinSplitRatio, 0.0)
        self.expandToInf = conf.getBoolean(self.ExpandToInfinity, True)
        self.logger.info(f"Config: mMRatio={self.mMRatio}, MinSplitRatio={self.fractionMinSplitSize}, ExpandToInf={self.expandToInf}")

    def construct(self,
                  summary,
                  sample: np.ndarray,
                  histogram: Optional[AbstractHistogram],
                  numPartitions: int):
        self.logger.info(f"Constructing partitions with numPartitions={numPartitions}")
        """
        summary: exposes coordinate dimension and bounds (min/max per dim or getMinCoord/getMaxCoord)
        sample: np.ndarray with shape (D, N)
        histogram: optional AbstractHistogram (weighted mode)
        """
        # Handle empty sample: fabricate uniform points within summary MBR (like Java)
        def _summary_dims():
            return int(summary.getCoordinateDimension())

        D = _summary_dims()
        self.logger.info(f"Summary dimension: {D}")
        # Merge summary to mbrPoints
        mins = np.array([summary.getMinCoord(d) for d in range(D)], dtype=float)
        maxs = np.array([summary.getMaxCoord(d) for d in range(D)], dtype=float)
        self.mbrPoints = EnvelopeNDLite(mins, maxs)

        if sample.size == 0:
            self.logger.warning("Sample is empty, fabricating points within summary MBR.")
            Nf = 1000
            fabricated = np.zeros((D, Nf), dtype=float)
            for d in range(D):
                fabricated[d, :] = np.random.rand(Nf) * (maxs[d] - mins[d]) + mins[d]
            sample = fabricated

        assert self.mMRatio > 0, "mMRatio cannot be zero. Call setup() first."

        if sample.shape[0] != D:
            self.logger.error(f"Sample shape mismatch: got {sample.shape[0]}, expected {D}")
            raise ValueError(f"sample must have shape (D, N) with D={D}")

        N = sample.shape[1]
        self.logger.info(f"Sample size: {N}")

        if histogram is None:
            # Unweighted mode: choose M,m from sample count
            M = int(math.ceil(N / float(numPartitions)))
            m = int(math.ceil(self.mMRatio * M))
            self.logger.info(f"Unweighted partitioning: M={M}, m={m}")
            self.logger.info(f"Calling partition_points with sample shape {sample.shape}, min_cap={m}, max_cap={M}, expand_to_inf={self.expandToInf}, fraction_min_split={self.fractionMinSplitSize}")
            boxes = partition_points(sample, m, M, self.expandToInf, self.fractionMinSplitSize)
            self.logger.info(f"partition_points returned {len(boxes)} boxes.")
        else:
            # Weighted mode: compute point weights from histogram, then split by total size
            weights = self.computePointWeights(sample, histogram)  # long[] in Java
            total_size = float(np.sum(weights))
            M = float(math.ceil(total_size / float(numPartitions)))
            m = float(total_size * self.mMRatio / float(numPartitions))
            self.logger.info(f"Weighted partitioning: total_size={total_size}, M={M}, m={m}")
            boxes = partition_weighted_points(sample, weights, m, M, self.expandToInf, self.fractionMinSplitSize)

        # Store min/max arrays
        P = len(boxes)
        self.logger.info(f"Constructed {P} partition boxes.")
        self.minCoord = np.vstack([b.mins for b in boxes]).T  # (D, P)
        self.maxCoord = np.vstack([b.maxs for b in boxes]).T  # (D, P)

        # Build auxiliary search structure
        self.logger.info("Building auxiliary search structure.")
        self.aux.build(boxes)

    # ---------- Helpers (API-compatible) ----------

    @staticmethod
    def computePointWeights(sample: np.ndarray, histogram: AbstractHistogram) -> np.ndarray:
        D, N = sample.shape
        assert D == histogram.getCoordinateDimension()
        num_bins = histogram.getNumBins()
        counts = np.zeros(num_bins, dtype=int)
        # First pass: count points per bin
        tmp = np.empty(D, dtype=float)
        bin_ids = np.empty(N, dtype=int)
        for i in range(N):
            tmp[:] = sample[:, i]
            b = histogram.getBinID(tmp)
            bin_ids[i] = b
            counts[b] += 1
        # Second pass: distribute bin weight to points in the bin
        weights = np.zeros(N, dtype=np.int64)
        for i in range(N):
            b = bin_ids[i]
            bin_val = histogram.getBinValue(b)
            w = 0 if counts[b] == 0 else int(bin_val // max(1, counts[b]))
            weights[i] = max(0, w)
        return weights

    def numPartitions(self) -> int:
        return 0 if self.minCoord is None else int(self.minCoord.shape[1])

    def isDisjoint(self) -> bool:
        return bool(self.disjointPartitions)

    def getCoordinateDimension(self) -> int:
        if self.minCoord is None:
            return 0
        return int(self.minCoord.shape[0])

    def overlapPartitions(self, mbr: EnvelopeNDLite, out: Optional[IntArray] = None) -> IntArray:
        if out is None:
            out = IntArray()
        out.clear()
        if mbr.isEmpty():
            # Match Java behavior: assign empty to a random partition for load-balance
            if self.numPartitions() > 0:
                out.append(self._rng.randrange(self.numPartitions()))
            return out
        return self.aux.search(mbr, out)

    def _partition_expansion(self, pid: int, env: EnvelopeNDLite) -> float:
        # Compute expansion within the global mbr bounds (like Java's Partition_expansion)
        D = self.getCoordinateDimension()
        vol_before = 1.0
        vol_after = 1.0
        for d in range(D):
            mb = max(self.mbrPoints.getMinCoord(d), self.minCoord[d, pid])
            xb = min(self.mbrPoints.getMaxCoord(d), self.maxCoord[d, pid])
            ma = max(self.mbrPoints.getMinCoord(d), min(self.minCoord[d, pid], env.getMinCoord(d)))
            xa = min(self.mbrPoints.getMaxCoord(d), max(self.maxCoord[d, pid], env.getMaxCoord(d)))
            vol_before *= max(0.0, xb - mb)
            vol_after *= max(0.0, xa - ma)
        return vol_after - vol_before

    def _partition_volume(self, pid: int) -> float:
        side = np.maximum(0.0, self.maxCoord[:, pid] - self.minCoord[:, pid])
        return float(np.prod(side))

    def overlapPartition(self, mbr: EnvelopeNDLite) -> int:
        if self.numPartitions() == 0:
            return -1
        if mbr.isEmpty():
            return self._rng.randrange(self.numPartitions())
        tmp = self.overlapPartitions(mbr, None)
        if len(tmp) == 1:
            return tmp[0]
        # Choose by minimal expansion; break ties by smaller area
        chosen = -1
        best_exp = float("inf")
        best_area = float("inf")
        for pid in tmp:
            exp = self._partition_expansion(pid, mbr)
            if exp < best_exp:
                best_exp = exp
                best_area = self._partition_volume(pid)
                chosen = pid
            elif exp == best_exp:
                vol = self._partition_volume(pid)
                if vol < best_area:
                    best_area = vol
                    chosen = pid
        return chosen if chosen >= 0 else tmp[0] if tmp else self._rng.randrange(self.numPartitions())

    def getPartitionMBR(self, partitionID: int, mbr_out: EnvelopeNDLite):
        D = self.getCoordinateDimension()
        mbr_out.setCoordinateDimension(D)
        mbr_out.setEmpty()
        mbr_out.mins = self.minCoord[:, partitionID].copy()
        mbr_out.maxs = self.maxCoord[:, partitionID].copy()

    def getEnvelope(self) -> EnvelopeNDLite:
        return self.mbrPoints
