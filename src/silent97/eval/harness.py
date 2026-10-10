"""Experiment harness: run (estimator x scenario x n x replicate) grids.

This module draws the surveyed customers and enforces that the labeler's training
data is disjoint from every labeled inference set (CLAUDE.md section 6, rule 4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["check_disjoint", "draw_respondents"]


def draw_respondents(
    pool: ArrayLike, weights: ArrayLike, n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Draw ``n`` distinct customers from ``pool`` with probability proportional to ``weights``.

    Uses Efraimidis-Spirakis weighted sampling without replacement (keys
    ``log(U) / w``; keep the ``n`` largest). This is successive sampling: each draw
    picks a remaining unit with probability proportional to its weight. With equal
    weights it is simple random sampling without replacement. With n much smaller
    than the pool, unit i's inclusion probability is close to ``n * w_i / sum(w)``.

    Parameters
    ----------
    pool : array_like of int, shape (M,)
        Candidate customer indices.
    weights : array_like of float, shape (M,)
        Strictly positive, finite weights (e.g. response propensities).
    n : int
        Number to draw, 1 <= n <= M.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int64, shape (n,), sorted ascending.
    """
    pool_arr = np.asarray(pool, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != pool_arr.shape:
        raise ValueError(f"weights length {w.shape} must match pool length {pool_arr.shape}")
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if n > pool_arr.size:
        raise ValueError(f"n = {n} exceeds the pool size {pool_arr.size}")
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    u = 1.0 - rng.random(pool_arr.size)  # in (0, 1], so log(u) is finite
    keys = np.log(u) / w
    chosen = np.argpartition(-keys, n - 1)[:n]
    return np.sort(pool_arr[chosen])


def check_disjoint(train_idx: ArrayLike, labeled_idx: ArrayLike) -> None:
    """Raise ``ValueError`` if any labeled inference index was used to train the labeler."""
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(labeled_idx))
    if overlap.size:
        raise ValueError(
            f"training leak: {overlap.size} labeled customer(s) were in the labeler's "
            f"training set (e.g. {overlap[:5].tolist()}); only ppi_crossfit may reuse "
            "labeled data, and it does so by cross-fitting"
        )
