"""Shared input validation and normal-interval construction for NPS estimators."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from silent97.estimators.nps import indicators, validate_predictions, validate_scores

__all__ = ["normal_interval", "validate_alpha", "validate_labeled", "validate_unlabeled"]


def validate_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float, raising ``ValueError`` unless 0 < alpha < 1."""
    a = float(alpha)
    if not (math.isfinite(a) and 0.0 < a < 1.0):
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    return a


def normal_interval(point: float, se: float, alpha: float) -> tuple[float, float]:
    """Two-sided Wald interval ``point -/+ z_{1 - alpha/2} * se``.

    This is the interval ``ppi_py`` builds with statsmodels' ``_zconfint_generic``.
    """
    half = float(norm.ppf(1.0 - alpha / 2.0)) * se
    return point - half, point + half


def validate_labeled(
    y_lab: ArrayLike, p_lab: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate labeled scores (n >= 2) and, if given, their predictions (n, 2).

    Returns the indicator matrix ``Z`` (n, 2) and the validated predictions.
    """
    scores = validate_scores(y_lab, min_n=2, name="y_lab")
    z = indicators(scores)
    if p_lab is None:
        return z, None
    p = validate_predictions(p_lab, name="p_lab")
    if p.shape[0] != z.shape[0]:
        raise ValueError(
            f"p_lab must have one row per labeled score: {p.shape[0]} rows vs {z.shape[0]} scores"
        )
    return z, p


def validate_unlabeled(p_unlab: ArrayLike) -> NDArray[np.float64]:
    """Validate unlabeled predictions, shape (N, 2) with N >= 1."""
    return validate_predictions(p_unlab, name="p_unlab")
