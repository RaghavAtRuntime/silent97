"""Core result and design types shared by every estimator and sampling design.

These interfaces are fixed by CLAUDE.md section 5; do not change them without approval.
"""

from typing import Any, NamedTuple

import numpy as np


class Estimate(NamedTuple):
    """Result of an NPS estimator.

    Attributes
    ----------
    point : float
        Point estimate of NPS on the -100..100 scale.
    lower, upper : float
        Bounds of the (1 - alpha) confidence interval, same scale as ``point``.
    alpha : float
        Miscoverage level; the interval targets coverage ``1 - alpha``.
    se : float
        Standard error of ``point``, same scale.
    n_labeled : int
        Number of surveyed (labeled) customers used.
    n_unlabeled : int
        Number of unlabeled customers whose predictions were used.
    method : str
        Estimator name, e.g. ``"classical"`` or ``"ppi_pp"``.
    diagnostics : dict
        Method-specific extras such as ``lambda_hat``, ``effective_n``, a weight
        summary or ``rhat``.
    """

    point: float
    lower: float
    upper: float
    alpha: float
    se: float
    n_labeled: int
    n_unlabeled: int
    method: str
    diagnostics: dict[str, Any]


class Design(NamedTuple):
    """A realized sampling design.

    Attributes
    ----------
    idx : np.ndarray
        Indices (into the population) of the surveyed customers.
    pi : np.ndarray
        Inclusion probability for EVERY customer in the population, shape ``(N,)``.
        Every entry must be strictly positive.
    name : str
        Design name, e.g. ``"uniform"`` or ``"active"``.
    """

    idx: np.ndarray
    pi: np.ndarray
    name: str
