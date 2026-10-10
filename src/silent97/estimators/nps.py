r"""NPS as a two-dimensional mean: indicator helpers and delta-method variance.

Net Promoter Score is a contrast of two proportions of the 0..10 recommendation score
:math:`Y`:

.. math::

    Z = (\mathbf{1}[Y \ge 9],\ \mathbf{1}[Y \le 6]), \qquad
    \mathrm{NPS} = 100\,(E[Z_1] - E[Z_2]) = g^\top E[Z], \qquad g = (100, -100).

Every estimator in this package estimates the 2-vector :math:`E[Z]` (or a corrected
version of it) together with its 2x2 covariance :math:`\Sigma`, and gets the NPS
variance by the delta method, :math:`\mathrm{Var}(\widehat{\mathrm{NPS}}) =
g^\top \Sigma g / n`. Because the map is linear the delta method is exact here, and it
equals the variance of the per-customer value :math:`V = 100 (Z_1 - Z_2) \in
\{-100, 0, 100\}`. Both routes are tested to agree. The cross term
:math:`-2 \cdot 100^2 \,\mathrm{Cov}(Z_1, Z_2)` is never dropped: promoter and
detractor indicators are mutually exclusive and so negatively correlated.

References
----------
Reichheld, F. F. (2003). The one number you need to grow. *Harvard Business Review*,
December 2003, 46-54.

All functions are pure and validate their inputs, raising ``ValueError`` with a message
that states what was expected.
"""

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DETRACTOR_MAX",
    "NPS_GRADIENT",
    "PROMOTER_MIN",
    "delta_method_se",
    "indicator_mean_cov",
    "indicators",
    "nps_from_counts",
    "nps_from_shares",
    "nps_values",
    "nps_variance",
    "validate_predictions",
    "validate_scores",
]

PROMOTER_MIN: Final = 9
"""Scores at or above this are promoters."""

DETRACTOR_MAX: Final = 6
"""Scores at or below this are detractors."""

SCORE_MIN: Final = 0
SCORE_MAX: Final = 10

_ROW_SUM_TOL: Final = 1e-9

NPS_GRADIENT: Final[NDArray[np.float64]] = np.array([100.0, -100.0])
"""Gradient of NPS with respect to (promoter share, detractor share). Read-only."""
NPS_GRADIENT.setflags(write=False)


# --------------------------------------------------------------------------- validation


def validate_scores(y: ArrayLike, *, min_n: int = 1, name: str = "y") -> NDArray[np.int64]:
    """Validate 0..10 recommendation scores and return them as an int64 copy.

    Parameters
    ----------
    y : array_like, shape (n,)
        Scores. Integer dtypes are accepted, and so are floats whose values are all
        integral (e.g. ``9.0``). Booleans, strings, NaN and infinities are rejected.
    min_n : int, default 1
        Minimum number of scores required.
    name : str, default "y"
        Name used in error messages.

    Returns
    -------
    ndarray of int64, shape (n,)

    Raises
    ------
    ValueError
        If ``y`` is not 1-D, has fewer than ``min_n`` entries, is non-numeric, contains
        NaN/inf or non-integral values, or has values outside 0..10.
    """
    arr = np.asarray(y)
    if arr.dtype == np.bool_:
        raise ValueError(f"{name} must hold integer scores 0..10, got a bool array")
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise ValueError(f"{name} must be numeric integer scores 0..10, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D with shape (n,), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; expected at least {max(min_n, 1)} score(s)")
    if arr.size < min_n:
        raise ValueError(f"{name} must have at least {min_n} scores, got {arr.size}")
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN; expected integer scores 0..10")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite; expected integer scores 0..10")
        if not np.all(arr == np.round(arr)):
            raise ValueError(f"{name} must hold integer values 0..10, got non-integral floats")
    if arr.min() < SCORE_MIN or arr.max() > SCORE_MAX:
        raise ValueError(f"{name} must lie in 0..10, got range [{arr.min()}, {arr.max()}]")
    return arr.astype(np.int64, copy=True)


def validate_predictions(p: ArrayLike, *, min_n: int = 1, name: str = "p") -> NDArray[np.float64]:
    """Validate predicted ``[P(promoter), P(detractor)]`` pairs and return a float64 copy.

    Parameters
    ----------
    p : array_like, shape (n, 2)
        Column 0 is P(promoter), column 1 is P(detractor). Each entry must lie in
        [0, 1] (exact 0 and 1 are allowed) and each row must sum to at most 1, since the
        remainder is P(passive).
    min_n : int, default 1
        Minimum number of rows required.
    name : str, default "p"
        Name used in error messages.

    Returns
    -------
    ndarray of float64, shape (n, 2)

    Raises
    ------
    ValueError
        On wrong shape, too few rows, NaN/inf, entries outside [0, 1], or rows summing
        to more than 1 (beyond a 1e-9 rounding tolerance).
    """
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (n, 2) = [P(promoter), P(detractor)], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty; expected at least {max(min_n, 1)} row(s)")
    if arr.shape[0] < min_n:
        raise ValueError(f"{name} must have at least {min_n} rows, got {arr.shape[0]}")
    if not np.isfinite(arr).all():  # one pass; identify NaN only on failure
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN; expected probabilities in [0, 1]")
        raise ValueError(f"{name} must be finite; expected probabilities in [0, 1]")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(f"{name} entries must lie in [0, 1], got range [{arr.min()}, {arr.max()}]")
    row_sum = arr[:, 0] + arr[:, 1]  # faster than a reduction along a length-2 axis
    if row_sum.max() > 1.0 + _ROW_SUM_TOL:
        raise ValueError(
            f"{name} rows must sum to at most 1 (P(promoter) + P(detractor) <= 1), "
            f"got max row sum {row_sum.max()}"
        )
    return arr.copy()


def _validate_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{label} must be an integer count, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return int(value)


# --------------------------------------------------------------------------- indicators


def indicators(y: ArrayLike) -> NDArray[np.float64]:
    """Map 0..10 scores to the indicator matrix ``Z = (1[y >= 9], 1[y <= 6])``.

    Parameters
    ----------
    y : array_like, shape (n,)
        Scores; validated by :func:`validate_scores`.

    Returns
    -------
    ndarray of float64, shape (n, 2)
        Column 0 is the promoter indicator, column 1 the detractor indicator.
    """
    scores = validate_scores(y)
    return np.column_stack((scores >= PROMOTER_MIN, scores <= DETRACTOR_MAX)).astype(np.float64)


def nps_values(y: ArrayLike) -> NDArray[np.float64]:
    """Per-customer NPS value ``V = 100 * (Z_1 - Z_2)`` in {-100, 0, 100}.

    Its mean is the NPS of the scores and its variance equals
    ``nps_variance(cov(Z))``; it exists mainly so tests can check that the two routes
    agree.
    """
    return indicators(y) @ NPS_GRADIENT


# --------------------------------------------------------------------------- point estimates


def nps_from_shares(promoter_share: float, detractor_share: float) -> float:
    """NPS = 100 * (promoter_share - detractor_share).

    Shares are not range-checked, because corrected estimates (e.g. PPI) of a share can
    fall slightly outside [0, 1]; they must be finite.
    """
    prom = float(promoter_share)
    det = float(detractor_share)
    if not (np.isfinite(prom) and np.isfinite(det)):
        raise ValueError(f"shares must be finite, got ({promoter_share}, {detractor_share})")
    return 100.0 * (prom - det)


def nps_from_counts(promoters: int, passives: int, detractors: int) -> float:
    """NPS from category counts: ``100 * (promoters - detractors) / total``.

    Raises
    ------
    ValueError
        If a count is not a non-negative integer or the total is zero.
    """
    n_p = _validate_count(promoters, "promoters")
    n_0 = _validate_count(passives, "passives")
    n_d = _validate_count(detractors, "detractors")
    total = n_p + n_0 + n_d
    if total == 0:
        raise ValueError("total count must be positive, got 0 promoters, passives, detractors")
    return 100.0 * (n_p - n_d) / total


# --------------------------------------------------------------------------- variance


def indicator_mean_cov(
    z: ArrayLike, *, ddof: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample mean (2,) and joint covariance (2, 2) of an ``(n, 2)`` matrix.

    ``z`` is usually the indicator matrix from :func:`indicators`, but any ``(n, 2)``
    real matrix is accepted (e.g. rectifiers ``z - lambda * p`` in PPI).

    Parameters
    ----------
    z : array_like, shape (n, 2)
        Requires n >= 2.
    ddof : {0, 1}, default 0
        Delta degrees of freedom. The default 0 matches ``ndarray.std(0)``, which the
        reference implementation ``ppi_py`` (v0.2.3) uses for its mean intervals.

    Returns
    -------
    mean : ndarray, shape (2,)
    cov : ndarray, shape (2, 2)
        ``sum_i (z_i - mean)(z_i - mean)^T / (n - ddof)``.
    """
    if isinstance(ddof, bool) or ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {ddof!r}")
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"z must have shape (n, 2), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 rows to estimate a covariance, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("z must be finite (no NaN or inf)")
    mean = arr.mean(axis=0)
    centered = arr - mean
    cov = centered.T @ centered / (arr.shape[0] - ddof)
    return mean, cov


def nps_variance(cov: ArrayLike) -> float:
    r"""Delta-method NPS variance :math:`g^\top \Sigma g` with :math:`g = (100, -100)`.

    .. math::

        g^\top \Sigma g = 100^2 \left(\Sigma_{11} + \Sigma_{22} - 2\Sigma_{12}\right)

    Parameters
    ----------
    cov : array_like, shape (2, 2)
        Symmetric covariance of (promoter, detractor) quantities. If ``cov`` is the
        per-observation covariance, the result is the per-observation variance; divide
        by n (see :func:`delta_method_se`) for the variance of the mean.

    Returns
    -------
    float
        Variance on the NPS (points squared) scale.
    """
    arr = np.asarray(cov, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError(f"cov must have shape (2, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("cov must be finite (no NaN or inf)")
    if not np.isclose(arr[0, 1], arr[1, 0], rtol=1e-12, atol=1e-15):
        raise ValueError(f"cov must be symmetric, got off-diagonals {arr[0, 1]}, {arr[1, 0]}")
    return float(NPS_GRADIENT @ arr @ NPS_GRADIENT)


def delta_method_se(cov: ArrayLike, n: int) -> float:
    r"""Standard error of an NPS mean: :math:`\sqrt{g^\top \Sigma g / n}`.

    Parameters
    ----------
    cov : array_like, shape (2, 2)
        Per-observation covariance of the (promoter, detractor) pair.
    n : int
        Number of observations the mean is taken over; must be a positive integer.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return float(np.sqrt(nps_variance(cov) / int(n)))
