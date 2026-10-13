r"""Delta-adjustment sensitivity analysis for nonresponse that covariates cannot explain.

Every estimator in this package is valid only if, after its correction (predictions,
cells, weights), non-respondents look like respondents: missing at random. Under
self-selection on satisfaction itself (MNAR) no method removes the bias. This module
answers the follow-up question: *how different would non-respondents have to be to
change the conclusion?*

Assumption family (delta adjustment; docs/PLAN.md, "Sensitivity analysis", reviewer
decision Q3 in docs/progress/phase-4.md). Let :math:`(P_U, D_U)` be the promoter and
detractor shares that the base estimator implicitly assigns to the non-respondents
(the "model-corrected respondent rates"). Under :math:`\delta` (percentage points), the
non-respondents' true shares are

.. math::

    P_U(\delta) = P_U + \delta / 100, \qquad D_U(\delta) = D_U - \delta / 100,

so their NPS moves by :math:`2\delta` points. With :math:`n` respondents and :math:`N`
non-respondents in a population of :math:`M = n + N`, and :math:`w_U = N / M`, the
population NPS is :math:`(n\,\mathrm{NPS}_R + N\,\mathrm{NPS}_U) / M`, hence

.. math::

    \hat\theta(\delta) = \hat\theta_0 + 2\,\delta\,w_U, \qquad
    [L(\delta), U(\delta)] = [L_0 + 2\delta w_U,\; U_0 + 2\delta w_U].

:math:`\delta` is an assumption, not an estimate, so it adds no sampling variance: the
interval shifts with the point and keeps the base standard error. For the NPS target the
family is complete: NPS depends on the shares only through :math:`P - D`, so any shift of
the non-respondents' shares acts on NPS like :math:`\delta = \Delta(P - D) \cdot 100 / 2`.

The breakpoint :math:`\delta^*` of a conclusion "NPS > c" is the :math:`\delta` closest
to zero at which the conclusion no longer holds at the interval's edge,
:math:`L(\delta^*) = c`, i.e. :math:`\delta^* = (c - L_0) / (2 w_U)` (negative: the
conclusion fails if non-respondents are less satisfied). For "NPS < c",
:math:`\delta^* = (c - U_0) / (2 w_U)`. If the conclusion does not hold at
:math:`\delta = 0`, :math:`\delta^* = 0`.

Validity: the base interval is valid under MAR given its correction; the shifted
interval is valid if non-respondents differ from the implied rates by exactly
:math:`\delta` on the NPS scale. The family does not model nonresponse among
respondents' own answers (measurement error), and shares outside [0, 1] are possible for
large :math:`|\delta|`: pass the implied shares to mark the feasible range.

Reference: the delta-adjustment (pattern-mixture) approach to MNAR sensitivity, as in
National Research Council (2010), *The Prevention and Treatment of Missing Data in
Clinical Trials*, ch. 5, and Little & Rubin (2019), *Statistical Analysis with Missing
Data*, 3rd ed., ch. 15.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.types import Estimate

__all__ = [
    "DIRECTIONS",
    "SensitivityCurve",
    "breakpoint",
    "delta_adjust",
    "feasible_range",
    "implied_nonrespondent_shares",
]

DIRECTIONS = ("above", "below")

FloatArray = NDArray[np.float64]


class SensitivityCurve(NamedTuple):
    """NPS and its interval as a function of :math:`\\delta` (percentage points)."""

    delta: FloatArray
    point: FloatArray
    lower: FloatArray
    upper: FloatArray
    nonrespondent_share: float
    delta_range: tuple[float, float] | None
    """Feasible :math:`\\delta` range, if the implied shares were given."""
    feasible: NDArray[np.bool_]


def _check_share(w: float) -> float:
    w = float(w)
    if not 0.0 <= w <= 1.0 or math.isnan(w):
        raise ValueError(f"nonrespondent_share must be in [0, 1], got {w}")
    return w


def _check_base(base: Estimate) -> None:
    if not all(math.isfinite(v) for v in (base.point, base.lower, base.upper)):
        raise ValueError("base estimate must have a finite point and interval")


def _check_shares(shares: Sequence[float], name: str = "shares") -> tuple[float, float]:
    if len(shares) != 2:
        raise ValueError(f"{name} must be (promoter, detractor), got {shares!r}")
    p, d = float(shares[0]), float(shares[1])
    if not (0.0 <= p <= 1.0 and 0.0 <= d <= 1.0 and p + d <= 1.0 + 1e-12):
        raise ValueError(f"{name} must be in [0, 1] with promoter + detractor <= 1, got {p, d}")
    return p, d


def implied_nonrespondent_shares(
    population_shares: Sequence[float],
    respondent_shares: Sequence[float],
    *,
    n_respondents: int,
    n_nonrespondents: int,
) -> tuple[float, float]:
    """The non-respondents' (promoter, detractor) shares implied by a base estimate.

    Solves :math:`\\hat\\theta = (n \\bar Z_R + N \\theta_U) / (n + N)` for
    :math:`\\theta_U`, component-wise: ``population_shares`` are the base estimate's
    shares, ``respondent_shares`` the respondents' observed shares. The result is not
    clipped; values outside [0, 1] mean the base estimate itself implies infeasible
    non-respondent rates.
    """
    if n_respondents < 0 or n_nonrespondents < 1:
        raise ValueError(
            "need n_respondents >= 0 and n_nonrespondents >= 1, "
            f"got {n_respondents}, {n_nonrespondents}"
        )
    m = n_respondents + n_nonrespondents
    pop = np.asarray(population_shares, dtype=float)
    resp = np.asarray(respondent_shares, dtype=float)
    if pop.shape != (2,) or resp.shape != (2,):
        raise ValueError("shares must be (promoter, detractor)")
    u = (m * pop - n_respondents * resp) / n_nonrespondents
    return float(u[0]), float(u[1])


def feasible_range(nonrespondent_shares: Sequence[float]) -> tuple[float, float]:
    """The :math:`\\delta` range (pp) keeping both adjusted shares within [0, 1].

    :math:`P_U + \\delta/100 \\in [0, 1]` and :math:`D_U - \\delta/100 \\in [0, 1]`
    (their sum is unchanged).
    """
    p, d = _check_shares(nonrespondent_shares)
    lo = max(-p, d - 1.0)
    hi = min(1.0 - p, d)
    return 100.0 * lo, 100.0 * hi


def delta_adjust(
    base: Estimate,
    deltas: ArrayLike,
    *,
    nonrespondent_share: float,
    nonrespondent_shares: Sequence[float] | None = None,
) -> SensitivityCurve:
    """Shift a base NPS estimate by :math:`2\\delta w_U` for each :math:`\\delta`.

    Parameters
    ----------
    base
        The estimate under MAR (any estimator in this package).
    deltas
        Percentage-point shifts: non-respondents' promoter share :math:`+\\delta`,
        detractor share :math:`-\\delta`, relative to the base estimate's implied rates.
    nonrespondent_share
        :math:`w_U = N / (n + N)`, the fraction of the target population that did not
        respond.
    nonrespondent_shares
        Optional implied (promoter, detractor) shares of the non-respondents
        (:func:`implied_nonrespondent_shares`), used to mark the feasible range.
    """
    _check_base(base)
    w = _check_share(nonrespondent_share)
    d = np.atleast_1d(np.asarray(deltas, dtype=float))
    if d.ndim != 1 or d.size == 0 or not np.isfinite(d).all():
        raise ValueError("deltas must be a non-empty 1-D array of finite values")
    shift = 2.0 * d * w
    if nonrespondent_shares is None:
        delta_range = None
        feasible = np.ones(d.shape, dtype=bool)
    else:
        delta_range = feasible_range(nonrespondent_shares)
        tol = 1e-9
        feasible = (d >= delta_range[0] - tol) & (d <= delta_range[1] + tol)
    return SensitivityCurve(
        delta=d,
        point=base.point + shift,
        lower=base.lower + shift,
        upper=base.upper + shift,
        nonrespondent_share=w,
        delta_range=delta_range,
        feasible=feasible,
    )


def breakpoint(
    base: Estimate,
    *,
    nonrespondent_share: float,
    threshold: float,
    direction: str = "above",
) -> float:
    """:math:`\\delta^*` (pp) at which "NPS above/below ``threshold``" stops holding.

    Signed: for ``"above"`` it is :math:`(c - L_0)/(2 w_U) \\le 0`, for ``"below"``
    :math:`(c - U_0)/(2 w_U) \\ge 0`. Returns 0 if the conclusion does not hold at
    :math:`\\delta = 0`, and :math:`\\mp\\infty` if it holds and :math:`w_U = 0`
    (everyone responded, so no assumption about non-respondents can flip it).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    _check_base(base)
    w = _check_share(nonrespondent_share)
    c = float(threshold)
    if direction == "above":
        if base.lower <= c:
            return 0.0
        return (c - base.lower) / (2.0 * w) if w > 0 else -math.inf
    if base.upper >= c:
        return 0.0
    return (c - base.upper) / (2.0 * w) if w > 0 else math.inf
