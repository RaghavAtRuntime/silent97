r"""Stratified PPI for NPS: ``ppi_stratified``.

Setting: the pool is split into strata h with sizes :math:`M_h` (surveyed plus
unsurveyed), and the survey is a simple random sample within each stratum
(:func:`silent97.sampling.designs.stratified`), possibly with unequal sampling rates
(Neyman allocation). In each stratum, PPI is applied to its own surveyed and
unsurveyed customers, giving :math:`\widehat{\mathrm{NPS}}_h` and its standard error
:math:`\widehat{\mathrm{se}}_h` from the 2-D (promoter, detractor) covariance. The
strata are then combined with *population* weights :math:`W_h = M_h / M`:

.. math::

    \widehat{\mathrm{NPS}} = \sum_h W_h \widehat{\mathrm{NPS}}_h, \qquad
    \widehat{\mathrm{se}}^2 = \sum_h W_h^2 \widehat{\mathrm{se}}_h^2,

since the strata are sampled independently (Fisch et al. 2024). The weights undo
unequal sampling rates across strata, which is the stratified counterpart of the
:math:`1/\pi` weights (:math:`\pi_i = n_h / M_h` within stratum h).

``lam`` chooses each stratum's :math:`\lambda_h`:

- ``"one"`` (default): plain PPI, :math:`\lambda_h = 1`;
- ``"plugin"``: PPI++ per stratum (Fisch et al.), with the same-sample plug-in, which
  has the small-n bias documented in Phase 3;
- ``"crossfit"``: :func:`silent97.estimators.ppi.ppi_pp_cf` per stratum, which needs
  ``rng``.

Validity: simple random sampling within strata, with at least 2 surveyed and at least
1 unsurveyed customer per stratum (enough surveyed per stratum for its variance
estimate to be reliable: a heavily weighted stratum with only a handful of responses
under-covers, e.g. 0.82 with 7 responses in a stratum of weight 0.46; ``min_n_h`` is
reported in the diagnostics); stratum labels are known for everyone; the model is
fixed independently of the surveyed set.

Reference: Fisch, Maynez, Hofer, Dhingra, Globerson, Cohen (2024). Stratified
prediction-powered inference for effective hybrid evaluation of language models.
*NeurIPS*. arXiv:2406.04291.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike

from silent97.estimators._common import normal_interval, validate_alpha
from silent97.estimators.ppi import _validated, ppi, ppi_pp, ppi_pp_cf
from silent97.types import Estimate

__all__ = ["LAMBDA_MODES", "ppi_stratified"]

LAMBDA_MODES: Final = ("one", "plugin", "crossfit")


def ppi_stratified(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    strata_lab: ArrayLike,
    strata_unlab: ArrayLike,
    alpha: float = 0.1,
    lam: str = "one",
    rng: np.random.Generator | None = None,
    n_folds: int = 5,
) -> Estimate:
    """Stratified PPI NPS (see module docstring).

    ``strata_lab`` (n,) and ``strata_unlab`` (N,) give each customer's stratum label.
    """
    a = validate_alpha(alpha)
    z, p_l, p_u = _validated(y_lab, p_lab, p_unlab)
    s_l, s_u = np.asarray(strata_lab), np.asarray(strata_unlab)
    if s_l.shape != (z.shape[0],) or s_u.shape != (p_u.shape[0],):
        raise ValueError("strata_lab and strata_unlab must label every labeled/unlabeled row")
    if lam not in LAMBDA_MODES:
        raise ValueError(f"lam must be one of {LAMBDA_MODES}, got {lam!r}")
    if lam == "crossfit" and rng is None:
        raise ValueError("lam='crossfit' needs rng (a numpy Generator) for the fold splits")
    y = np.asarray(y_lab)
    labels = np.unique(np.concatenate([s_l, s_u]))
    m_total = s_l.size + s_u.size
    point, var = 0.0, 0.0
    per_stratum: list[dict[str, Any]] = []
    min_n_h = z.shape[0]
    for h in labels:
        in_l, in_u = s_l == h, s_u == h
        n_h, big_n_h = int(in_l.sum()), int(in_u.sum())
        if n_h < 2:
            raise ValueError(f"stratum {h!r} needs at least 2 surveyed customers, got {n_h}")
        if big_n_h < 1:
            raise ValueError(f"stratum {h!r} has no unlabeled customers")
        args = (y[in_l], p_l[in_l], p_u[in_u])
        if lam == "one":
            e = ppi(*args, alpha=a)
        elif lam == "plugin":
            e = ppi_pp(*args, alpha=a)
        else:
            assert rng is not None
            e = ppi_pp_cf(*args, alpha=a, n_folds=min(n_folds, n_h), rng=rng)
        min_n_h = min(min_n_h, n_h)
        w = (n_h + big_n_h) / m_total
        point += w * e.point
        var += w**2 * e.se**2
        per_stratum.append(
            {
                "stratum": str(h),
                "weight": w,
                "n": n_h,
                "point": e.point,
                "se": e.se,
                "lambda_hat": e.diagnostics.get("lambda_hat"),
            }
        )
    se = math.sqrt(var)
    lower, upper = normal_interval(point, se, a)
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=a,
        se=se,
        n_labeled=z.shape[0],
        n_unlabeled=p_u.shape[0],
        method="ppi_stratified",
        diagnostics={
            "n_strata": int(labels.size),
            "min_n_h": min_n_h,
            "lam_mode": lam,
            "strata": per_stratum,
        },
    )
