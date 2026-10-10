r"""Survey-only NPS estimators: ``classical`` and the ``naive_model`` baseline.

``classical``
    Respondent NPS with a delta-method Wald interval:

    .. math::

        \hat\theta = \bar Z, \qquad \widehat{\mathrm{NPS}} = g^\top \hat\theta, \qquad
        \widehat{\mathrm{se}} = \sqrt{g^\top \hat\Sigma_Z g / n},

    with :math:`Z = (1[y \ge 9], 1[y \le 6])`, :math:`g = (100, -100)` and
    :math:`\hat\Sigma_Z` the ddof-0 covariance of Z (as in
    ``ppi_py.classical_mean_ci``). Valid when the labeled set is a simple random
    sample (MCAR), by the central limit theorem. Applied to self-selected
    respondents (MAR/MNAR) it is also the "take respondents at face value" baseline,
    which is expected to fail there.

``naive_model``
    NPS of the average prediction over everyone (labeled and unlabeled), with
    :math:`\mathrm{se} = \mathrm{sd}(h) / \sqrt{n + N}`, :math:`h = g^\top f`. It
    treats the model as the truth, so it is never valid unless the model is unbiased.
    It is included to show that failure.

References
----------
Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023). Prediction-powered inference.
*Science* 382(6671), 669-674. (Classical baseline as implemented in ``ppi_py``.)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from silent97.estimators._common import (
    normal_interval,
    validate_alpha,
    validate_labeled,
    validate_unlabeled,
)
from silent97.estimators.nps import NPS_GRADIENT, delta_method_se, indicator_mean_cov
from silent97.types import Estimate

__all__ = ["classical", "naive_model"]


def classical(
    y_lab: ArrayLike,
    p_lab: ArrayLike | None = None,
    p_unlab: ArrayLike | None = None,
    *,
    alpha: float = 0.1,
) -> Estimate:
    """Classical survey NPS with a delta-method interval (see module docstring).

    Parameters
    ----------
    y_lab : array_like, shape (n,)
        Integer scores 0..10, n >= 2.
    p_lab, p_unlab : array_like, optional
        Accepted for interface compatibility and validated, but not used.
    alpha : float, default 0.1
        The interval targets coverage ``1 - alpha``.
    """
    a = validate_alpha(alpha)
    z, _ = validate_labeled(y_lab, p_lab)
    if p_unlab is not None:
        validate_unlabeled(p_unlab)
    n = z.shape[0]
    mean, cov = indicator_mean_cov(z, ddof=0)
    point = float(NPS_GRADIENT @ mean)
    se = delta_method_se(cov, n)
    lower, upper = normal_interval(point, se, a)
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=a,
        se=se,
        n_labeled=n,
        n_unlabeled=0,
        method="classical",
        diagnostics={
            "promoter_share": float(mean[0]),
            "detractor_share": float(mean[1]),
            "effective_n": float(n),
        },
    )


def naive_model(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
) -> Estimate:
    """NPS of the mean prediction over everyone (baseline; deliberately invalid).

    ``y_lab`` is validated but ignored: this is what reporting a model's average
    score as "the NPS" amounts to.
    """
    a = validate_alpha(alpha)
    _, p_l = validate_labeled(y_lab, p_lab)
    p_u = validate_unlabeled(p_unlab)
    if p_l is None:  # pragma: no cover - p_lab is a required argument here
        raise ValueError("p_lab is required")
    h = np.concatenate([p_l, p_u]) @ NPS_GRADIENT
    point = float(h.mean())
    se = float(h.std() / math.sqrt(h.size))
    lower, upper = normal_interval(point, se, a)
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=a,
        se=se,
        n_labeled=p_l.shape[0],
        n_unlabeled=p_u.shape[0],
        method="naive_model",
        diagnostics={},
    )
