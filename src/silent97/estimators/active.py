r"""Active (inverse-probability-weighted) PPI for NPS under Poisson sampling: ``ppi_active``.

Setting (Zrnic & Candès 2024, "Active statistical inference"): every customer i in a
pool of M is surveyed independently with a known probability :math:`\pi_i > 0`
(:func:`silent97.sampling.designs.active`), :math:`\xi_i = 1` if surveyed. With
indicators :math:`Z_i = (1[y_i \ge 9], 1[y_i \le 6])` (observed only if :math:`\xi_i = 1`)
and predictions :math:`f_i`, define for *every* pool customer

.. math::

    a_i = \lambda f_i + \frac{\xi_i}{\pi_i} (Z_i - \lambda f_i), \qquad
    \hat\theta = \frac{1}{M} \sum_{i=1}^{M} a_i, \qquad
    \hat\Sigma = \frac{\mathrm{cov}_0(a)}{M}.

Unsurveyed customers contribute :math:`a_i = \lambda f_i`, so the estimator needs
:math:`\pi` only for the surveyed. :math:`E[\xi_i/\pi_i] = 1` makes
:math:`E[\hat\theta \mid \text{pool}] = \bar Z_{\text{pool}}` exactly, for any fixed
:math:`\lambda` and any :math:`\pi` that does not depend on the outcomes (here it depends
only on the predictions). The :math:`a_i` are independent, and :math:`\hat\Sigma` is the
i.i.d.-stream variance of Zrnic & Candès. For the finite pool's own NPS it is slightly
conservative, by a term of order :math:`\mathrm{Var}(Z)/M`. NPS
:math:`= g^\top \hat\theta` with :math:`g = (100, -100)`, and the Wald interval uses
:math:`\sqrt{g^\top \hat\Sigma g}`, the joint 2-D covariance as everywhere in this
package.

``lam = 1`` (the default) is plain active PPI. A fixed other value is allowed. No
tuned :math:`\lambda` is offered here: a same-sample plug-in has the small-n bias
documented for ``ppi_pp`` (Phase 3).

Validity: :math:`\pi_i > 0` for everyone, known, and independent of the outcomes
given the predictions; Poisson (independent) sampling; the model is fixed independently
of the surveyed set. Forgetting the :math:`1/\pi` weights biases the estimate whenever
the rectifier :math:`Z - f` is correlated with :math:`\pi` (tested).

Reference: Zrnic, Candès (2024). Active statistical inference. *ICML*, PMLR 235.
arXiv:2403.03208.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.estimators._common import normal_interval, validate_alpha
from silent97.estimators.nps import NPS_GRADIENT, indicator_mean_cov, nps_variance
from silent97.estimators.ppi import _validated
from silent97.types import Estimate

__all__ = ["ppi_active"]

FloatArray = NDArray[np.float64]


def ppi_active(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    pi_lab: ArrayLike,
    alpha: float = 0.1,
    lam: float = 1.0,
) -> Estimate:
    """Active PPI NPS (see module docstring).

    Parameters
    ----------
    y_lab : array_like, shape (n,)
        Scores 0..10 of the surveyed customers, n >= 2.
    p_lab : array_like, shape (n, 2)
        Predictions [P(promoter), P(detractor)] for the surveyed customers.
    p_unlab : array_like, shape (N, 2)
        Predictions for the pool customers who were not surveyed.
    pi_lab : array_like, shape (n,)
        Inclusion probabilities of the surveyed customers, in (0, 1].
    alpha : float, default 0.1
    lam : float, default 1.0
    """
    a = validate_alpha(alpha)
    z, p_l, p_u = _validated(y_lab, p_lab, p_unlab)
    pi = np.asarray(pi_lab, dtype=np.float64)
    if pi.shape != (z.shape[0],) or not np.isfinite(pi).all() or (pi <= 0).any() or (pi > 1).any():
        raise ValueError(f"pi_lab must be {z.shape[0]} probabilities in (0, 1]")
    if not math.isfinite(lam):
        raise ValueError(f"lam must be finite, got {lam}")
    n, big_n = z.shape[0], p_u.shape[0]
    m = n + big_n
    a_lab = lam * p_l + (z - lam * p_l) / pi[:, None]
    a_all = np.concatenate([a_lab, lam * p_u], axis=0)
    theta: FloatArray = a_all.mean(axis=0)
    centered = a_all - theta
    cov: FloatArray = centered.T @ centered / m / m
    se = math.sqrt(max(nps_variance(cov), 0.0))
    point = float(NPS_GRADIENT @ theta)
    lower, upper = normal_interval(point, se, a)
    _, cov_z = indicator_mean_cov(z, ddof=0)
    var_classical = nps_variance(cov_z) / n
    effective_n = n * var_classical / se**2 if se > 0 else math.inf
    weights = 1.0 / pi
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=a,
        se=se,
        n_labeled=n,
        n_unlabeled=big_n,
        method="ppi_active",
        diagnostics={
            "lambda_hat": float(lam),
            "promoter_share": float(theta[0]),
            "detractor_share": float(theta[1]),
            "effective_n": float(effective_n),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_ess": float(weights.sum() ** 2 / (weights**2).sum()),
        },
    )
