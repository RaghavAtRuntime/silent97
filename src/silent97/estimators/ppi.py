r"""Prediction-powered NPS estimators: ``ppi``, ``ppi_pp``, ``ppi_pp_cf`` and ``ppi_crossfit``.

Notation
--------
Labeled set: n customers with indicators :math:`Z_i = (1[y_i \ge 9], 1[y_i \le 6])` and
predictions :math:`f_i = (\hat P(\text{promoter}), \hat P(\text{detractor}))`.
Unlabeled set: N customers with predictions :math:`\tilde f_j`.
NPS gradient :math:`g = (100, -100)`. :math:`\mathrm{cov}_0` is the ddof-0 sample
covariance.

PPI with power tuning (the core)
--------------------------------
For a scalar :math:`\lambda` shared by both coordinates,

.. math::

    \hat\theta_\lambda = \lambda \frac{1}{N}\sum_{j=1}^N \tilde f_j
        + \frac{1}{n}\sum_{i=1}^n (Z_i - \lambda f_i),

    \hat\Sigma_\lambda = \frac{\mathrm{cov}_0(\lambda \tilde f)}{N}
        + \frac{\mathrm{cov}_0(Z - \lambda f)}{n}, \qquad
    \widehat{\mathrm{NPS}} = g^\top\hat\theta_\lambda, \qquad
    \widehat{\mathrm{se}} = \sqrt{g^\top \hat\Sigma_\lambda g}.

The first term uses the *unlabeled* predictions. The two terms are independent
(disjoint samples), so their variances add. The interval is the Wald interval
:math:`\widehat{\mathrm{NPS}} \mp z_{1-\alpha/2}\,\widehat{\mathrm{se}}`. Validity:
the labeled set is an i.i.d./simple random sample from the same population as the
unlabeled set, and the model producing f is fixed independently of the labeled set.
f need not be accurate or calibrated; a poor model costs width, not validity.

``ppi`` uses :math:`\lambda = 1` (Angelopoulos et al. 2023).

Variance-optimal :math:`\lambda` for the NPS contrast (``ppi_pp``)
------------------------------------------------------------------
Because :math:`g^\top \hat\theta_\lambda = \lambda \bar h_{U} + \overline{(v - \lambda h)}_L`
with :math:`v_i = g^\top Z_i` and :math:`h_i = g^\top f_i`, PPI++ for the NPS contrast
is scalar PPI++ on :math:`(v, h)`. Its asymptotic variance,

.. math::

    \frac{\mathrm{Var}(v) - 2\lambda\,\mathrm{Cov}(v, h) + \lambda^2 \mathrm{Var}(h)}{n}
    + \frac{\lambda^2 \mathrm{Var}(h)}{N},

is a convex quadratic in :math:`\lambda`, minimized at

.. math::

    \lambda^\star = \frac{\mathrm{Cov}(v, h)}{(1 + n/N)\,\mathrm{Var}(h)}
    = \frac{g^\top C g}{2(1 + n/N)\, g^\top V g},

with :math:`C = \mathrm{Cov}(Z, f) + \mathrm{Cov}(f, Z)` (2x2) and
:math:`V = \mathrm{Var}(f)`. This optimizes the NPS variance directly, not each share
separately. The plug-in follows ``ppi_py``'s ``_calc_lam_glm``:
:math:`\hat C = (\tilde Z^\top \tilde F + \tilde F^\top \tilde Z)/n` from centered
labeled data, and :math:`\hat V` the ddof-1 covariance of the pooled labeled and
unlabeled predictions. The interval treats :math:`\hat\lambda` as fixed, which is
asymptotically valid because :math:`\hat\lambda \to \lambda^\star`
(Angelopoulos, Duchi & Zrnic 2023).

Clipping: the PPI++ paper notes the mean objective is convex for every
:math:`\lambda \in \mathbb{R}`, so no clipping is needed. ``ppi_py`` nevertheless
clips to [0, 1] for means. ``clip=False`` (the default) follows the paper;
``clip=True`` reproduces ``ppi_py``. If :math:`g^\top \hat V g = 0` (constant
predictions) every :math:`\lambda` gives the classical estimate, and
:math:`\hat\lambda = 0` is returned with ``lambda_degenerate=True``.

PPI++ with a cross-fitted :math:`\lambda` (``ppi_pp_cf``)
---------------------------------------------------------
Plug-in PPI++ estimates :math:`\lambda` on the same n responses that it then
rectifies, so :math:`\hat\lambda` is correlated with the rectifier and the estimate
has an :math:`O(1/n)` bias. It is negligible for large n, but visible for small,
skewed samples (E6 on Amazon reviews: +2.2 NPS points and 0.84 coverage at n = 50).
``ppi_pp_cf`` removes that dependence by cross-fitting the tuning parameter, as in
double/debiased machine learning (Chernozhukov et al. 2018). Split the labeled rows
into K folds :math:`I_1, \dots, I_K` (sizes :math:`n_k`, weights :math:`w_k = n_k/n`).
For each fold, :math:`\lambda_k` is the plug-in optimal :math:`\lambda` above,
computed from the labeled rows *outside* :math:`I_k` only (both :math:`\hat C` and
:math:`\hat V`; the unlabeled set enters only through N in the factor
:math:`1 + n/N`). Then

.. math::

    \hat\theta = \frac{1}{n}\sum_{i=1}^n \big(Z_i - \lambda_{k(i)} f_i\big)
        + \bar\lambda\,\frac{1}{N}\sum_{j=1}^N \tilde f_j,
    \qquad \bar\lambda = \sum_k w_k \lambda_k .

Unbiasedness (finite sample): :math:`\lambda_k` depends only on rows outside
:math:`I_k`, which are independent of :math:`I_k` and of the unlabeled set, so
:math:`E[\lambda_k (\bar{\tilde f} - \bar f_{I_k})] = E[\lambda_k] \cdot 0` and
:math:`E[\hat\theta] = E[Z]` whenever :math:`E|\lambda_k| < \infty`.

Variance, treating the :math:`\lambda_k` as fixed:

.. math::

    \hat\Sigma = \frac{1}{n^2} \sum_k n_k \big(\hat C_{ZZ}
        - \lambda_k (\hat C_{Zf} + \hat C_{fZ}) + \lambda_k^2 \hat C_{ff}\big)
        + \bar\lambda^2 \frac{\mathrm{cov}_0(\tilde f)}{N},

with :math:`\hat C` the ddof-0 covariances over all n labeled rows. The bracket is
:math:`\mathrm{cov}_0(Z - \lambda_k f)`, so with equal :math:`\lambda_k` this is
exactly the PPI variance at that :math:`\lambda`. The NPS se is
:math:`\sqrt{g^\top \hat\Sigma g}`, with a Wald interval as above. The variability
of the :math:`\lambda_k` themselves is ignored; it vanishes asymptotically
(:math:`\lambda_k \to \lambda^\star`), and small-n coverage is checked by simulation
(tests/coverage, E1, E6). Validity: the same assumptions as PPI, plus a random
fold split independent of the data. Folds come from ``rng`` (or are given).

Cross-prediction (``ppi_crossfit``)
-----------------------------------
No separate training data: the labeled set is split into K folds, model k is fit on
the other K - 1 folds, and it predicts fold k and every unlabeled row. Then

.. math::

    \hat\theta = \frac{1}{N}\sum_j \frac{1}{K}\sum_k f^{(k)}(\tilde X_j)
        + \frac{1}{n}\sum_i \big(Z_i - f^{(k_i)}(X_i)\big),

with the variance computed as in PPI with :math:`\lambda = 1`, using the
fold-averaged unlabeled predictions (Zrnic & Candès 2024, as implemented in
``ppi_py.crossppi_mean_ci``). Validity additionally needs the fitting procedure to
be stable, so that the K models are similar.

References
----------
Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023). Prediction-powered inference.
*Science* 382(6671), 669-674. arXiv:2301.09633.

Angelopoulos, Duchi, Zrnic (2023). PPI++: Efficient prediction-powered inference.
arXiv:2311.01453.

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018).
Double/debiased machine learning for treatment and structural parameters.
*The Econometrics Journal* 21(1), C1-C68.

Zrnic, Candès (2024). Cross-prediction-powered inference. *PNAS* 121(15),
e2322083121. arXiv:2309.16598.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.estimators._common import (
    normal_interval,
    validate_alpha,
    validate_labeled,
    validate_unlabeled,
)
from silent97.estimators.nps import (
    NPS_GRADIENT,
    indicator_mean_cov,
    nps_variance,
    validate_predictions,
    validate_scores,
)
from silent97.types import Estimate

__all__ = [
    "CrossFitPredictions",
    "FitFn",
    "PPIMeanResult",
    "cross_fit",
    "optimal_lambda",
    "ppi",
    "ppi_cf_from_lambdas",
    "ppi_crossfit",
    "ppi_mean_ci",
    "ppi_mean_cov",
    "ppi_mean_pointestimate",
    "ppi_pp",
    "ppi_pp_cf",
]

FloatArray = NDArray[np.float64]

FitFn = Callable[[FloatArray, NDArray[np.int64]], Callable[[FloatArray], FloatArray]]
"""``fit(x_train, scores_train)`` returns a predictor mapping (m, p) -> (m, 2)."""


# --------------------------------------------------------------------------- generic core


def _as_2d(a: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n,) or (n, d) with n >= 1, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


def _cov0(a: FloatArray) -> FloatArray:
    centered = a - a.mean(axis=0)
    out: FloatArray = centered.T @ centered / a.shape[0]
    return out


def _core_inputs(
    y: ArrayLike, f: ArrayLike, f_unlab: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y2, f2, u2 = _as_2d(y, "y"), _as_2d(f, "f"), _as_2d(f_unlab, "f_unlab")
    if y2.shape != f2.shape:
        raise ValueError(f"y and f must have the same shape, got {y2.shape} and {f2.shape}")
    if u2.shape[1] != y2.shape[1]:
        raise ValueError(f"f_unlab must have {y2.shape[1]} column(s), got {u2.shape[1]}")
    if y2.shape[0] < 2:
        raise ValueError(f"need at least 2 labeled rows, got {y2.shape[0]}")
    return y2, f2, u2


def ppi_mean_pointestimate(
    y: ArrayLike, f: ArrayLike, f_unlab: ArrayLike, lam: float
) -> FloatArray:
    r""":math:`\lambda \bar{\tilde f} + \overline{(y - \lambda f)}`, shape (d,)."""
    y2, f2, u2 = _core_inputs(y, f, f_unlab)
    out: FloatArray = lam * u2.mean(axis=0) + (y2 - lam * f2).mean(axis=0)
    return out


def ppi_mean_cov(y: ArrayLike, f: ArrayLike, f_unlab: ArrayLike, lam: float) -> FloatArray:
    r"""Estimated covariance of the PPI mean, :math:`\mathrm{cov}_0(\lambda \tilde f)/N
    + \mathrm{cov}_0(y - \lambda f)/n`, shape (d, d)."""
    y2, f2, u2 = _core_inputs(y, f, f_unlab)
    out: FloatArray = _cov0(lam * u2) / u2.shape[0] + _cov0(y2 - lam * f2) / y2.shape[0]
    return out


def optimal_lambda(
    y: ArrayLike,
    f: ArrayLike,
    f_unlab: ArrayLike,
    *,
    contrast: ArrayLike | None = None,
    clip: bool = False,
) -> tuple[float, bool, bool]:
    r"""Plug-in variance-optimal :math:`\lambda` (PPI++), see the module docstring.

    Parameters
    ----------
    y, f : array_like, shape (n,) or (n, d)
    f_unlab : array_like, shape (N,) or (N, d)
    contrast : array_like, shape (d,), optional
        Optimize the variance of ``contrast @ theta``. If omitted, optimize the trace
        of the covariance (``ppi_py``'s ``lam_optim_mode="overall"``).
    clip : bool, default False
        Clip to [0, 1] as ``ppi_py`` does for means.

    Returns
    -------
    (lam, degenerate, clipped)
        ``degenerate`` is True when the prediction variance along the contrast is
        zero, in which case ``lam = 0``. ``clipped`` is True when clipping moved it.
    """
    y2, f2, u2 = _core_inputs(y, f, f_unlab)
    n, big_n = y2.shape[0], u2.shape[0]
    y_c = y2 - y2.mean(axis=0)
    f_c = f2 - f2.mean(axis=0)
    c_sym = (y_c.T @ f_c + f_c.T @ y_c) / n
    pooled = np.concatenate([f2, u2], axis=0)
    v = np.atleast_2d(np.cov(pooled.T, ddof=1))
    scale = 2.0 * (1.0 + n / big_n)
    if contrast is None:
        num, den = float(np.trace(c_sym)), scale * float(np.trace(v))
        norm2 = float(y2.shape[1])
    else:
        c = np.asarray(contrast, dtype=np.float64)
        if c.shape != (y2.shape[1],):
            raise ValueError(f"contrast must have shape ({y2.shape[1]},), got {c.shape}")
        num, den = float(c @ c_sym @ c), scale * float(c @ v @ c)
        norm2 = float(c @ c)
    if not den > 1e-15 * norm2:
        return 0.0, True, False
    lam = num / den
    if clip:
        clipped_lam = min(max(lam, 0.0), 1.0)
        return clipped_lam, False, clipped_lam != lam
    return lam, False, False


class PPIMeanResult(NamedTuple):
    """Result of :func:`ppi_mean_ci` (arrays have shape (d,))."""

    point: FloatArray
    lower: FloatArray
    upper: FloatArray
    se: FloatArray
    lam: float
    degenerate: bool
    clipped: bool


def ppi_mean_ci(
    y: ArrayLike,
    f: ArrayLike,
    f_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
    lam: float | None = None,
    contrast: ArrayLike | None = None,
    clip: bool = False,
) -> PPIMeanResult:
    """Coordinate-wise PPI(++) Wald intervals for a d-dimensional mean.

    ``lam=None`` estimates lambda with :func:`optimal_lambda`. With ``clip=True`` and
    no contrast, this reproduces ``ppi_py.ppi_mean_ci(..., lam=None)``. With a given
    ``lam`` it reproduces ``ppi_py.ppi_mean_ci(..., lam=lam)``.
    """
    a = validate_alpha(alpha)
    degenerate = clipped = False
    if lam is None:
        lam, degenerate, clipped = optimal_lambda(y, f, f_unlab, contrast=contrast, clip=clip)
    point = ppi_mean_pointestimate(y, f, f_unlab, lam)
    se = np.sqrt(np.diag(ppi_mean_cov(y, f, f_unlab, lam)))
    lower, upper = normal_interval(point, se, a)  # type: ignore[arg-type]
    return PPIMeanResult(point, np.asarray(lower), np.asarray(upper), se, lam, degenerate, clipped)


# --------------------------------------------------------------------------- NPS estimators


def _nps_ppi(
    z: FloatArray,
    p_lab: FloatArray,
    p_unlab: FloatArray,
    lam: float,
    alpha: float,
    method: str,
    diagnostics: dict[str, Any],
) -> Estimate:
    n, big_n = z.shape[0], p_unlab.shape[0]
    theta = ppi_mean_pointestimate(z, p_lab, p_unlab, lam)
    var_imputed = nps_variance(_cov0(lam * p_unlab)) / big_n
    var_rectifier = nps_variance(_cov0(z - lam * p_lab)) / n
    se = math.sqrt(var_imputed + var_rectifier)
    point = float(NPS_GRADIENT @ theta)
    lower, upper = normal_interval(point, se, alpha)
    _, cov_z = indicator_mean_cov(z, ddof=0)
    var_classical = nps_variance(cov_z) / n
    if se > 0:
        effective_n = n * var_classical / se**2
    else:
        effective_n = float(n) if var_classical == 0 else math.inf
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=alpha,
        se=se,
        n_labeled=n,
        n_unlabeled=big_n,
        method=method,
        diagnostics={
            "lambda_hat": float(lam),
            "promoter_share": float(theta[0]),
            "detractor_share": float(theta[1]),
            "se_imputed": math.sqrt(var_imputed),
            "se_rectifier": math.sqrt(var_rectifier),
            "effective_n": float(effective_n),
            **diagnostics,
        },
    )


def _validated(
    y_lab: ArrayLike, p_lab: ArrayLike, p_unlab: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    z, p_l = validate_labeled(y_lab, p_lab)
    if p_l is None:  # pragma: no cover - p_lab is a required argument here
        raise ValueError("p_lab is required")
    return z, p_l, validate_unlabeled(p_unlab)


def ppi(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
) -> Estimate:
    """Prediction-powered NPS with lambda = 1 (see module docstring).

    Parameters
    ----------
    y_lab : array_like, shape (n,)
        Scores 0..10 of surveyed customers, n >= 2.
    p_lab : array_like, shape (n, 2)
        [P(promoter), P(detractor)] for the surveyed customers, from a model not
        trained on them.
    p_unlab : array_like, shape (N, 2)
        Predictions for the unsurveyed customers.
    alpha : float, default 0.1
    """
    a = validate_alpha(alpha)
    z, p_l, p_u = _validated(y_lab, p_lab, p_unlab)
    return _nps_ppi(z, p_l, p_u, 1.0, a, "ppi", {})


def ppi_pp(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
    clip: bool = False,
) -> Estimate:
    """PPI++ NPS: lambda chosen to minimize the variance of the NPS contrast.

    Same inputs as :func:`ppi`. ``clip=True`` clips lambda to [0, 1] as ``ppi_py``
    does; the default follows the PPI++ paper (no clipping for means).
    """
    a = validate_alpha(alpha)
    z, p_l, p_u = _validated(y_lab, p_lab, p_unlab)
    lam, degenerate, clipped = optimal_lambda(z, p_l, p_u, contrast=NPS_GRADIENT, clip=clip)
    return _nps_ppi(
        z,
        p_l,
        p_u,
        lam,
        a,
        "ppi_pp",
        {"lambda_degenerate": degenerate, "lambda_clipped": clipped, "clip": clip},
    )


def _fold_lambda(z_out: FloatArray, f_out: FloatArray, n: int, big_n: int) -> tuple[float, bool]:
    """Plug-in NPS-contrast lambda from out-of-fold labeled rows only (module docstring)."""
    m = z_out.shape[0]
    z_c = z_out - z_out.mean(axis=0)
    f_c = f_out - f_out.mean(axis=0)
    c_sym = (z_c.T @ f_c + f_c.T @ z_c) / m
    v = np.atleast_2d(np.cov(f_out.T, ddof=1))
    g = NPS_GRADIENT
    num = float(g @ c_sym @ g)
    den = 2.0 * (1.0 + n / big_n) * float(g @ v @ g)
    if not den > 1e-15 * float(g @ g):
        return 0.0, True
    return num / den, False


def ppi_cf_from_lambdas(
    z: FloatArray,
    p_lab: FloatArray,
    p_unlab: FloatArray,
    folds: NDArray[np.int64],
    lambdas: FloatArray,
    *,
    alpha: float,
    method: str = "ppi_pp_cf",
    diagnostics: dict[str, Any] | None = None,
) -> Estimate:
    """The ``ppi_pp_cf`` point estimate and interval for given fold lambdas.

    ``z`` (n, 2) indicators, ``folds`` (n,) labels in 0..K-1, ``lambdas`` (K,). See the
    module docstring for the formulas.
    """
    n, big_n = z.shape[0], p_unlab.shape[0]
    k = lambdas.shape[0]
    counts = np.bincount(folds, minlength=k).astype(np.float64)
    lam_i = lambdas[folds]
    lam_bar = float(counts @ lambdas / n)
    theta: FloatArray = (z - lam_i[:, None] * p_lab).mean(axis=0) + lam_bar * p_unlab.mean(axis=0)
    joint = _cov0(np.concatenate([z, p_lab], axis=1))
    c_zz, c_zf, c_ff = joint[:2, :2], joint[:2, 2:], joint[2:, 2:]
    var_rectifier = 0.0
    for lam, n_k in zip(lambdas, counts, strict=True):
        cov_k = c_zz - lam * (c_zf + c_zf.T) + lam**2 * c_ff
        var_rectifier += n_k * nps_variance(cov_k)
    var_rectifier /= n**2
    var_imputed = lam_bar**2 * nps_variance(_cov0(p_unlab)) / big_n
    se = math.sqrt(max(var_imputed + var_rectifier, 0.0))
    point = float(NPS_GRADIENT @ theta)
    lower, upper = normal_interval(point, se, alpha)
    _, cov_z = indicator_mean_cov(z, ddof=0)
    var_classical = nps_variance(cov_z) / n
    if se > 0:
        effective_n = n * var_classical / se**2
    else:
        effective_n = float(n) if var_classical == 0 else math.inf
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=alpha,
        se=se,
        n_labeled=n,
        n_unlabeled=big_n,
        method=method,
        diagnostics={
            "lambda_hat": lam_bar,
            "lambda_folds": [float(x) for x in lambdas],
            "n_folds": int(k),
            "promoter_share": float(theta[0]),
            "detractor_share": float(theta[1]),
            "se_imputed": math.sqrt(var_imputed),
            "se_rectifier": math.sqrt(max(var_rectifier, 0.0)),
            "effective_n": float(effective_n),
            **(diagnostics or {}),
        },
    )


def ppi_pp_cf(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
    n_folds: int = 10,
    folds: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> Estimate:
    """PPI++ NPS with a cross-fitted lambda (see module docstring).

    Same inputs as :func:`ppi`, plus the fold split: either ``folds`` (an (n,) array of
    integer fold labels) or ``rng`` with ``n_folds`` (a random split into near-equal
    folds). Every fold must leave at least 2 labeled rows outside it.

    Prefer this over :func:`ppi_pp` for small surveys (roughly n < 200): it removes the
    same-sample lambda bias. For large n the two agree.
    """
    a = validate_alpha(alpha)
    z, p_l, p_u = _validated(y_lab, p_lab, p_unlab)
    n = z.shape[0]
    if folds is None:
        if rng is None:
            raise ValueError("pass folds or rng (a numpy Generator) to split the labeled rows")
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 2 <= n_folds <= n:
            raise ValueError(f"n_folds must be an integer in [2, n = {n}], got {n_folds!r}")
        fold_ids = np.empty(n, dtype=np.int64)
        for k, chunk in enumerate(np.array_split(rng.permutation(n), n_folds)):
            fold_ids[chunk] = k
    else:
        raw = np.asarray(folds)
        if raw.shape != (n,) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"folds must be an integer array of shape ({n},), got {raw.shape}")
        _, inverse = np.unique(raw, return_inverse=True)
        fold_ids = inverse.astype(np.int64)
    k_total = int(fold_ids.max()) + 1
    if k_total < 2:
        raise ValueError("folds must contain at least 2 distinct folds")
    counts = np.bincount(fold_ids, minlength=k_total)
    if (n - counts).min() < 2:
        raise ValueError("every fold must leave at least 2 labeled rows outside it")
    big_n = p_u.shape[0]
    lambdas = np.empty(k_total)
    degenerate = 0
    for k in range(k_total):
        out = fold_ids != k
        lambdas[k], deg = _fold_lambda(z[out], p_l[out], n, big_n)
        degenerate += deg
    return ppi_cf_from_lambdas(
        z, p_l, p_u, fold_ids, lambdas, alpha=a, diagnostics={"lambda_degenerate_folds": degenerate}
    )


def ppi_crossfit(
    y_lab: ArrayLike,
    p_lab: ArrayLike,
    p_unlab: ArrayLike,
    *,
    alpha: float = 0.1,
) -> Estimate:
    """Cross-prediction-powered NPS (see module docstring).

    Parameters
    ----------
    y_lab : array_like, shape (n,)
    p_lab : array_like, shape (n, 2)
        Cross-fit predictions: row i comes from the model that did not see row i.
    p_unlab : array_like, shape (N, K, 2)
        Predictions of each of the K fold models for every unlabeled customer.
        Produce both with :func:`cross_fit`.
    alpha : float, default 0.1
    """
    a = validate_alpha(alpha)
    u = np.asarray(p_unlab, dtype=np.float64)
    if u.ndim != 3 or u.shape[2] != 2 or u.shape[1] < 2 or u.shape[0] < 1:
        raise ValueError(f"p_unlab must have shape (N, K, 2) with K >= 2, got {u.shape}")
    validate_predictions(u.reshape(-1, 2), name="p_unlab")
    z, p_l, p_u = _validated(y_lab, p_lab, u.mean(axis=1))
    return _nps_ppi(z, p_l, p_u, 1.0, a, "ppi_crossfit", {"n_folds": int(u.shape[1])})


class CrossFitPredictions(NamedTuple):
    """Output of :func:`cross_fit`."""

    p_lab: FloatArray
    """(n, 2) predictions for each labeled row from the model that did not see it."""
    p_unlab: FloatArray | None
    """(N, K, 2) predictions of each fold model on the unlabeled rows (None if no
    ``x_unlab`` was given)."""
    folds: NDArray[np.int64]
    """(n,) fold id of each labeled row."""
    predictors: tuple[Callable[[FloatArray], FloatArray], ...]
    """The K fold models; ``predictors[k]`` never saw fold k."""


def cross_fit(
    x_lab: ArrayLike,
    y_lab: ArrayLike,
    x_unlab: ArrayLike | None,
    *,
    fit: FitFn,
    n_folds: int = 10,
    rng: np.random.Generator,
) -> CrossFitPredictions:
    """K-fold cross-fitting of a model on the labeled set.

    Rows are assigned to ``n_folds`` folds of near-equal size by a random permutation.
    For each fold k, ``fit`` is called on the labeled rows outside fold k, and the
    returned predictor is applied to fold k and (if given) to all of ``x_unlab``.
    Pass ``x_unlab=None`` to predict the unlabeled rows yourself from
    ``predictors`` (e.g. in one batched pass).
    """
    x_l = np.asarray(x_lab, dtype=np.float64)
    scores = validate_scores(y_lab, min_n=2, name="y_lab")
    n = scores.shape[0]
    if x_l.ndim != 2 or x_l.shape[0] != n:
        raise ValueError(f"x_lab must have shape (n, p) with n = {n}, got {x_l.shape}")
    x_u = None if x_unlab is None else np.asarray(x_unlab, dtype=np.float64)
    if x_u is not None and (x_u.ndim != 2 or x_u.shape[1] != x_l.shape[1]):
        raise ValueError(f"x_unlab must have shape (N, {x_l.shape[1]}), got {x_u.shape}")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be an integer in [2, n = {n}], got {n_folds!r}")

    folds = np.empty(n, dtype=np.int64)
    for k, chunk in enumerate(np.array_split(rng.permutation(n), n_folds)):
        folds[chunk] = k
    p_lab = np.empty((n, 2))
    p_unlab = None if x_u is None else np.empty((x_u.shape[0], n_folds, 2))
    predictors = []
    for k in range(n_folds):
        held_out = folds == k
        predict = fit(x_l[~held_out], scores[~held_out])
        predictors.append(predict)
        p_lab[held_out] = predict(x_l[held_out])
        if p_unlab is not None and x_u is not None:
            p_unlab[:, k, :] = predict(x_u)
    # p_unlab is validated by ppi_crossfit; validating 2 x N x K values twice is costly.
    validate_predictions(p_lab, name="cross-fit p_lab")
    return CrossFitPredictions(p_lab, p_unlab, folds, tuple(predictors))
