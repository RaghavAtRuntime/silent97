r"""Multilevel regression and poststratification (MRP) for NPS: ``mrp`` and ``mrp_ppi``.

Cells
-----
The population is cross-classified into C cells (here province x age band x tenure
band, reviewer decision Q2 in ``docs/progress/phase-4.md``). Each cell c has a known
population count :math:`N_c`, the levels of F grouping factors (province, age band,
tenure band and the province x age segment), and, from the survey, :math:`n_c`
respondents, of whom :math:`k_c` are promoters (or detractors).

Model (fit separately for promoters and for detractors; CLAUDE.md rule 7)
--------------------------------------------------------------------------

.. math::

    k_c \sim \mathrm{Binomial}(n_c, p_c), \qquad
    \operatorname{logit} p_c = \beta_0 + \beta_1 x_c + \sum_{f=1}^{F} u_{f, \ell_f(c)},
    \qquad u_{f, \ell} \sim N(0, \sigma_f^2),

with priors :math:`\beta_j \sim N(0, 2.5^2)` and :math:`\sigma_f \sim
\mathrm{HalfNormal}(1)`. The random intercepts pool information across cells that
share a province, age band, tenure band or segment (partial pooling), so thin cells
borrow strength. ``mrp`` has no :math:`x_c`. ``mrp_ppi`` adds a cell-level covariate
from the model's predictions: the cell mean of the predicted promoter (detractor)
logit over the population, standardized.

Poststratification
------------------
For each posterior draw, :math:`\mathrm{NPS} = 100 \sum_c W_c (p^{P}_c - p^{D}_c)` with
:math:`W_c = N_c / \sum N`; a segment's NPS uses the same sum over its own cells. The point
estimate is the posterior mean, ``se`` the posterior sd, and the interval the equal-tailed
:math:`(1 - \alpha)` posterior interval. The promoter and detractor models are fit
independently, so their posterior draws are combined as independent; the small
negative correlation between the two shares is ignored.

Fitting
-------
- ``fit="nuts"``: PyMC NUTS with a non-centered parameterization (4 chains). It raises
  :class:`ConvergenceError` unless every parameter has R-hat <= 1.01 and bulk ESS >= 400
  (rule 7).
- ``fit="laplace"`` (the default, for the 1,000-replicate sweeps): a glmer-style
  Laplace approximation. The log random-effect sds :math:`t_f = \log\sigma_f` maximize
  the Laplace-approximated marginal posterior

  .. math::

      \ell(t) = L(\hat\theta_t; t) - \tfrac12 \log\lvert H_t\rvert
          - \sum_f q_f t_f + \sum_f \big(t_f - \tfrac12 e^{2 t_f}\big),

  where :math:`\theta = (\beta, u)`, :math:`L` is the log joint density (binomial
  log-likelihood plus the Gaussian log priors on :math:`\beta` and :math:`u`),
  :math:`\hat\theta_t` its mode (found by Newton's method),
  :math:`H_t = -\nabla^2 L` there, :math:`q_f` the number of levels of factor f, and
  the last sum is the HalfNormal(1) log prior with the Jacobian of :math:`t`. Then
  :math:`\theta \approx N(\hat\theta, H^{-1})` at :math:`\hat t`. This ignores the
  uncertainty in :math:`\hat t`, as glmer-based MRP does; it is validated against NUTS
  on a subset (rule 7), and the agreement is reported in the phase 4 note.
  (``statsmodels``' ``BinomialBayesMixedGLM.fit_map`` maximizes the *joint* posterior
  over the variance parameters, which collapsed to zero here; see the phase 4 note.)

Validity: missing at random given the cell (respondents are exchangeable with
non-respondents within each cell), known population cell counts, and a model that fits.
Unlike PPI, MRP's intervals are model-based: shrinkage trades some bias in thin cells
for much lower variance, and coverage is checked by simulation (E4, E5), not
guaranteed.

References: Gelman & Little (1997), Poststratification into many categories using
hierarchical logistic regression, *Survey Methodology* 23(2); Wang, Rothschild, Goel &
Gelman (2015), Forecasting elections with non-representative polls, *International
Journal of Forecasting* 31(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from silent97.estimators.nps import validate_scores
from silent97.types import Estimate

__all__ = [
    "BETA_PRIOR_SD",
    "FITS",
    "CellFrame",
    "ConvergenceError",
    "check_convergence",
    "fit_cell_probabilities",
    "mrp",
    "mrp_ppi",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BETA_PRIOR_SD: Final = 2.5
SIGMA_PRIOR_SD: Final = 1.0
FITS: Final = ("laplace", "nuts")
MAX_RHAT: Final = 1.01
MIN_ESS_BULK: Final = 400.0


class ConvergenceError(RuntimeError):
    """NUTS diagnostics failed the R-hat / ESS thresholds."""


@dataclass(frozen=True)
class CellFrame:
    """Population cells for MRP.

    Attributes
    ----------
    factors : (C, F) int array
        Level index of each grouping factor for each cell (0-based).
    factor_names : tuple of F str
    pop_count : (C,) array
        Known population count per cell; all > 0.
    segment : (C,) int array, optional
        Segment label per cell, for segment-level estimates.
    """

    factors: IntArray
    factor_names: tuple[str, ...]
    pop_count: FloatArray
    segment: IntArray | None = None
    factor_sizes: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        f = np.asarray(self.factors)
        if f.ndim != 2 or not np.issubdtype(f.dtype, np.integer) or (f < 0).any():
            raise ValueError("factors must be a (C, F) array of non-negative integers")
        if len(self.factor_names) != f.shape[1]:
            raise ValueError(f"factor_names must name all {f.shape[1]} factors")
        pop = np.asarray(self.pop_count, dtype=np.float64)
        if pop.shape != (f.shape[0],) or not (pop > 0).all():
            raise ValueError("pop_count must be one positive count per cell")
        object.__setattr__(self, "factors", f.astype(np.int64))
        object.__setattr__(self, "pop_count", pop)
        object.__setattr__(self, "factor_sizes", tuple(int(c.max()) + 1 for c in f.T))
        if self.segment is not None:
            s = np.asarray(self.segment)
            if s.shape != (f.shape[0],):
                raise ValueError("segment must give one label per cell")
            object.__setattr__(self, "segment", s.astype(np.int64))

    @property
    def n_cells(self) -> int:
        return int(self.factors.shape[0])


@dataclass(frozen=True)
class _Design:
    a: FloatArray  # (C, p + q): fixed-effect columns, then one-hot random-effect columns
    p: int
    sizes: tuple[int, ...]

    @property
    def dim(self) -> int:
        return int(self.a.shape[1])


def _design(cells: CellFrame, x: FloatArray | None) -> _Design:
    cols = [np.ones((cells.n_cells, 1))]
    if x is not None:
        cols.append(x[:, None])
    for f, size in enumerate(cells.factor_sizes):
        cols.append(np.eye(size)[cells.factors[:, f]])
    return _Design(np.hstack(cols), 1 + (x is not None), cells.factor_sizes)


def _precision(design: _Design, log_sd: FloatArray) -> FloatArray:
    prec = [np.full(design.p, 1.0 / BETA_PRIOR_SD**2)]
    for t, size in zip(log_sd, design.sizes, strict=True):
        prec.append(np.full(size, math.exp(-2.0 * t)))
    return np.concatenate(prec)


def _log_joint(
    design: _Design, k: FloatArray, n: FloatArray, log_sd: FloatArray, theta: FloatArray
) -> float:
    eta = design.a @ theta
    loglik = float(k @ eta - n @ np.logaddexp(0.0, eta))
    return loglik - 0.5 * float(_precision(design, log_sd) @ theta**2)


def _conditional_mode(
    design: _Design,
    k: FloatArray,
    n: FloatArray,
    log_sd: FloatArray,
    start: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Newton's method for the mode of the log joint at fixed sds; returns (mode, H, L)."""
    prec = _precision(design, log_sd)
    theta = np.zeros(design.dim) if start is None else start.copy()
    value = _log_joint(design, k, n, log_sd, theta)
    for _ in range(100):
        p = expit(design.a @ theta)
        grad = design.a.T @ (k - n * p) - prec * theta
        hess = (design.a.T * (n * p * (1 - p))) @ design.a + np.diag(prec)
        step = np.linalg.solve(hess, grad)
        if float(grad @ step) < 1e-12:  # Newton decrement: at the mode to machine precision
            break
        t = 1.0
        while True:
            cand = theta + t * step
            cand_value = _log_joint(design, k, n, log_sd, cand)
            if cand_value >= value - 1e-12 or t < 1e-8:
                break
            t /= 2
        theta, value = cand, cand_value
    p = expit(design.a @ theta)
    hess = (design.a.T * (n * p * (1 - p))) @ design.a + np.diag(prec)
    return theta, hess, value


def _laplace(
    design: _Design, k: FloatArray, n: FloatArray, rng: np.random.Generator, n_draws: int
) -> tuple[FloatArray, dict[str, Any]]:
    cache: dict[str, FloatArray] = {}
    sizes = np.asarray(design.sizes, dtype=np.float64)

    def neg_marginal(t: FloatArray) -> float:
        mode, hess, value = _conditional_mode(design, k, n, t, cache.get("mode"))
        cache["mode"] = mode
        _, logdet = np.linalg.slogdet(hess)
        log_prior = float(np.sum(t - 0.5 * np.exp(2.0 * t) / SIGMA_PRIOR_SD**2))
        return -(value - 0.5 * float(logdet) - float(sizes @ t) + log_prior)

    t0 = np.full(len(design.sizes), math.log(0.3))
    bounds = [(math.log(1e-3), math.log(10.0))] * len(design.sizes)
    res = minimize(neg_marginal, t0, method="L-BFGS-B", bounds=bounds)
    mode, hess, _ = _conditional_mode(design, k, n, res.x, cache.get("mode"))
    chol = np.linalg.cholesky(hess)
    z = rng.standard_normal((n_draws, design.dim))
    theta = mode + np.linalg.solve(chol.T, z.T).T
    info = {"sd": np.exp(res.x).tolist(), "converged": bool(res.success), "beta": mode[: design.p]}
    return expit(theta @ design.a.T), info


def check_convergence(rhat: ArrayLike, ess_bulk: ArrayLike) -> None:
    """Raise :class:`ConvergenceError` unless R-hat <= 1.01 and bulk ESS >= 400 everywhere."""
    r, e = np.asarray(rhat, dtype=np.float64), np.asarray(ess_bulk, dtype=np.float64)
    if not np.isfinite(r).all() or r.max() > MAX_RHAT:
        raise ConvergenceError(f"NUTS R-hat {np.nanmax(r):.4f} exceeds {MAX_RHAT}")
    if not np.isfinite(e).all() or e.min() < MIN_ESS_BULK:
        raise ConvergenceError(f"NUTS bulk ESS {np.nanmin(e):.0f} below {MIN_ESS_BULK:.0f}")


def _nuts(
    design: _Design,
    k: FloatArray,
    n: FloatArray,
    rng: np.random.Generator,
    *,
    draws: int,
    tune: int,
    chains: int,
) -> tuple[FloatArray, dict[str, Any]]:  # pragma: no cover - exercised by the slow test
    # arviz (via PyMC 5) registers its styles in matplotlib.style.core, which matplotlib
    # 3.11 no longer imports with matplotlib.style; load the submodule first.
    import matplotlib.style.core  # noqa: F401
    import pymc as pm
    from arviz import summary as az_summary

    fixed = design.a[:, : design.p]
    offsets = np.cumsum([design.p, *design.sizes])
    with pm.Model():
        beta = pm.Normal("beta", 0.0, BETA_PRIOR_SD, shape=design.p)
        sigma = pm.HalfNormal("sigma", SIGMA_PRIOR_SD, shape=len(design.sizes))
        eta = pm.math.dot(fixed, beta)
        for f, size in enumerate(design.sizes):
            z = pm.Normal(f"z{f}", 0.0, 1.0, shape=size)
            onehot = design.a[:, offsets[f] : offsets[f] + size]
            eta = eta + sigma[f] * pm.math.dot(onehot, z)
        pm.Binomial("k", n=n, p=pm.math.invlogit(eta), observed=k)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, progressbar=False,
            random_seed=int(rng.integers(2**31 - 1)), compute_convergence_checks=False,
            target_accept=0.95,
        )  # fmt: skip
    # Unrounded diagnostics (az.summary rounds to 2 decimals by default, which would let
    # an R-hat of 1.014 pass as 1.01).
    summary = az_summary(idata, kind="diagnostics", round_to="none")
    check_convergence(summary["r_hat"].to_numpy(), summary["ess_bulk"].to_numpy())
    post = idata.posterior.stack(sample=("chain", "draw"))
    eta_draws = fixed @ post["beta"].to_numpy()
    sig = post["sigma"].to_numpy()
    for f, size in enumerate(design.sizes):
        onehot = design.a[:, offsets[f] : offsets[f] + size]
        eta_draws = eta_draws + onehot @ (post[f"z{f}"].to_numpy() * sig[f])
    info = {
        "sd": sig.mean(axis=1).tolist(),
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "beta": post["beta"].to_numpy().mean(axis=1),
    }
    return expit(eta_draws.T), info


def _fit(
    cells: CellFrame,
    k: ArrayLike,
    n: ArrayLike,
    *,
    x: FloatArray | None,
    rng: np.random.Generator,
    fit: str,
    n_draws: int,
    nuts_draws: int,
    nuts_tune: int,
    nuts_chains: int,
) -> tuple[FloatArray, dict[str, Any]]:
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    design = _design(cells, x)
    kk, nn = np.asarray(k, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if fit == "laplace":
        return _laplace(design, kk, nn, rng, n_draws)
    return _nuts(design, kk, nn, rng, draws=nuts_draws, tune=nuts_tune, chains=nuts_chains)


def fit_cell_probabilities(
    cells: CellFrame,
    k: ArrayLike,
    n: ArrayLike,
    *,
    rng: np.random.Generator,
    x: ArrayLike | None = None,
    fit: str = "laplace",
    n_draws: int = 4000,
) -> FloatArray:
    """Posterior draws, shape (n_draws, C), of each cell's success probability."""
    xx = None if x is None else np.asarray(x, dtype=np.float64)
    draws, _ = _fit(
        cells, k, n, x=xx, rng=rng, fit=fit, n_draws=n_draws,
        nuts_draws=2000, nuts_tune=1000, nuts_chains=4,
    )  # fmt: skip
    return draws


def _standardize(x: FloatArray, w: FloatArray) -> FloatArray:
    m = float(w @ x)
    s = math.sqrt(float(w @ (x - m) ** 2))
    return (x - m) / s if s > 0 else x - m


def _summarize(draws: FloatArray, alpha: float) -> tuple[float, float, float, float]:
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(draws.mean()), float(lo), float(hi), float(draws.std(ddof=1))


def _mrp(
    y_lab: ArrayLike,
    cell_lab: ArrayLike,
    cells: CellFrame,
    *,
    x_cell: ArrayLike | None,
    rng: np.random.Generator,
    alpha: float,
    fit: str,
    n_draws: int,
    nuts_draws: int,
    nuts_tune: int,
    nuts_chains: int,
    method: str,
) -> Estimate:
    y = validate_scores(y_lab)
    cell = np.asarray(cell_lab)
    if y.size < 2:
        raise ValueError(f"need at least 2 respondents, got {y.size}")
    if (
        cell.shape != y.shape
        or not np.issubdtype(cell.dtype, np.integer)
        or (cell < 0).any()
        or (cell >= cells.n_cells).any()
    ):
        raise ValueError(f"cell_lab must give each respondent a cell index in [0, {cells.n_cells})")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    w = cells.pop_count / cells.pop_count.sum()
    xs: list[FloatArray | None] = [None, None]
    if x_cell is not None:
        xc = np.asarray(x_cell, dtype=np.float64)
        if xc.shape != (cells.n_cells, 2) or not np.isfinite(xc).all():
            raise ValueError(f"x_cell must be a finite ({cells.n_cells}, 2) array")
        xs = [_standardize(xc[:, 0], w), _standardize(xc[:, 1], w)]
    n_c = np.bincount(cell, minlength=cells.n_cells)
    k_prom = np.bincount(cell[y >= 9], minlength=cells.n_cells)
    k_det = np.bincount(cell[y <= 6], minlength=cells.n_cells)
    kw = {"rng": rng, "fit": fit, "n_draws": n_draws, "nuts_draws": nuts_draws,
          "nuts_tune": nuts_tune, "nuts_chains": nuts_chains}  # fmt: skip
    p_prom, info_p = _fit(cells, k_prom, n_c, x=xs[0], **kw)  # type: ignore[arg-type]
    p_det, info_d = _fit(cells, k_det, n_c, x=xs[1], **kw)  # type: ignore[arg-type]
    m = min(p_prom.shape[0], p_det.shape[0])
    nps_cells = 100.0 * (p_prom[:m] - p_det[:m])
    point, lower, upper, se = _summarize(nps_cells @ w, alpha)
    diagnostics: dict[str, Any] = {
        "fit": fit,
        "sd_promoter": info_p["sd"],
        "sd_detractor": info_d["sd"],
        "n_cells": cells.n_cells,
        "n_cells_observed": int((n_c > 0).sum()),
    }
    if x_cell is not None:
        diagnostics["slope_promoter"] = float(info_p["beta"][1])
        diagnostics["slope_detractor"] = float(info_d["beta"][1])
    if fit == "nuts":
        diagnostics["max_rhat"] = max(info_p["max_rhat"], info_d["max_rhat"])
        diagnostics["min_ess_bulk"] = min(info_p["min_ess_bulk"], info_d["min_ess_bulk"])
        diagnostics["divergences"] = info_p["divergences"] + info_d["divergences"]
    else:
        diagnostics["laplace_converged"] = bool(info_p["converged"] and info_d["converged"])
    if cells.segment is not None:
        labels = np.unique(cells.segment)
        seg: dict[str, list[Any]] = {k: [] for k in ("label", "point", "lower", "upper", "n_resp")}
        for s in labels:
            m_s = cells.segment == s
            w_s = cells.pop_count[m_s] / cells.pop_count[m_s].sum()
            pt, lo, hi, _ = _summarize(nps_cells[:, m_s] @ w_s, alpha)
            seg["label"].append(int(s))
            seg["point"].append(pt)
            seg["lower"].append(lo)
            seg["upper"].append(hi)
            seg["n_resp"].append(int(n_c[m_s].sum()))
        diagnostics["segments"] = seg
    return Estimate(
        point=point,
        lower=lower,
        upper=upper,
        alpha=alpha,
        se=se,
        n_labeled=int(y.size),
        n_unlabeled=round(float(cells.pop_count.sum())) - int(y.size),
        method=method,
        diagnostics=diagnostics,
    )


def mrp(
    y_lab: ArrayLike,
    cell_lab: ArrayLike,
    cells: CellFrame,
    *,
    rng: np.random.Generator,
    alpha: float = 0.1,
    fit: str = "laplace",
    n_draws: int = 4000,
    nuts_draws: int = 2000,
    nuts_tune: int = 1000,
    nuts_chains: int = 4,
) -> Estimate:
    """MRP NPS (see module docstring).

    Parameters
    ----------
    y_lab : array_like, shape (n,)
        Respondents' scores 0..10, n >= 2.
    cell_lab : array_like of int, shape (n,)
        Each respondent's cell index into ``cells``.
    cells : CellFrame
        Population cells (factor levels, population counts, optional segments).
    rng : numpy.random.Generator
        For the posterior draws (Laplace) or the NUTS seed.
    """
    return _mrp(
        y_lab, cell_lab, cells, x_cell=None, rng=rng, alpha=alpha, fit=fit, n_draws=n_draws,
        nuts_draws=nuts_draws, nuts_tune=nuts_tune, nuts_chains=nuts_chains, method="mrp",
    )  # fmt: skip


def mrp_ppi(
    y_lab: ArrayLike,
    cell_lab: ArrayLike,
    cells: CellFrame,
    *,
    x_cell: ArrayLike,
    rng: np.random.Generator,
    alpha: float = 0.1,
    fit: str = "laplace",
    n_draws: int = 4000,
    nuts_draws: int = 2000,
    nuts_tune: int = 1000,
    nuts_chains: int = 4,
) -> Estimate:
    """MRP with the model's predictions as a cell-level covariate (see module docstring).

    ``x_cell`` (C, 2): per cell, the population mean of the predicted promoter logit and
    of the predicted detractor logit. It is standardized internally.
    """
    return _mrp(
        y_lab, cell_lab, cells, x_cell=x_cell, rng=rng, alpha=alpha, fit=fit,
        n_draws=n_draws, nuts_draws=nuts_draws, nuts_tune=nuts_tune,
        nuts_chains=nuts_chains, method="mrp_ppi",
    )  # fmt: skip
