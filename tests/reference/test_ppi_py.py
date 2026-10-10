"""Cross-checks against the reference implementation ``ppi_py`` (acceptance criterion A2).

Point estimates and interval bounds must agree to 1e-6 across 25 seeds and several
(n, N) pairs: for scalar means (our generic core) and for the NPS contrast (our NPS
estimators against ``ppi_py`` applied to v = g'Z and h = g'f).
"""

import itertools

import numpy as np
import pytest
from ppi_py import classical_mean_ci, crossppi_mean_ci, ppi_mean_ci, ppi_mean_pointestimate
from ppi_py.cross_ppi import crossppi_mean_pointestimate
from ppi_py.ppi import _calc_lam_glm

from silent97.estimators import classical, nps, ppi

TOL = 1e-6
SEEDS = range(25)
SIZES = [(50, 500), (100, 5_000), (500, 2_000), (1_000, 100_000)]
CASES = list(itertools.product(SEEDS, SIZES))
G = nps.NPS_GRADIENT


def _scalar(seed: int, n: int, big_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real-valued outcome with a biased, noisy prediction."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n + big_n)
    y = 2.0 + x + rng.normal(scale=0.8, size=n + big_n)
    yhat = 0.3 + 0.8 * x + rng.normal(scale=0.5 + 0.1 * (seed % 5), size=n + big_n)
    return y[:n], yhat[:n], yhat[n:]


def _nps(seed: int, n: int, big_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 11, size=n + big_n)
    quality = 0.2 + 0.03 * (seed % 20)
    p = (
        quality * nps.indicators(y)
        + (1 - quality) * rng.dirichlet([2, 3, 2], size=n + big_n)[:, :2]
    )
    return y[:n], p[:n], p[n:]


def _ref_lambda(y: np.ndarray, yhat: np.ndarray, yhat_unlab: np.ndarray, clip: bool) -> float:
    """``ppi_py``'s own lambda (its _calc_lam_glm, as called by ppi_mean_ci)."""
    y2, f2, u2 = y[:, None], yhat[:, None], yhat_unlab[:, None]
    theta = ppi_mean_pointestimate(y2, f2, u2, lam=1)
    return float(
        _calc_lam_glm(y2 - theta, f2 - theta, u2 - theta, np.eye(1), coord=None, clip=clip)
    )


def _close(ours: float, ref: object) -> None:
    assert float(ours) == pytest.approx(float(np.asarray(ref).squeeze()), abs=TOL, rel=0)


# --------------------------------------------------------------------------- scalar means


@pytest.mark.parametrize("seed, sizes", CASES)
def test_scalar_ppi_lambda_one(seed: int, sizes: tuple[int, int]) -> None:
    y, yhat, yhat_u = _scalar(seed, *sizes)
    ours = ppi.ppi_mean_ci(y, yhat, yhat_u, alpha=0.1, lam=1.0)
    lo, hi = ppi_mean_ci(y, yhat, yhat_u, alpha=0.1, lam=1)
    _close(ours.point[0], ppi_mean_pointestimate(y, yhat, yhat_u, lam=1))
    _close(ours.lower[0], lo)
    _close(ours.upper[0], hi)


@pytest.mark.parametrize("seed, sizes", CASES)
def test_scalar_ppi_pp_clipped_matches_ppi_py_default(seed: int, sizes: tuple[int, int]) -> None:
    y, yhat, yhat_u = _scalar(seed, *sizes)
    ours = ppi.ppi_mean_ci(y, yhat, yhat_u, alpha=0.1, clip=True)
    lo, hi = ppi_mean_ci(y, yhat, yhat_u, alpha=0.1)
    _close(ours.lam, _ref_lambda(y, yhat, yhat_u, clip=True))
    _close(ours.point[0], ppi_mean_pointestimate(y, yhat, yhat_u))
    _close(ours.lower[0], lo)
    _close(ours.upper[0], hi)


@pytest.mark.parametrize("seed, sizes", CASES)
def test_scalar_ppi_pp_unclipped(seed: int, sizes: tuple[int, int]) -> None:
    """Default (paper) variant: same lambda formula without clipping."""
    y, yhat, yhat_u = _scalar(seed, *sizes)
    ours = ppi.ppi_mean_ci(y, yhat, yhat_u, alpha=0.1)
    ref_lam = _ref_lambda(y, yhat, yhat_u, clip=False)
    _close(ours.lam, ref_lam)
    lo, hi = ppi_mean_ci(y, yhat, yhat_u, alpha=0.1, lam=ref_lam)
    _close(ours.lower[0], lo)
    _close(ours.upper[0], hi)


@pytest.mark.parametrize("seed, sizes", CASES)
def test_scalar_crossfit(seed: int, sizes: tuple[int, int]) -> None:
    n, big_n = sizes
    y, yhat, _ = _scalar(seed, n, big_n)
    rng = np.random.default_rng(seed + 100)
    yhat_u = rng.normal(size=(big_n, 1)) + rng.normal(scale=0.1, size=(big_n, 5))  # (N, K)
    ours = ppi.ppi_mean_ci(y, yhat, yhat_u.mean(axis=1), alpha=0.1, lam=1.0)
    lo, hi = crossppi_mean_ci(y, yhat, yhat_u, alpha=0.1)
    _close(ours.point[0], crossppi_mean_pointestimate(y, yhat, yhat_u))
    _close(ours.lower[0], lo)
    _close(ours.upper[0], hi)


# --------------------------------------------------------------------------- NPS contrast


@pytest.mark.parametrize("seed, sizes", CASES)
def test_nps_classical(seed: int, sizes: tuple[int, int]) -> None:
    y, _, _ = _nps(seed, *sizes)
    est = classical.classical(y, alpha=0.1)
    lo, hi = classical_mean_ci(nps.nps_values(y), alpha=0.1)
    _close(est.point, nps.nps_values(y).mean())
    _close(est.lower, lo)
    _close(est.upper, hi)


@pytest.mark.parametrize("seed, sizes", CASES)
def test_nps_ppi(seed: int, sizes: tuple[int, int]) -> None:
    y, p_lab, p_unlab = _nps(seed, *sizes)
    v, h, h_u = nps.nps_values(y), p_lab @ G, p_unlab @ G
    est = ppi.ppi(y, p_lab, p_unlab, alpha=0.1)
    lo, hi = ppi_mean_ci(v, h, h_u, alpha=0.1, lam=1)
    _close(est.point, ppi_mean_pointestimate(v, h, h_u, lam=1))
    _close(est.lower, lo)
    _close(est.upper, hi)


@pytest.mark.parametrize("seed, sizes", CASES)
@pytest.mark.parametrize("clip", [True, False])
def test_nps_ppi_pp(seed: int, sizes: tuple[int, int], clip: bool) -> None:
    """2-D PPI++ with lambda optimized for the contrast == scalar PPI++ on (v, h)."""
    y, p_lab, p_unlab = _nps(seed, *sizes)
    v, h, h_u = nps.nps_values(y), p_lab @ G, p_unlab @ G
    est = ppi.ppi_pp(y, p_lab, p_unlab, alpha=0.1, clip=clip)
    ref_lam = _ref_lambda(v, h, h_u, clip=clip)
    _close(est.diagnostics["lambda_hat"], ref_lam)
    lo, hi = ppi_mean_ci(v, h, h_u, alpha=0.1, lam=ref_lam)
    _close(est.point, ppi_mean_pointestimate(v, h, h_u, lam=ref_lam))
    _close(est.lower, lo)
    _close(est.upper, hi)
    if clip:
        lo_d, hi_d = ppi_mean_ci(v, h, h_u, alpha=0.1)  # ppi_py default path
        _close(est.lower, lo_d)
        _close(est.upper, hi_d)


@pytest.mark.parametrize("seed, sizes", CASES)
def test_nps_crossfit(seed: int, sizes: tuple[int, int]) -> None:
    n, big_n = sizes
    y, p_lab, p_unlab = _nps(seed, n, big_n)
    rng = np.random.default_rng(seed + 200)
    folds = p_unlab[:, None, :] * rng.uniform(0.85, 1.0, size=(big_n, 4, 1))  # (N, K, 2)
    est = ppi.ppi_crossfit(y, p_lab, folds, alpha=0.1)
    v, h = nps.nps_values(y), p_lab @ G
    h_u = folds @ G  # (N, K)
    lo, hi = crossppi_mean_ci(v, h, h_u, alpha=0.1)
    _close(est.point, crossppi_mean_pointestimate(v, h, h_u))
    _close(est.lower, lo)
    _close(est.upper, hi)


def test_lambda_outside_unit_interval_is_exercised() -> None:
    """The unclipped and clipped variants genuinely differ on some inputs."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=2_000)
    p = 0.3 * nps.indicators(y) + 0.2
    v, h = nps.nps_values(y), p @ G
    est = ppi.ppi_pp(y[:200], p[:200], p[200:])
    ref_lam = _ref_lambda(v[:200], h[:200], h[200:], clip=False)
    assert ref_lam > 1.0
    _close(est.diagnostics["lambda_hat"], ref_lam)
