"""MRP: hierarchical binomial models (Laplace and NUTS) poststratified to population cells."""

import math

import numpy as np
import pytest

from silent97.estimators import mrp


def _cells(rng: np.random.Generator, sd: float = 0.4) -> mrp.CellFrame:
    """240 cells: province (10) x age (6) x tenure (4), with random population counts."""
    prov, age, ten = np.meshgrid(np.arange(10), np.arange(6), np.arange(4), indexing="ij")
    factors = np.column_stack([prov.ravel(), age.ravel(), ten.ravel(), (prov * 6 + age).ravel()])
    pop = rng.integers(200, 2000, size=factors.shape[0])
    return mrp.CellFrame(
        factors=factors,
        factor_names=("province", "age", "tenure", "segment"),
        pop_count=pop,
        segment=factors[:, 3],
    )


def _truth_probs(cells: mrp.CellFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    def probs(base: float) -> np.ndarray:
        eta = np.full(cells.n_cells, base)
        for f, size in enumerate(cells.factor_sizes):
            eta += 0.4 * rng.normal(size=size)[cells.factors[:, f]]
        return 1 / (1 + np.exp(-eta))

    p_prom = probs(-0.3)
    p_det = (1 - p_prom) * probs(-0.6)
    return p_prom, p_det


def _survey(
    cells: mrp.CellFrame, p_prom: np.ndarray, p_det: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    cell = rng.choice(cells.n_cells, size=n, p=cells.pop_count / cells.pop_count.sum())
    u = rng.random(n)
    y = np.where(u < p_prom[cell], 10, np.where(u < p_prom[cell] + p_det[cell], 3, 7))
    return y.astype(np.int64), cell.astype(np.int64)


def test_cell_frame_validation() -> None:
    with pytest.raises(ValueError, match="pop_count"):
        mrp.CellFrame(
            factors=np.zeros((3, 1), int), factor_names=("a",), pop_count=np.array([1, 0, 2])
        )
    with pytest.raises(ValueError, match="factor_names"):
        mrp.CellFrame(factors=np.zeros((3, 2), int), factor_names=("a",), pop_count=np.ones(3))


def test_laplace_mode_matches_direct_optimization() -> None:
    """For fixed sds, the Newton conditional mode equals a generic optimizer's answer."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    cells = _cells(rng)
    k = rng.integers(0, 6, cells.n_cells)
    n = k + rng.integers(0, 6, cells.n_cells)
    design = mrp._design(cells, None)
    log_sd = np.log(np.array([0.5, 0.3, 0.2, 0.4]))
    mode, _, _ = mrp._conditional_mode(design, k, n, log_sd)

    def neg(beta_u: np.ndarray) -> float:
        return -mrp._log_joint(design, k, n, log_sd, beta_u)

    ref = minimize(neg, np.zeros_like(mode), method="BFGS", options={"gtol": 1e-8})
    np.testing.assert_allclose(mode, ref.x, atol=1e-4)


def test_consistent_with_large_cells() -> None:
    """With data from the (additive) model and very large cells, the fit recovers each
    cell's true probability. (An additive model cannot reproduce arbitrary cell-level
    interactions, so the data are generated from it.)"""
    rng = np.random.default_rng(1)
    cells = _cells(rng)
    p_prom, _ = _truth_probs(cells, rng)
    n = rng.integers(4000, 6000, cells.n_cells)
    k = rng.binomial(n, p_prom)
    draws = mrp.fit_cell_probabilities(cells, k, n, rng=rng, fit="laplace", n_draws=200)
    np.testing.assert_allclose(draws.mean(axis=0), p_prom, atol=0.015)


def test_mrp_recovers_population_nps() -> None:
    rng = np.random.default_rng(2)
    cells = _cells(rng)
    p_prom, p_det = _truth_probs(cells, rng)
    w = cells.pop_count / cells.pop_count.sum()
    truth = 100 * float(w @ (p_prom - p_det))
    y, cell = _survey(cells, p_prom, p_det, 20_000, rng)
    est = mrp.mrp(y, cell, cells, rng=rng, alpha=0.1)
    assert est.method == "mrp"
    assert abs(est.point - truth) < 3 * est.se
    assert est.lower < est.point < est.upper
    assert est.diagnostics["fit"] == "laplace"


def test_mrp_interval_coverage_small_monte_carlo() -> None:
    """MRP intervals (Laplace) cover the population NPS about as often as stated."""
    rng = np.random.default_rng(3)
    cells = _cells(rng)
    p_prom, p_det = _truth_probs(cells, rng)
    w = cells.pop_count / cells.pop_count.sum()
    truth = 100 * float(w @ (p_prom - p_det))
    hits = 0
    for _ in range(120):
        y, cell = _survey(cells, p_prom, p_det, 600, rng)
        e = mrp.mrp(y, cell, cells, rng=rng, alpha=0.1, n_draws=1000)
        hits += e.lower <= truth <= e.upper
    assert 0.8 <= hits / 120 <= 0.98


def test_segment_estimates_shrink_thin_segments() -> None:
    rng = np.random.default_rng(4)
    cells = _cells(rng)
    p_prom, p_det = _truth_probs(cells, rng)
    y, cell = _survey(cells, p_prom, p_det, 400, rng)
    est = mrp.mrp(y, cell, cells, rng=rng)
    seg = est.diagnostics["segments"]
    assert set(seg) == {"label", "point", "lower", "upper", "n_resp"}
    assert len(seg["label"]) == 60
    assert all(
        lo <= pt <= hi for lo, pt, hi in zip(seg["lower"], seg["point"], seg["upper"], strict=True)
    )
    assert sum(seg["n_resp"]) == 400


def test_mrp_ppi_uses_the_cell_covariate() -> None:
    """An informative cell covariate improves the thin segment estimates. (The overall
    NPS uncertainty is dominated by the overall level, which still comes from the
    respondents, so its se barely changes.)"""
    rng = np.random.default_rng(5)
    cells = _cells(rng)
    p_prom, p_det = _truth_probs(cells, rng)
    x = np.column_stack([np.log(p_prom / (1 - p_prom)), np.log(p_det / (1 - p_det))])
    seg_truth = []
    for s in range(60):
        m = cells.segment == s
        w = cells.pop_count[m] / cells.pop_count[m].sum()
        seg_truth.append(100 * float(w @ (p_prom[m] - p_det[m])))
    err_plain, err_x, slopes = [], [], []
    for r in range(6):
        y, cell = _survey(cells, p_prom, p_det, 500, rng)
        plain = mrp.mrp(y, cell, cells, rng=np.random.default_rng(r))
        with_x = mrp.mrp_ppi(y, cell, cells, x_cell=x, rng=np.random.default_rng(r))
        assert with_x.method == "mrp_ppi"
        slopes.append(with_x.diagnostics["slope_promoter"])
        err_plain += list(np.array(plain.diagnostics["segments"]["point"]) - seg_truth)
        err_x += list(np.array(with_x.diagnostics["segments"]["point"]) - seg_truth)
    assert min(slopes) > 0.3
    rmse_plain = float(np.sqrt(np.mean(np.square(err_plain))))
    rmse_x = float(np.sqrt(np.mean(np.square(err_x))))
    assert rmse_x < 0.8 * rmse_plain, (rmse_x, rmse_plain)


def test_convergence_check() -> None:
    mrp.check_convergence(np.array([1.0, 1.005]), np.array([900.0, 450.0]))
    with pytest.raises(mrp.ConvergenceError, match="R-hat"):
        mrp.check_convergence(np.array([1.0, 1.02]), np.array([900.0, 900.0]))
    with pytest.raises(mrp.ConvergenceError, match="ESS"):
        mrp.check_convergence(np.array([1.0, 1.0]), np.array([900.0, 120.0]))


def test_input_validation() -> None:
    rng = np.random.default_rng(6)
    cells = _cells(rng)
    with pytest.raises(ValueError, match="cell"):
        mrp.mrp(np.array([10, 3]), np.array([0, 10_000]), cells, rng=rng)
    with pytest.raises(ValueError, match="at least 2"):
        mrp.mrp(np.array([10]), np.array([0]), cells, rng=rng)
    with pytest.raises(ValueError, match="fit"):
        mrp.mrp(np.array([10, 3]), np.array([0, 1]), cells, rng=rng, fit="vi")
    with pytest.raises(ValueError, match="x_cell"):
        mrp.mrp_ppi(np.array([10, 3]), np.array([0, 1]), cells, x_cell=np.zeros((3, 2)), rng=rng)


@pytest.mark.slow
def test_nuts_agrees_with_laplace() -> None:
    rng = np.random.default_rng(7)
    cells = _cells(rng)
    p_prom, p_det = _truth_probs(cells, rng)
    y, cell = _survey(cells, p_prom, p_det, 800, rng)
    lap = mrp.mrp(y, cell, cells, rng=np.random.default_rng(0))
    nuts = mrp.mrp(y, cell, cells, rng=np.random.default_rng(0), fit="nuts")
    assert nuts.diagnostics["fit"] == "nuts"
    assert nuts.diagnostics["max_rhat"] <= 1.01 and nuts.diagnostics["min_ess_bulk"] >= 400
    # Laplace plugs in the estimated sds (glmer-style), so a small shift is expected;
    # the systematic NUTS-vs-Laplace comparison is in E4/E5 (phase 4 note).
    assert abs(nuts.point - lap.point) < 0.5 * lap.se
    assert math.isclose(nuts.se, lap.se, rel_tol=0.2)
