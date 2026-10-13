"""PPI++ with a cross-fitted lambda (``ppi_pp_cf``): tests written before the estimator."""

import math

import numpy as np
import pytest

from silent97 import estimators
from silent97.estimators import classical, ppi
from silent97.estimators.nps import NPS_GRADIENT


def _skewed(n: int, big_n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """A skewed population like the gift-card reviews (about 84% promoters)."""
    m = n + big_n
    latent = rng.normal(size=m)
    noisy = 0.85 * latent + math.sqrt(1 - 0.85**2) * rng.normal(size=m)
    y = np.where(latent > -1.0, 10, np.where(latent > -1.25, 8, 2)).astype(np.int64)
    p_prom = 1 / (1 + np.exp(-(2.2 * noisy + 2.3)))
    p_det = (1 - p_prom) / (1 + np.exp(1.5 * noisy + 0.5))
    p = np.column_stack([p_prom, p_det])
    return y[:n], p[:n], p[n:]


def _data(seed: int = 0, n: int = 60, big_n: int = 2000) -> tuple[np.ndarray, ...]:
    return _skewed(n, big_n, np.random.default_rng(seed))


def test_registered_and_valid_under_mcar() -> None:
    assert estimators.REGISTRY["ppi_pp_cf"] is ppi.ppi_pp_cf
    assert "ppi_pp_cf" in estimators.VALID_UNDER_MCAR


def test_equal_fold_lambdas_reduce_to_ppi_with_that_lambda() -> None:
    y, p_l, p_u = _data()
    z = np.column_stack([y >= 9, y <= 6]).astype(float)
    folds = np.arange(y.size) % 5
    est = ppi.ppi_cf_from_lambdas(z, p_l, p_u, folds, np.full(5, 0.7), alpha=0.1)
    ref = ppi._nps_ppi(z, p_l, p_u, 0.7, 0.1, "ref", {})
    assert est.point == pytest.approx(ref.point, abs=1e-12)
    assert est.se == pytest.approx(ref.se, rel=1e-12)


def test_point_formula_with_different_fold_lambdas() -> None:
    y, p_l, p_u = _data(1)
    z = np.column_stack([y >= 9, y <= 6]).astype(float)
    folds = np.arange(y.size) % 3
    lams = np.array([0.2, 0.9, 1.4])
    est = ppi.ppi_cf_from_lambdas(z, p_l, p_u, folds, lams, alpha=0.1)
    lam_i = lams[folds]
    w = np.bincount(folds) / y.size
    theta = (z - lam_i[:, None] * p_l).mean(axis=0) + (w @ lams) * p_u.mean(axis=0)
    assert est.point == pytest.approx(float(NPS_GRADIENT @ theta), abs=1e-10)
    assert est.diagnostics["lambda_hat"] == pytest.approx(float(w @ lams))


def test_fold_lambda_uses_only_out_of_fold_labeled_rows() -> None:
    y, p_l, p_u = _data(2)
    folds = np.arange(y.size) % 4
    base = ppi.ppi_pp_cf(y, p_l, p_u, folds=folds)
    y2 = y.copy()
    y2[folds == 0] = 10 - y2[folds == 0]  # change fold 0's answers only
    changed = ppi.ppi_pp_cf(y2, p_l, p_u, folds=folds)
    assert changed.diagnostics["lambda_folds"][0] == base.diagnostics["lambda_folds"][0]
    assert changed.diagnostics["lambda_folds"][1] != base.diagnostics["lambda_folds"][1]
    # Unlabeled predictions enter lambda only through N (same N, different values).
    other_u = np.random.default_rng(9).dirichlet([1, 1, 1], size=p_u.shape[0])[:, :2]
    again = ppi.ppi_pp_cf(y, p_l, other_u, folds=folds)
    assert again.diagnostics["lambda_folds"] == base.diagnostics["lambda_folds"]


def test_rng_folds_are_reproducible_and_balanced() -> None:
    y, p_l, p_u = _data(3)
    a = ppi.ppi_pp_cf(y, p_l, p_u, n_folds=10, rng=np.random.default_rng(4))
    b = ppi.ppi_pp_cf(y, p_l, p_u, n_folds=10, rng=np.random.default_rng(4))
    assert a == b
    assert a.method == "ppi_pp_cf" and a.diagnostics["n_folds"] == 10
    assert len(a.diagnostics["lambda_folds"]) == 10


def test_permuting_rows_with_their_folds_changes_nothing() -> None:
    y, p_l, p_u = _data(5)
    folds = np.random.default_rng(0).integers(0, 5, size=y.size)
    est = ppi.ppi_pp_cf(y, p_l, p_u, folds=folds)
    perm = np.random.default_rng(1).permutation(y.size)
    est_p = ppi.ppi_pp_cf(y[perm], p_l[perm], p_u[::-1], folds=folds[perm])
    assert est_p.point == pytest.approx(est.point, abs=1e-10)
    assert est_p.se == pytest.approx(est.se, rel=1e-10)


def test_constant_predictions_reduce_to_classical() -> None:
    y, _, _ = _data(6, n=200)
    p_l = np.tile([0.5, 0.2], (y.size, 1))
    p_u = np.tile([0.5, 0.2], (3000, 1))
    est = ppi.ppi_pp_cf(y, p_l, p_u, n_folds=5, rng=np.random.default_rng(0))
    ref = classical.classical(y)
    assert est.point == pytest.approx(ref.point, abs=1e-10)
    assert est.se == pytest.approx(ref.se, rel=0.01)
    assert est.diagnostics["lambda_folds"] == [0.0] * 5


def test_removes_the_same_sample_lambda_bias() -> None:
    """Monte Carlo on a skewed population at small n: plug-in PPI++ is biased upward,
    the cross-fitted version is unbiased (within Monte Carlo error)."""
    rng = np.random.default_rng(2026)
    y_all, p_all, _ = _skewed(400_000, 0, rng)
    v = 100.0 * ((y_all >= 9).astype(float) - (y_all <= 6))
    truth = float(v.mean())
    err_pp, err_cf = [], []
    for r in range(3000):
        idx = rng.choice(y_all.size, size=30 + 4000, replace=False)
        lab, unl = idx[:30], idx[30:]
        args = (y_all[lab], p_all[lab], p_all[unl])
        err_pp.append(ppi.ppi_pp(*args).point - truth)
        err_cf.append(ppi.ppi_pp_cf(*args, n_folds=10, rng=np.random.default_rng(r)).point - truth)
    pp, cf = np.array(err_pp), np.array(err_cf)
    mc_cf = cf.std() / math.sqrt(cf.size)
    assert abs(cf.mean()) < 3 * mc_cf, f"cross-fit bias {cf.mean():.3f} (MC se {mc_cf:.3f})"
    assert pp.mean() > 3 * pp.std() / math.sqrt(pp.size), "plug-in bias not detected"


def test_input_validation() -> None:
    y, p_l, p_u = _data(7)
    with pytest.raises(ValueError, match="folds or rng"):
        ppi.ppi_pp_cf(y, p_l, p_u)
    with pytest.raises(ValueError, match="n_folds"):
        ppi.ppi_pp_cf(y[:5], p_l[:5], p_u, n_folds=10, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="folds"):
        ppi.ppi_pp_cf(y, p_l, p_u, folds=np.zeros(y.size - 1, dtype=int))
    with pytest.raises(ValueError, match="at least 2"):
        folds = np.zeros(y.size, dtype=int)
        folds[-1] = 1  # fold 0 leaves only one out-of-fold row
        ppi.ppi_pp_cf(y, p_l, p_u, folds=folds)
    with pytest.raises(ValueError):
        ppi.ppi_pp_cf(y[:1], p_l[:1], p_u, folds=np.zeros(1, dtype=int))
    bad = p_l.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        ppi.ppi_pp_cf(y, bad, p_u, n_folds=5, rng=np.random.default_rng(0))
