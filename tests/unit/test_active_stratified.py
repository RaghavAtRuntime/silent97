"""Design-based estimators: ppi_active (Poisson, 1/pi weights) and ppi_stratified."""

import math

import numpy as np
import pytest

from silent97 import estimators
from silent97.estimators import active, ppi, stratified
from silent97.estimators.nps import NPS_GRADIENT
from silent97.sampling import designs


def _population(m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores and calibrated-ish predictions; uncertainty varies a lot across customers."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=m)
    noisy = 0.8 * latent + 0.6 * rng.normal(size=m)
    y = np.clip(np.round(7.5 + 2.4 * latent), 0, 10).astype(np.int64)
    p_prom = 1 / (1 + np.exp(-(1.8 * noisy - 0.2)))
    p_det = (1 - p_prom) / (1 + np.exp(1.8 * noisy + 0.4))
    return y, np.column_stack([p_prom, p_det])


def _truth(y: np.ndarray) -> float:
    return float(100 * ((y >= 9).mean() - (y <= 6).mean()))


# --------------------------------------------------------------------------- ppi_active


def test_registered() -> None:
    assert estimators.REGISTRY["ppi_active"] is active.ppi_active
    assert estimators.REGISTRY["ppi_stratified"] is stratified.ppi_stratified


def test_active_point_and_covariance_formula() -> None:
    y, p = _population(500, 0)
    rng = np.random.default_rng(1)
    d = designs.active(60, designs.nps_uncertainty(p), rng)
    lab = np.zeros(500, dtype=bool)
    lab[d.idx] = True
    est = active.ppi_active(y[lab], p[lab], p[~lab], pi_lab=d.pi[lab], lam=0.8)
    z = np.column_stack([y >= 9, y <= 6]).astype(float)
    a = 0.8 * p.copy()
    a[lab] += (z[lab] - 0.8 * p[lab]) / d.pi[lab][:, None]
    theta = a.mean(axis=0)
    cov = np.cov(a.T, ddof=0) / 500
    assert est.point == pytest.approx(float(NPS_GRADIENT @ theta), abs=1e-10)
    assert est.se == pytest.approx(math.sqrt(NPS_GRADIENT @ cov @ NPS_GRADIENT), rel=1e-10)
    assert est.n_labeled == lab.sum() and est.n_unlabeled == 500 - lab.sum()
    assert est.method == "ppi_active"


def test_active_constant_pi_is_ppi_on_the_whole_pool() -> None:
    y, p = _population(400, 2)
    lab = np.zeros(400, dtype=bool)
    lab[:40] = True
    est = active.ppi_active(y[lab], p[lab], p[~lab], pi_lab=np.full(40, 0.1))
    z = np.column_stack([y >= 9, y <= 6]).astype(float)
    theta = p.mean(axis=0) + (z[lab] - p[lab]).sum(axis=0) / (0.1 * 400)
    assert est.point == pytest.approx(float(NPS_GRADIENT @ theta), abs=1e-10)


def test_active_is_unbiased_and_forgetting_weights_is_not() -> None:
    """Probing test: an estimator that ignores 1/pi is biased under the active design."""
    y, p = _population(4000, 3)
    truth = _truth(y)
    u = designs.nps_uncertainty(p)
    rng = np.random.default_rng(4)
    err_w, err_now, cover = [], [], 0
    for _ in range(1500):
        d = designs.active(150, u, rng)
        lab = np.zeros(y.size, dtype=bool)
        lab[d.idx] = True
        est = active.ppi_active(y[lab], p[lab], p[~lab], pi_lab=d.pi[lab])
        err_w.append(est.point - truth)
        cover += est.lower <= truth <= est.upper
        err_now.append(ppi.ppi(y[lab], p[lab], p[~lab]).point - truth)
    w, now = np.array(err_w), np.array(err_now)
    assert abs(w.mean()) < 3 * w.std() / math.sqrt(w.size)
    assert abs(now.mean()) > 5 * now.std() / math.sqrt(now.size)
    assert 0.86 < cover / 1500 < 0.94


def test_active_validation() -> None:
    y, p = _population(100, 5)
    with pytest.raises(ValueError, match="pi_lab"):
        active.ppi_active(y[:10], p[:10], p[10:], pi_lab=np.full(9, 0.5))
    with pytest.raises(ValueError, match="pi_lab"):
        active.ppi_active(y[:10], p[:10], p[10:], pi_lab=np.r_[np.full(9, 0.5), 0.0])
    with pytest.raises(ValueError, match="pi_lab"):
        active.ppi_active(y[:10], p[:10], p[10:], pi_lab=np.r_[np.full(9, 0.5), 1.5])
    with pytest.raises(ValueError):
        active.ppi_active(y[:1], p[:1], p[1:], pi_lab=np.array([0.5]))


# --------------------------------------------------------------------------- ppi_stratified


def test_single_stratum_matches_ppi_and_ppi_pp() -> None:
    y, p = _population(600, 6)
    s_lab, s_unl = np.zeros(80, dtype=int), np.zeros(520, dtype=int)
    one = stratified.ppi_stratified(y[:80], p[:80], p[80:], strata_lab=s_lab, strata_unlab=s_unl)
    ref = ppi.ppi(y[:80], p[:80], p[80:])
    assert one.point == pytest.approx(ref.point, abs=1e-10)
    assert one.se == pytest.approx(ref.se, rel=1e-10)
    plug = stratified.ppi_stratified(
        y[:80], p[:80], p[80:], strata_lab=s_lab, strata_unlab=s_unl, lam="plugin"
    )
    assert plug.point == pytest.approx(ppi.ppi_pp(y[:80], p[:80], p[80:]).point, abs=1e-10)


def test_stratified_combines_with_population_weights() -> None:
    y, p = _population(900, 7)
    strata = np.repeat([0, 1, 2], [300, 500, 100])
    lab = np.zeros(900, dtype=bool)
    lab[np.r_[0:20, 300:330, 800:850]] = True  # stratum 2 heavily oversampled
    est = stratified.ppi_stratified(
        y[lab], p[lab], p[~lab], strata_lab=strata[lab], strata_unlab=strata[~lab]
    )
    parts = []
    for h, size in zip(range(3), (300, 500, 100), strict=True):
        m_l, m_u = lab & (strata == h), ~lab & (strata == h)
        e = ppi.ppi(y[m_l], p[m_l], p[m_u])
        parts.append((size / 900, e.point, e.se))
    assert est.point == pytest.approx(sum(w * pt for w, pt, _ in parts), abs=1e-10)
    assert est.se == pytest.approx(math.sqrt(sum((w * se) ** 2 for w, _, se in parts)), rel=1e-10)
    assert est.diagnostics["n_strata"] == 3


def test_stratified_unbiased_under_neyman_while_plain_ppi_is_not() -> None:
    y, p = _population(3000, 8)
    truth = _truth(y)
    strata = np.digitize(p[:, 0] - p[:, 1], [-0.3, 0.3])  # 3 strata by predicted NPS
    # Skewed allocation: unequal sampling rates across strata (n_h about 63 / 36 / 21).
    # A stratum with only a handful of responses would make its variance estimate
    # unreliable (see docs/progress/phase-4.md); that is reported, not tested here.
    sig = designs.neyman_sigma(strata, p) * np.array([2.0, 1.0, 0.5])
    rng = np.random.default_rng(9)
    err_s, err_p, cover = [], [], 0
    for _ in range(1200):
        d = designs.stratified(120, strata, rng, allocation="neyman", sigma=sig)
        lab = np.zeros(y.size, dtype=bool)
        lab[d.idx] = True
        e = stratified.ppi_stratified(
            y[lab], p[lab], p[~lab], strata_lab=strata[lab], strata_unlab=strata[~lab]
        )
        err_s.append(e.point - truth)
        cover += e.lower <= truth <= e.upper
        err_p.append(ppi.ppi(y[lab], p[lab], p[~lab]).point - truth)
    s, pl = np.array(err_s), np.array(err_p)
    assert abs(s.mean()) < 3 * s.std() / math.sqrt(s.size)
    assert abs(pl.mean()) > 5 * pl.std() / math.sqrt(pl.size)
    assert 0.86 < cover / 1200 < 0.94


def test_stratified_crossfit_lambda_needs_rng() -> None:
    y, p = _population(400, 10)
    s_lab, s_unl = np.repeat([0, 1], 20), np.repeat([0, 1], 180)
    with pytest.raises(ValueError, match="rng"):
        stratified.ppi_stratified(
            y[:40], p[:40], p[40:], strata_lab=s_lab, strata_unlab=s_unl, lam="crossfit"
        )
    e = stratified.ppi_stratified(
        y[:40], p[:40], p[40:], strata_lab=s_lab, strata_unlab=s_unl, lam="crossfit",
        rng=np.random.default_rng(0),
    )  # fmt: skip
    assert math.isfinite(e.point) and e.se > 0


def test_stratified_validation() -> None:
    y, p = _population(200, 11)
    s_lab = np.r_[np.zeros(19, dtype=int), 1]
    with pytest.raises(ValueError, match="at least 2"):
        stratified.ppi_stratified(
            y[:20], p[:20], p[20:], strata_lab=s_lab, strata_unlab=np.repeat([0, 1], 90)
        )
    with pytest.raises(ValueError, match="unlabeled"):
        stratified.ppi_stratified(
            y[:20], p[:20], p[20:], strata_lab=np.repeat([0, 1], 10),
            strata_unlab=np.zeros(180, dtype=int),
        )  # fmt: skip
    with pytest.raises(ValueError, match="lam"):
        stratified.ppi_stratified(
            y[:20], p[:20], p[20:], strata_lab=np.zeros(20, dtype=int),
            strata_unlab=np.zeros(180, dtype=int), lam="best",
        )  # fmt: skip
