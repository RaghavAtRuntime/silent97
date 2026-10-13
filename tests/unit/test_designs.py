"""Sampling designs: every design returns pi > 0 for every pool customer (CLAUDE.md rule 5)."""

import numpy as np
import pytest

from silent97.sampling import designs
from silent97.types import Design


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- uniform


def test_uniform() -> None:
    d = designs.uniform(50, 1000, _rng())
    assert isinstance(d, Design) and d.name == "uniform"
    assert d.idx.size == 50 and np.unique(d.idx).size == 50
    assert d.pi.shape == (1000,)
    np.testing.assert_allclose(d.pi, 0.05)
    assert (d.idx >= 0).all() and (d.idx < 1000).all()


def test_uniform_is_seeded() -> None:
    a, b = designs.uniform(10, 100, _rng(3)), designs.uniform(10, 100, _rng(3))
    np.testing.assert_array_equal(a.idx, b.idx)


@pytest.mark.parametrize("n, size", [(0, 10), (11, 10), (-1, 10)])
def test_uniform_rejects_bad_n(n: int, size: int) -> None:
    with pytest.raises(ValueError, match="n"):
        designs.uniform(n, size, _rng())


# --------------------------------------------------------------------------- stratified


def _strata() -> np.ndarray:
    return np.repeat([0, 1, 2, 3], [100, 300, 400, 200])


def test_stratified_proportional() -> None:
    strata = _strata()
    d = designs.stratified(100, strata, _rng(), allocation="proportional")
    assert d.name == "stratified_proportional"
    counts = np.bincount(strata[d.idx], minlength=4)
    np.testing.assert_array_equal(counts, [10, 30, 40, 20])
    sizes = np.bincount(strata)
    np.testing.assert_allclose(d.pi, (counts / sizes)[strata])
    assert np.unique(d.idx).size == 100


def test_stratified_neyman_favours_variable_strata() -> None:
    strata = _strata()
    sigma = np.array([1.0, 1.0, 1.0, 4.0])
    d = designs.stratified(100, strata, _rng(), allocation="neyman", sigma=sigma)
    counts = np.bincount(strata[d.idx], minlength=4)
    assert counts.sum() == 100
    # n_h proportional to N_h * sigma_h: 100, 300, 400, 800 (out of 1600).
    np.testing.assert_array_equal(counts, [6, 19, 25, 50])
    assert d.name == "stratified_neyman"


def test_stratified_minimum_per_stratum_and_caps() -> None:
    strata = np.repeat([0, 1], [3, 997])
    d = designs.stratified(20, strata, _rng(), allocation="proportional")
    counts = np.bincount(strata[d.idx], minlength=2)
    assert counts[0] >= 2 and counts.sum() == 20
    d2 = designs.stratified(12, strata, _rng(), allocation="neyman", sigma=np.array([100.0, 1.0]))
    assert np.bincount(strata[d2.idx], minlength=2)[0] == 3  # capped at N_h
    assert (d2.pi > 0).all() and (d2.pi <= 1).all()


def test_stratified_validation() -> None:
    strata = _strata()
    with pytest.raises(ValueError, match="sigma"):
        designs.stratified(100, strata, _rng(), allocation="neyman")
    with pytest.raises(ValueError, match="allocation"):
        designs.stratified(100, strata, _rng(), allocation="optimal")
    with pytest.raises(ValueError, match="2 per stratum"):
        designs.stratified(6, strata, _rng())


def test_neyman_sigma_from_predictions() -> None:
    p = np.array([[0.9, 0.05], [0.9, 0.05], [0.4, 0.4], [0.1, 0.8]])
    strata = np.array([0, 0, 1, 1])
    sig = designs.neyman_sigma(strata, p)
    # Stratum 0: identical rows, so sigma = sqrt(within) = sqrt(pP + pD - (pP - pD)^2).
    assert sig[0] == pytest.approx(np.sqrt(0.95 - 0.85**2))
    assert sig[1] > sig[0]


# --------------------------------------------------------------------------- active


def test_nps_uncertainty() -> None:
    p = np.array([[0.5, 0.2], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(designs.nps_uncertainty(p), np.sqrt([0.7 - 0.09, 0.0, 0.0]))


def test_active_probabilities_budget_floor_and_order() -> None:
    rng = _rng(1)
    u = rng.uniform(0, 1, size=5000) ** 3
    pi = designs.active_probabilities(200, u, floor=0.1)
    assert pi.shape == (5000,)
    assert pi.sum() == pytest.approx(200)
    assert pi.min() >= 0.1 * 200 / 5000 - 1e-12
    assert pi.max() <= 1
    order = np.argsort(u)
    assert (np.diff(pi[order]) >= -1e-12).all()


def test_active_probabilities_cap_at_one_keeps_budget() -> None:
    u = np.ones(100)
    u[:3] = 1e6
    pi = designs.active_probabilities(20, u, floor=0.1)
    np.testing.assert_allclose(pi[:3], 1.0)
    assert pi.sum() == pytest.approx(20)
    assert (pi[3:] > 0).all()


def test_active_with_equal_uncertainty_is_uniform_rate() -> None:
    pi = designs.active_probabilities(10, np.full(100, 0.3), floor=0.1)
    np.testing.assert_allclose(pi, 0.1)
    pi0 = designs.active_probabilities(10, np.zeros(100), floor=0.1)
    np.testing.assert_allclose(pi0, 0.1)


def test_active_poisson_sampling_matches_pi() -> None:
    rng = _rng(2)
    u = np.linspace(0.05, 1, 400)
    sizes, hits = [], np.zeros(400)
    for _ in range(4000):
        d = designs.active(40, u, rng, floor=0.1)
        sizes.append(d.idx.size)
        hits[d.idx] += 1
    assert d.name == "active"
    assert np.mean(sizes) == pytest.approx(40, abs=0.5)
    pi = designs.active_probabilities(40, u, floor=0.1)
    se = np.sqrt(pi * (1 - pi) / 4000)
    assert (np.abs(hits / 4000 - pi) < 5 * se + 1e-3).all()


def test_active_validation() -> None:
    with pytest.raises(ValueError, match="floor"):
        designs.active_probabilities(10, np.ones(100), floor=0.0)
    with pytest.raises(ValueError, match="uncertainty"):
        designs.active_probabilities(10, np.array([1.0, -1.0, 1.0]), floor=0.1)
    with pytest.raises(ValueError, match="uncertainty"):
        designs.active_probabilities(1, np.array([1.0, np.nan]), floor=0.1)
    with pytest.raises(ValueError, match="n"):
        designs.active_probabilities(200, np.ones(100), floor=0.1)


def test_check_pi_asserts_positivity() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        designs.check_pi(np.array([0.1, 0.0]))
    with pytest.raises(ValueError, match="at most 1"):
        designs.check_pi(np.array([0.1, 1.2]))
    designs.check_pi(np.array([0.1, 1.0]))
