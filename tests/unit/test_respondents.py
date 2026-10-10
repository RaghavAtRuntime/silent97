"""Tests for fixed-n respondent draws and the disjointness guard in the harness."""

import numpy as np
import pytest

from silent97.eval import harness


def test_draw_basic() -> None:
    pool = np.arange(10, 1010)
    idx = harness.draw_respondents(pool, np.ones(1000), 50, np.random.default_rng(0))
    assert idx.shape == (50,)
    assert len(np.unique(idx)) == 50
    assert np.isin(idx, pool).all()
    assert np.all(np.diff(idx) > 0)


def test_draw_deterministic() -> None:
    pool = np.arange(500)
    w = np.linspace(1, 2, 500)
    a = harness.draw_respondents(pool, w, 40, np.random.default_rng(7))
    b = harness.draw_respondents(pool, w, 40, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_equal_weights_uniform_inclusion() -> None:
    pool = np.arange(20)
    rng = np.random.default_rng(1)
    counts = np.zeros(20)
    reps = 20_000
    for _ in range(reps):
        counts[harness.draw_respondents(pool, np.ones(20), 5, rng)] += 1
    # Each unit's inclusion probability is 5/20; sd of the frequency ~0.003.
    np.testing.assert_allclose(counts / reps, 0.25, atol=0.015)


def test_weights_raise_inclusion() -> None:
    pool = np.arange(1000)
    w = np.where(pool < 500, 1.0, 4.0)
    rng = np.random.default_rng(2)
    counts = np.zeros(1000)
    for _ in range(400):
        counts[harness.draw_respondents(pool, w, 20, rng)] += 1
    ratio = counts[500:].sum() / counts[:500].sum()
    assert 3.5 < ratio < 4.5  # n << N, so inclusion is close to proportional


@pytest.mark.parametrize(
    "weights, n, match",
    [
        (np.ones(10), 11, "exceeds"),
        (np.ones(10), 0, "positive"),
        (np.r_[np.ones(9), 0.0], 3, "positive"),
        (np.r_[np.ones(9), np.nan], 3, "finite"),
        (np.ones(9), 3, "length"),
    ],
)
def test_draw_rejects(weights: np.ndarray, n: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        harness.draw_respondents(np.arange(10), weights, n, np.random.default_rng(0))


def test_check_disjoint() -> None:
    harness.check_disjoint(np.array([1, 2, 3]), np.array([4, 5]))
    with pytest.raises(ValueError, match="leak"):
        harness.check_disjoint(np.array([1, 2, 3]), np.array([3, 9]))
