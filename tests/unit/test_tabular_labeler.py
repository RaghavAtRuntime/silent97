"""Tests for the LightGBM + isotonic tabular labeler."""

import numpy as np
import pytest

from silent97.data import simulate
from silent97.data.simulate import SimConfig
from silent97.estimators import nps
from silent97.labelers import logistic
from silent97.labelers.tabular import TabularLabeler


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    pop = simulate.simulate_population(
        SimConfig(n_customers=30_000, signal_strength=0.8), np.random.default_rng(0)
    )
    x, _ = logistic.design_matrix(pop.table)
    return x, pop.table.column("score").to_numpy()


@pytest.fixture(scope="module")
def fitted(data: tuple[np.ndarray, np.ndarray]) -> TabularLabeler:
    x, y = data
    return TabularLabeler(random_state=1).fit(x[:8000], y[:8000])


def test_predictions_valid(fitted: TabularLabeler, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = data
    p = fitted.predict(x[8000:])
    assert p.shape == (22_000, 2)
    nps.validate_predictions(p)
    assert fitted.calibrated_


def _ece(p: np.ndarray, target: np.ndarray, width: float = 0.05) -> float:
    """Expected calibration error over fixed-width probability bins."""
    bins = np.minimum((p / width).astype(int), round(1 / width) - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(p[m].mean() - target[m].mean())
    return total / p.size


def test_calibrated_on_held_out(
    fitted: TabularLabeler, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """Held-out calibration: in the large, and ECE over 0.05-wide bins below 0.02.

    Fixed-width bins (not quantile bins) because isotonic maps produce plateaus. The
    sampling-noise floor of this ECE at 22,000 rows is roughly 0.01.
    """
    x, y = data
    p = fitted.predict(x[8000:])
    z = nps.indicators(y[8000:])
    np.testing.assert_allclose(p.mean(axis=0), z.mean(axis=0), atol=0.015)
    for col in (0, 1):
        assert _ece(p[:, col], z[:, col]) < 0.02, col


def test_informative(fitted: TabularLabeler, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    h = fitted.predict(x[8000:]) @ nps.NPS_GRADIENT
    assert np.corrcoef(h, nps.nps_values(y[8000:]))[0, 1] > 0.6


def test_deterministic(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    a = TabularLabeler(random_state=3, n_estimators=50).fit(x[:3000], y[:3000]).predict(x[-500:])
    b = TabularLabeler(random_state=3, n_estimators=50).fit(x[:3000], y[:3000]).predict(x[-500:])
    np.testing.assert_array_equal(a, b)


def test_two_classes_like_cfpb() -> None:
    """Scores 0 (disputed) and 7 (not disputed): promoter probability is exactly 0."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2000, 3))
    y = np.where(rng.random(2000) < 1 / (1 + np.exp(-2 * x[:, 0])), 0, 7)
    lab = TabularLabeler(random_state=0, n_estimators=50).fit(x, y)
    p = lab.predict(x)
    assert np.all(p[:, 0] == 0.0)
    assert np.corrcoef(p[:, 1], y == 0)[0, 1] > 0.4


def test_single_class_constant() -> None:
    x = np.random.default_rng(5).normal(size=(50, 2))
    p = TabularLabeler().fit(x, np.full(50, 9)).predict(x[:3])
    np.testing.assert_array_equal(p, [[1.0, 0.0]] * 3)


def test_rare_class_skips_calibration_folds() -> None:
    """A class with fewer members than calibration folds still fits (fewer folds)."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(300, 2))
    y = np.where(np.arange(300) < 3, 10, np.where(x[:, 0] > 0, 0, 7))
    lab = TabularLabeler(n_estimators=20).fit(x, y)
    nps.validate_predictions(lab.predict(x))


def test_predict_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TabularLabeler().predict(np.zeros((2, 2)))
