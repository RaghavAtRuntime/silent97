"""Tests for the minimal logistic-regression labeler used from Phase 2."""

import numpy as np
import pyarrow as pa
import pytest

from silent97.data import schema, simulate
from silent97.data.simulate import SimConfig
from silent97.estimators import nps
from silent97.labelers import logistic


@pytest.fixture(scope="module")
def pop() -> simulate.Population:
    return simulate.simulate_population(
        SimConfig(n_customers=20_000, signal_strength=0.8), np.random.default_rng(0)
    )


def _scores(pop: simulate.Population) -> np.ndarray:
    return pop.table.column("score").to_numpy()


def test_score_classes() -> None:
    np.testing.assert_array_equal(
        logistic.score_classes(np.arange(11)), [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    )


def test_design_matrix_shape(pop: simulate.Population) -> None:
    x, names = logistic.design_matrix(pop.table)
    assert x.shape == (20_000, len(names))
    assert x.dtype == np.float64
    assert np.isfinite(x).all()
    assert "signal" in names
    assert not any(n.startswith(c) for n in names for c in schema.GROUND_TRUTH)


def test_design_matrix_ignores_ground_truth(pop: simulate.Population) -> None:
    """Scrambling every outcome column must leave the features unchanged."""
    x, _ = logistic.design_matrix(pop.table)
    table = pop.table
    rng = np.random.default_rng(1)
    for col in schema.GROUND_TRUTH + schema.RESPONSE_COLUMNS:
        i = table.schema.get_field_index(col)
        values = rng.permutation(table.column(col).to_numpy())
        table = table.set_column(i, table.schema.field(i), pa.array(values))
    x2, _ = logistic.design_matrix(table)
    np.testing.assert_array_equal(x, x2)


def test_fit_predict_valid_and_calibrated_in_the_large(pop: simulate.Population) -> None:
    x, _ = logistic.design_matrix(pop.table)
    y = _scores(pop)
    predict = logistic.fit_logistic(x[:5000], y[:5000])
    p = predict(x[5000:])
    nps.validate_predictions(p)
    z = nps.indicators(y[5000:])
    # Out-of-sample mean prediction tracks the true shares (training n = 5000).
    np.testing.assert_allclose(p.mean(axis=0), z.mean(axis=0), atol=0.02)


def test_predictions_informative(pop: simulate.Population) -> None:
    x, _ = logistic.design_matrix(pop.table)
    y = _scores(pop)
    p = logistic.fit_logistic(x[:5000], y[:5000])(x[5000:])
    h = p @ nps.NPS_GRADIENT
    v = nps.nps_values(y[5000:])
    assert np.corrcoef(h, v)[0, 1] > 0.6


def test_missing_class_gets_zero_probability() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3))
    y = np.where(x[:, 0] > 0, 10, 2)  # no passives
    p = logistic.fit_logistic(x, y)(x)
    nps.validate_predictions(p)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)  # passive probability is 0


@pytest.mark.parametrize("score, expected", [(10, [1.0, 0.0]), (3, [0.0, 1.0]), (7, [0.0, 0.0])])
def test_single_class_is_constant(score: int, expected: list[float]) -> None:
    x = np.random.default_rng(3).normal(size=(20, 3))
    p = logistic.fit_logistic(x, np.full(20, score))(x[:5])
    np.testing.assert_array_equal(p, np.tile(expected, (5, 1)))


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="rows"):
        logistic.fit_logistic(np.zeros((5, 2)), np.zeros(4, dtype=int))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_fast_predictor_matches_sklearn(n_classes: int) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(loc=3.0, scale=[1.0, 5.0, 0.2], size=(400, 3))
    y = np.digitize(x[:, 0] + rng.normal(size=400), [2.5, 3.5]) * 4 + 1  # scores 1, 5, 9
    if n_classes == 2:
        y = np.where(y == 5, 1, y)
    predict = logistic.fit_logistic(x, y)
    proba = predict.model.predict_proba(x)  # type: ignore[attr-defined]
    classes = list(predict.model.classes_)  # type: ignore[attr-defined]
    full = np.zeros((400, 3))
    full[:, classes] = proba
    np.testing.assert_allclose(predict(x), full[:, [2, 0]], atol=1e-12)


def test_predict_folds_matches_individual_predictors() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 4))
    y_full = np.digitize(x[:, 0] + rng.normal(size=300), [-0.5, 0.5]) * 4 + 1
    y_two = np.where(y_full == 5, 1, y_full)  # a fold model missing the passive class
    preds = [logistic.fit_logistic(x, y_full), logistic.fit_logistic(x, y_two)]
    stacked = logistic.predict_folds(preds, x)
    assert stacked.shape == (300, 2, 2)
    for k, p in enumerate(preds):
        np.testing.assert_allclose(stacked[:, k, :], p(x), atol=1e-12)
    # A non-linear (constant) predictor falls back to calling each model.
    preds.append(logistic.fit_logistic(x, np.full(300, 10)))
    mixed = logistic.predict_folds(preds, x)
    np.testing.assert_allclose(mixed[:, 2, :], np.tile([1.0, 0.0], (300, 1)))
    np.testing.assert_allclose(mixed[:, :2, :], stacked, atol=1e-12)
