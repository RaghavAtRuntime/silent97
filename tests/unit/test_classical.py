"""Tests for the classical estimator and the two naive baselines."""

import numpy as np
import pytest
from scipy.stats import norm

from silent97.estimators import classical, nps
from silent97.types import Estimate

Z90 = norm.ppf(0.95)


def _data(seed: int = 0, n: int = 200, big_n: int = 1000) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 11, size=n)
    raw = rng.dirichlet([1, 1, 1], size=n + big_n)[:, :2]
    return y, raw[:n], raw[n:]


class TestClassical:
    def test_formula(self) -> None:
        y, p_lab, p_unlab = _data()
        est = classical.classical(y, p_lab, p_unlab, alpha=0.1)
        z = nps.indicators(y)
        point = 100 * (z[:, 0].mean() - z[:, 1].mean())
        cov = np.cov(z.T, ddof=0)
        se = np.sqrt(1e4 * (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]) / len(y))
        assert isinstance(est, Estimate)
        assert est.point == pytest.approx(point)
        assert est.se == pytest.approx(se)
        assert est.lower == pytest.approx(point - Z90 * se)
        assert est.upper == pytest.approx(point + Z90 * se)
        assert est.alpha == 0.1
        assert est.n_labeled == len(y)
        assert est.n_unlabeled == 0
        assert est.method == "classical"
        assert est.diagnostics["effective_n"] == pytest.approx(len(y))

    def test_predictions_optional_and_ignored(self) -> None:
        y, p_lab, p_unlab = _data()
        a = classical.classical(y)
        b = classical.classical(y, p_lab, p_unlab)
        assert a.point == b.point and a.lower == b.lower and a.upper == b.upper

    def test_alpha_changes_width(self) -> None:
        y, _, _ = _data()
        w90 = classical.classical(y, alpha=0.1)
        w95 = classical.classical(y, alpha=0.05)
        assert (w95.upper - w95.lower) / (w90.upper - w90.lower) == pytest.approx(
            norm.ppf(0.975) / Z90
        )

    def test_n_equals_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            classical.classical(np.array([9]))

    def test_all_promoters(self) -> None:
        est = classical.classical(np.full(30, 10))
        assert (est.point, est.lower, est.upper, est.se) == (100.0, 100.0, 100.0, 0.0)

    def test_all_detractors(self) -> None:
        est = classical.classical(np.full(30, 0))
        assert (est.point, est.lower, est.upper, est.se) == (-100.0, -100.0, -100.0, 0.0)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            classical.classical(np.array([1.0, np.nan, 9.0]))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), 2])
    def test_bad_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            classical.classical(np.array([1, 9, 10]), alpha=alpha)

    def test_mismatched_predictions(self) -> None:
        y, p_lab, p_unlab = _data()
        with pytest.raises(ValueError, match="p_lab"):
            classical.classical(y, p_lab[:-1], p_unlab)


class TestNaiveRespondent:
    def test_same_numbers_as_classical(self) -> None:
        y, p_lab, p_unlab = _data(3)
        a = classical.classical(y, p_lab, p_unlab)
        b = classical.naive_respondent(y, p_lab, p_unlab)
        assert (a.point, a.lower, a.upper, a.se) == (b.point, b.lower, b.upper, b.se)
        assert b.method == "naive_respondent"


class TestNaiveModel:
    def test_formula(self) -> None:
        y, p_lab, p_unlab = _data(4)
        est = classical.naive_model(y, p_lab, p_unlab)
        allp = np.vstack([p_lab, p_unlab])
        h = 100 * (allp[:, 0] - allp[:, 1])
        assert est.point == pytest.approx(h.mean())
        assert est.se == pytest.approx(h.std() / np.sqrt(len(h)))
        assert est.n_unlabeled == len(p_unlab)
        assert est.method == "naive_model"

    def test_ignores_outcomes(self) -> None:
        y, p_lab, p_unlab = _data(5)
        a = classical.naive_model(y, p_lab, p_unlab)
        b = classical.naive_model(10 - y, p_lab, p_unlab)
        assert a.point == b.point and a.se == b.se

    def test_requires_predictions(self) -> None:
        y, p_lab, _ = _data()
        with pytest.raises(ValueError, match="p_unlab"):
            classical.naive_model(y, p_lab, np.zeros((0, 2)))
