"""Unit, property and probing tests for ppi, ppi_pp and ppi_crossfit."""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import norm

from silent97 import estimators
from silent97.estimators import classical, nps, ppi
from silent97.types import Estimate

Z90 = norm.ppf(0.95)
G = np.array([100.0, -100.0])


def _data(
    seed: int = 0, n: int = 200, big_n: int = 2000, quality: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores plus predictions that are informative but imperfect."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 11, size=n + big_n)
    z = nps.indicators(y)
    noise = rng.dirichlet([1, 1, 1], size=n + big_n)[:, :2]
    p = quality * z + (1 - quality) * noise
    return y[:n], p[:n], p[n:]


# --------------------------------------------------------------------------- plain PPI


class TestPPI:
    def test_formula(self) -> None:
        y, p_lab, p_unlab = _data()
        est = ppi.ppi(y, p_lab, p_unlab, alpha=0.1)
        z = nps.indicators(y)
        n, big_n = len(y), len(p_unlab)
        theta = p_unlab.mean(axis=0) + (z - p_lab).mean(axis=0)
        sigma = np.cov(p_unlab.T, ddof=0) / big_n + np.cov((z - p_lab).T, ddof=0) / n
        se = np.sqrt(G @ sigma @ G)
        assert isinstance(est, Estimate)
        assert est.point == pytest.approx(G @ theta)
        assert est.se == pytest.approx(se)
        assert est.lower == pytest.approx(G @ theta - Z90 * se)
        assert est.upper == pytest.approx(G @ theta + Z90 * se)
        assert est.n_labeled == n and est.n_unlabeled == big_n
        assert est.method == "ppi"
        assert est.diagnostics["lambda_hat"] == 1.0

    def test_probe_uses_unlabeled_mean_of_predictions(self) -> None:
        """Fails if the first term used mean(p_lab) instead of mean(p_unlab).

        With mean(p_lab) the two prediction terms cancel and PPI collapses to the
        classical point estimate.
        """
        y, p_lab, _ = _data(1)
        p_unlab = np.tile([0.7, 0.1], (500, 1))  # mean very different from p_lab
        est = ppi.ppi(y, p_lab, p_unlab)
        z = nps.indicators(y)
        expected = G @ (p_unlab.mean(axis=0) + (z - p_lab).mean(axis=0))
        wrong = G @ (p_lab.mean(axis=0) + (z - p_lab).mean(axis=0))
        assert est.point == pytest.approx(expected)
        assert abs(est.point - wrong) > 10
        assert wrong == pytest.approx(classical.classical(y).point)

    def test_probe_variance_keeps_cross_covariance(self) -> None:
        """Rectifier promoter/detractor parts are correlated; dropping it changes se."""
        y, p_lab, p_unlab = _data(2, quality=0.2)
        est = ppi.ppi(y, p_lab, p_unlab)
        r = nps.indicators(y) - p_lab
        cr = np.cov(r.T, ddof=0)
        cu = np.cov(p_unlab.T, ddof=0)
        full = (G @ cr @ G) / len(y) + (G @ cu @ G) / len(p_unlab)
        independent = 1e4 * ((cr[0, 0] + cr[1, 1]) / len(y) + (cu[0, 0] + cu[1, 1]) / len(p_unlab))
        assert est.se**2 == pytest.approx(full)
        assert abs(est.se**2 - independent) / full > 0.05

    def test_effective_n(self) -> None:
        y, p_lab, p_unlab = _data(3, quality=0.9)
        est = ppi.ppi(y, p_lab, p_unlab)
        cl = classical.classical(y)
        assert est.diagnostics["effective_n"] == pytest.approx(len(y) * (cl.se / est.se) ** 2)
        assert est.diagnostics["effective_n"] > 2 * len(y)

    def test_n_equals_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ppi.ppi(np.array([9]), np.array([[0.5, 0.2]]), np.array([[0.5, 0.2]] * 3))

    def test_requires_unlabeled(self) -> None:
        y, p_lab, _ = _data()
        with pytest.raises(ValueError, match="p_unlab"):
            ppi.ppi(y, p_lab, np.zeros((0, 2)))

    def test_mismatched_lengths(self) -> None:
        y, p_lab, p_unlab = _data()
        with pytest.raises(ValueError, match="p_lab"):
            ppi.ppi(y, p_lab[:-1], p_unlab)

    def test_rejects_nan(self) -> None:
        y, p_lab, p_unlab = _data()
        p_unlab = p_unlab.copy()
        p_unlab[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ppi.ppi(y, p_lab, p_unlab)

    def test_exact_zero_one_predictions(self) -> None:
        y = np.array([10, 0, 8, 9, 3])
        p = nps.indicators(y)  # entries exactly 0 or 1
        est = ppi.ppi(y, p, p)
        assert np.isfinite(est.point) and est.lower <= est.point <= est.upper

    def test_all_promoters(self) -> None:
        y = np.full(40, 10)
        _, p_lab, p_unlab = _data(4, n=40)
        est = ppi.ppi(y, p_lab, p_unlab)
        assert np.isfinite(est.se) and est.lower <= est.point <= est.upper

    @pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
    def test_bad_alpha(self, alpha: float) -> None:
        y, p_lab, p_unlab = _data()
        with pytest.raises(ValueError, match="alpha"):
            ppi.ppi(y, p_lab, p_unlab, alpha=alpha)


# --------------------------------------------------------------------------- PPI++


def _lambda_by_hand(y: np.ndarray, p_lab: np.ndarray, p_unlab: np.ndarray) -> float:
    """Scalar PPI++ lambda for the NPS contrast, computed independently."""
    v = nps.nps_values(y)
    h_lab, h_unlab = p_lab @ G, p_unlab @ G
    n, big_n = len(v), len(h_unlab)
    cov_vh = np.mean((v - v.mean()) * (h_lab - h_lab.mean()))
    var_h = np.var(np.concatenate([h_lab, h_unlab]), ddof=1)
    return float(cov_vh / ((1 + n / big_n) * var_h))


class TestPPIPlusPlus:
    def test_lambda_formula(self) -> None:
        y, p_lab, p_unlab = _data(5, quality=0.4)
        est = ppi.ppi_pp(y, p_lab, p_unlab)
        assert est.diagnostics["lambda_hat"] == pytest.approx(_lambda_by_hand(y, p_lab, p_unlab))
        assert est.method == "ppi_pp"

    def test_equals_ppi_core_at_lambda(self) -> None:
        y, p_lab, p_unlab = _data(6, quality=0.4)
        est = ppi.ppi_pp(y, p_lab, p_unlab)
        lam = est.diagnostics["lambda_hat"]
        z = nps.indicators(y)
        theta = lam * p_unlab.mean(axis=0) + (z - lam * p_lab).mean(axis=0)
        sigma = np.cov((lam * p_unlab).T, ddof=0) / len(p_unlab) + np.cov(
            (z - lam * p_lab).T, ddof=0
        ) / len(y)
        assert est.point == pytest.approx(G @ theta)
        assert est.se == pytest.approx(np.sqrt(G @ sigma @ G))

    def test_lambda_can_exceed_one_without_clipping(self) -> None:
        """Under-confident (shrunk) predictions make the optimal lambda > 1."""
        rng = np.random.default_rng(7)
        y = rng.integers(0, 11, size=3000)
        p = 0.3 * nps.indicators(y) + 0.2
        est = ppi.ppi_pp(y[:300], p[:300], p[300:])
        assert est.diagnostics["lambda_hat"] > 1.5
        assert est.diagnostics["lambda_clipped"] is False
        clipped = ppi.ppi_pp(y[:300], p[:300], p[300:], clip=True)
        assert clipped.diagnostics["lambda_hat"] == 1.0
        assert clipped.diagnostics["lambda_clipped"] is True
        # The unclipped optimum is narrower (it minimizes the estimated variance).
        assert est.se < clipped.se

    def test_constant_predictions_reduce_to_classical(self) -> None:
        y, _, _ = _data(8)
        p_lab = np.tile([0.4, 0.3], (len(y), 1))
        p_unlab = np.tile([0.4, 0.3], (900, 1))
        est = ppi.ppi_pp(y, p_lab, p_unlab)
        cl = classical.classical(y)
        assert est.diagnostics["lambda_hat"] == 0.0
        assert est.diagnostics["lambda_degenerate"] is True
        assert est.point == pytest.approx(cl.point, abs=1e-9)
        assert est.lower == pytest.approx(cl.lower, abs=1e-9)
        assert est.upper == pytest.approx(cl.upper, abs=1e-9)

    def test_never_much_wider_than_classical(self) -> None:
        """Useless predictions: PPI++ falls back toward classical; PPI does not."""
        rng = np.random.default_rng(9)
        y = rng.integers(0, 11, size=200)
        noise = rng.dirichlet([1, 1, 1], size=5200)[:, :2]
        est = ppi.ppi_pp(y, noise[:200], noise[200:])
        plain = ppi.ppi(y, noise[:200], noise[200:])
        cl = classical.classical(y)
        assert est.se < 1.05 * cl.se
        assert plain.se > est.se


# --------------------------------------------------------------------------- cross-fit


def _crossfit_inputs(
    seed: int = 0, n: int = 120, big_n: int = 800, k: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, p_lab, p_unlab = _data(seed, n=n, big_n=big_n)
    jitter = rng.uniform(0.9, 1.0, size=(big_n, k, 1))
    return y, p_lab, p_unlab[:, None, :] * jitter


class TestCrossFit:
    def test_equals_ppi_on_fold_averaged_predictions(self) -> None:
        y, p_lab, p_unlab = _crossfit_inputs()
        est = ppi.ppi_crossfit(y, p_lab, p_unlab)
        ref = ppi.ppi(y, p_lab, p_unlab.mean(axis=1))
        assert est.point == pytest.approx(ref.point)
        assert est.lower == pytest.approx(ref.lower)
        assert est.upper == pytest.approx(ref.upper)
        assert est.method == "ppi_crossfit"
        assert est.diagnostics["n_folds"] == 5
        assert est.n_unlabeled == p_unlab.shape[0]

    @pytest.mark.parametrize("shape", [(800, 2), (800, 1, 2), (800, 5, 3)])
    def test_rejects_bad_unlabeled_shape(self, shape: tuple[int, ...]) -> None:
        y, p_lab, _ = _crossfit_inputs()
        with pytest.raises(ValueError, match="p_unlab"):
            ppi.ppi_crossfit(y, p_lab, np.full(shape, 0.3))

    def test_rejects_invalid_fold_predictions(self) -> None:
        y, p_lab, p_unlab = _crossfit_inputs()
        p_unlab = p_unlab.copy()
        p_unlab[0, 2] = [0.8, 0.8]
        with pytest.raises(ValueError, match="sum"):
            ppi.ppi_crossfit(y, p_lab, p_unlab)


def _recording_fit() -> tuple[Callable[..., Callable[[np.ndarray], np.ndarray]], list[set[int]]]:
    """A fit function whose predictor encodes which rows it was trained on."""
    seen: list[set[int]] = []

    def fit(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        rows = {int(r) for r in x[:, 0]}
        seen.append(rows)
        share = float(np.mean(y >= 9))

        def predict(x_new: np.ndarray) -> np.ndarray:
            leaked = np.array([int(r) in rows for r in x_new[:, 0]], dtype=float)
            # Promoter prob = training share; detractor prob flags any training row.
            return np.column_stack([np.full(len(x_new), share * 0.5), leaked * 0.5])

        return predict

    return fit, seen


class TestCrossFitHelper:
    def test_no_row_predicted_by_a_model_trained_on_it(self) -> None:
        """Leakage probe: every labeled prediction comes from a model blind to that row."""
        n, big_n = 53, 40
        x_lab = np.arange(n, dtype=float)[:, None]
        x_unlab = np.arange(1000, 1000 + big_n, dtype=float)[:, None]
        y = np.random.default_rng(0).integers(0, 11, size=n)
        fit, seen = _recording_fit()
        out = ppi.cross_fit(x_lab, y, x_unlab, fit=fit, n_folds=5, rng=np.random.default_rng(1))
        assert out.p_lab.shape == (n, 2)
        assert out.p_unlab is not None and out.p_unlab.shape == (big_n, 5, 2)
        assert len(out.predictors) == 5
        for k, predict in enumerate(out.predictors):
            np.testing.assert_array_equal(out.p_unlab[:, k, :], predict(x_unlab))
        fit2, _ = _recording_fit()
        skipped = ppi.cross_fit(x_lab, y, None, fit=fit2, n_folds=5, rng=np.random.default_rng(1))
        assert skipped.p_unlab is None
        np.testing.assert_array_equal(skipped.p_lab, out.p_lab)
        assert np.all(out.p_lab[:, 1] == 0.0), "a labeled row was predicted by its own model"
        assert len(seen) == 5
        for k, rows in enumerate(seen):
            held_out = set(np.flatnonzero(out.folds == k).tolist())
            assert rows.isdisjoint(held_out)
            assert rows | held_out == set(range(n))

    def test_folds_balanced_and_deterministic(self) -> None:
        x = np.random.default_rng(0).normal(size=(53, 2))
        y = np.random.default_rng(1).integers(0, 11, size=53)
        fit, _ = _recording_fit()
        a = ppi.cross_fit(x, y, x[:5], fit=fit, n_folds=10, rng=np.random.default_rng(3))
        b = ppi.cross_fit(x, y, x[:5], fit=fit, n_folds=10, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a.folds, b.folds)
        sizes = np.bincount(a.folds)
        assert sizes.max() - sizes.min() <= 1 and len(sizes) == 10

    @pytest.mark.parametrize("k", [1, 54])
    def test_bad_fold_count(self, k: int) -> None:
        x = np.zeros((53, 1))
        fit, _ = _recording_fit()
        with pytest.raises(ValueError, match="n_folds"):
            ppi.cross_fit(
                x, np.zeros(53, dtype=int), x, fit=fit, n_folds=k, rng=np.random.default_rng(0)
            )


# --------------------------------------------------------------------------- properties

scores = arrays(np.int64, st.integers(5, 60), elements=st.integers(0, 10))


@settings(max_examples=150, deadline=None)
@given(scores, st.integers(1, 5))
def test_perfect_predictions_never_wider_than_classical(y: np.ndarray, k: int) -> None:
    """p equals the true indicator, and the unlabeled set has the same distribution."""
    z = nps.indicators(y)
    p_unlab = np.tile(z, (k, 1))
    cl = classical.classical(y)
    for est in (ppi.ppi(y, z, p_unlab), ppi.ppi_pp(y, z, p_unlab)):
        assert est.upper - est.lower <= cl.upper - cl.lower + 1e-9


@settings(max_examples=150, deadline=None)
@given(
    scores,
    st.floats(0, 1),
    st.floats(0, 1),
    st.integers(2, 300),
)
def test_constant_predictions_ppi_pp_is_classical(
    y: np.ndarray, a: float, b: float, big_n: int
) -> None:
    c = np.array([a, (1 - a) * b])
    est = ppi.ppi_pp(y, np.tile(c, (len(y), 1)), np.tile(c, (big_n, 1)))
    cl = classical.classical(y)
    assert est.point == pytest.approx(cl.point, abs=1e-9)
    assert est.lower == pytest.approx(cl.lower, abs=1e-9)
    assert est.upper == pytest.approx(cl.upper, abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000))
def test_permutation_invariance(seed: int) -> None:
    y, p_lab, p_unlab = _data(seed, n=60, big_n=150)
    rng = np.random.default_rng(seed + 1)
    pl, pu = rng.permutation(len(y)), rng.permutation(len(p_unlab))
    pairs = [
        (classical.classical(y), classical.classical(y[pl])),
        (
            classical.naive_model(y, p_lab, p_unlab),
            classical.naive_model(y[pl], p_lab[pl], p_unlab[pu]),
        ),
        (ppi.ppi(y, p_lab, p_unlab), ppi.ppi(y[pl], p_lab[pl], p_unlab[pu])),
        (ppi.ppi_pp(y, p_lab, p_unlab), ppi.ppi_pp(y[pl], p_lab[pl], p_unlab[pu])),
    ]
    folds = np.stack([p_unlab, 0.95 * p_unlab], axis=1)
    pairs.append((ppi.ppi_crossfit(y, p_lab, folds), ppi.ppi_crossfit(y[pl], p_lab[pl], folds[pu])))
    for a, b in pairs:
        assert a.point == pytest.approx(b.point, abs=1e-9)
        assert a.lower == pytest.approx(b.lower, abs=1e-9)
        assert a.upper == pytest.approx(b.upper, abs=1e-9)


def test_registry() -> None:
    assert set(estimators.REGISTRY) == {
        "classical",
        "naive_model",
        "ppi",
        "ppi_pp",
        "ppi_pp_cf",
        "ppi_crossfit",
    }
    assert (
        frozenset({"classical", "ppi", "ppi_pp", "ppi_pp_cf", "ppi_crossfit"})
        == estimators.VALID_UNDER_MCAR
    )


class TestCoreValidation:
    @pytest.mark.parametrize(
        "y, f, u, match",
        [
            (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((3, 2)), "shape"),
            (np.array([1.0, np.nan]), np.zeros(2), np.zeros(3), "finite"),
            (np.zeros(3), np.zeros(4), np.zeros(5), "same shape"),
            (np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((5, 1)), "column"),
            (np.zeros(1), np.zeros(1), np.zeros(5), "at least 2"),
        ],
    )
    def test_rejects(self, y: np.ndarray, f: np.ndarray, u: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ppi.ppi_mean_pointestimate(y, f, u, 1.0)

    def test_bad_contrast(self) -> None:
        with pytest.raises(ValueError, match="contrast"):
            ppi.optimal_lambda(np.zeros((4, 2)), np.ones((4, 2)), np.ones((5, 2)), contrast=[1.0])

    def test_trace_mode_matches_scalar(self) -> None:
        y, p_lab, p_unlab = _data(11)
        z = nps.indicators(y)
        lam, _, _ = ppi.optimal_lambda(z[:, :1], p_lab[:, :1], p_unlab[:, :1])
        lam_c, _, _ = ppi.optimal_lambda(z[:, :1], p_lab[:, :1], p_unlab[:, :1], contrast=[1.0])
        assert lam == pytest.approx(lam_c)

    def test_cross_fit_feature_shapes(self) -> None:
        fit, _ = _recording_fit()
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="x_lab"):
            ppi.cross_fit(
                np.zeros((9, 1)), y, None, fit=fit, n_folds=2, rng=np.random.default_rng(0)
            )
        with pytest.raises(ValueError, match="x_unlab"):
            ppi.cross_fit(
                np.zeros((10, 1)),
                y,
                np.zeros((5, 2)),
                fit=fit,
                n_folds=2,
                rng=np.random.default_rng(0),
            )
