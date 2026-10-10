"""Tests for the NPS indicator helpers and delta-method variance."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from silent97.estimators import nps

scores_strategy = arrays(
    dtype=np.int64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.integers(min_value=0, max_value=10),
)


# --------------------------------------------------------------------------- scores


class TestValidateScores:
    def test_accepts_ints_and_lists(self) -> None:
        out = nps.validate_scores([0, 5, 10])
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, [0, 5, 10])

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint8, np.int64])
    def test_accepts_integer_dtypes(self, dtype: type) -> None:
        out = nps.validate_scores(np.array([1, 9, 10], dtype=dtype))
        assert out.dtype == np.int64

    def test_accepts_integral_floats(self) -> None:
        np.testing.assert_array_equal(nps.validate_scores(np.array([0.0, 9.0])), [0, 9])

    def test_returns_copy(self) -> None:
        y = np.array([1, 2, 3])
        out = nps.validate_scores(y)
        out[0] = 7
        assert y[0] == 1

    @pytest.mark.parametrize(
        "bad, match",
        [
            (np.array([1.0, np.nan]), "NaN"),
            (np.array([1.0, np.inf]), "finite"),
            (np.array([9.5, 3.0]), "integer"),
            (np.array([-1, 3]), r"0\.\.10"),
            (np.array([11, 3]), r"0\.\.10"),
            (np.array([[1, 2], [3, 4]]), "1-D"),
            (np.array([], dtype=np.int64), "empty"),
            (np.array([True, False]), "bool"),
            (np.array(["9", "10"]), "numeric"),
        ],
    )
    def test_rejects_bad_input(self, bad: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            nps.validate_scores(bad)

    def test_min_n(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            nps.validate_scores([9], min_n=2)


class TestIndicators:
    def test_thresholds(self) -> None:
        z = nps.indicators([0, 6, 7, 8, 9, 10])
        expected = np.array([[0, 1], [0, 1], [0, 0], [0, 0], [1, 0], [1, 0]], dtype=float)
        np.testing.assert_array_equal(z, expected)
        assert z.dtype == np.float64
        assert z.shape == (6, 2)

    def test_every_score(self) -> None:
        z = nps.indicators(np.arange(11))
        np.testing.assert_array_equal(z[:, 0], np.arange(11) >= 9)
        np.testing.assert_array_equal(z[:, 1], np.arange(11) <= 6)
        # No score is both a promoter and a detractor.
        assert np.all(z.sum(axis=1) <= 1)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            nps.indicators(np.array([np.nan, 9.0]))

    def test_nps_values(self) -> None:
        np.testing.assert_array_equal(
            nps.nps_values([0, 6, 7, 8, 9, 10]), [-100, -100, 0, 0, 100, 100]
        )


# --------------------------------------------------------------------------- predictions


class TestValidatePredictions:
    def test_accepts_exact_zero_and_one(self) -> None:
        p = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.3, 0.7]])
        out = nps.validate_predictions(p)
        np.testing.assert_array_equal(out, p)
        assert out.dtype == np.float64

    def test_returns_copy(self) -> None:
        p = np.array([[0.5, 0.2]])
        out = nps.validate_predictions(p)
        out[0, 0] = 0.0
        assert p[0, 0] == 0.5

    @pytest.mark.parametrize(
        "bad, match",
        [
            (np.array([[0.5, np.nan]]), "NaN"),
            (np.array([[0.5, np.inf]]), "finite"),
            (np.array([[-0.1, 0.5]]), r"\[0, 1\]"),
            (np.array([[1.1, 0.0]]), r"\[0, 1\]"),
            (np.array([[0.6, 0.5]]), "sum"),
            (np.array([0.5, 0.2]), r"\(n, 2\)"),
            (np.array([[0.2, 0.3, 0.5]]), r"\(n, 2\)"),
            (np.zeros((0, 2)), "empty"),
        ],
    )
    def test_rejects_bad_input(self, bad: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            nps.validate_predictions(bad)

    def test_min_n(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            nps.validate_predictions(np.array([[0.5, 0.2]]), min_n=2)

    def test_tolerates_rounding_in_row_sum(self) -> None:
        nps.validate_predictions(np.array([[0.7, 0.3 + 1e-12]]))


# --------------------------------------------------------------------------- point estimates


class TestNpsFromCountsAndShares:
    def test_counts(self) -> None:
        assert nps.nps_from_counts(50, 30, 20) == pytest.approx(30.0)

    def test_all_promoters(self) -> None:
        assert nps.nps_from_counts(7, 0, 0) == 100.0

    def test_all_detractors(self) -> None:
        assert nps.nps_from_counts(0, 0, 7) == -100.0

    @pytest.mark.parametrize(
        "counts, match",
        [
            ((0, 0, 0), "positive"),
            ((-1, 2, 3), "non-negative"),
            ((1.5, 2, 3), "integer"),
            ((True, 2, 3), "integer"),
        ],
    )
    def test_counts_reject(self, counts: tuple[object, object, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            nps.nps_from_counts(*counts)  # type: ignore[arg-type]

    def test_shares(self) -> None:
        assert nps.nps_from_shares(0.5, 0.2) == pytest.approx(30.0)

    def test_shares_allow_out_of_range_estimates(self) -> None:
        # Rectified (PPI) share estimates can leave [0, 1]; the contrast is still defined.
        assert nps.nps_from_shares(-0.01, 0.02) == pytest.approx(-3.0)

    def test_shares_reject_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            nps.nps_from_shares(float("nan"), 0.2)

    @given(scores_strategy)
    def test_counts_shares_and_values_agree(self, y: np.ndarray) -> None:
        z = nps.indicators(y)
        n_prom = int((y >= 9).sum())
        n_det = int((y <= 6).sum())
        from_counts = nps.nps_from_counts(n_prom, len(y) - n_prom - n_det, n_det)
        from_shares = nps.nps_from_shares(z[:, 0].mean(), z[:, 1].mean())
        from_values = float(nps.nps_values(y).mean())
        assert from_counts == pytest.approx(from_shares, abs=1e-9)
        assert from_counts == pytest.approx(from_values, abs=1e-9)


# --------------------------------------------------------------------------- variance


class TestGradient:
    def test_gradient_value(self) -> None:
        np.testing.assert_array_equal(nps.NPS_GRADIENT, [100.0, -100.0])

    def test_gradient_read_only(self) -> None:
        with pytest.raises(ValueError):
            nps.NPS_GRADIENT[0] = 1.0


class TestIndicatorMeanCov:
    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 11, size=300)
        z = nps.indicators(y)
        for ddof in (0, 1):
            mean, cov = nps.indicator_mean_cov(z, ddof=ddof)
            np.testing.assert_allclose(mean, z.mean(axis=0))
            np.testing.assert_allclose(cov, np.cov(z.T, ddof=ddof))
            assert cov.shape == (2, 2)

    def test_default_ddof_is_zero(self) -> None:
        z = nps.indicators([0, 10, 8, 9])
        _, cov = nps.indicator_mean_cov(z)
        np.testing.assert_allclose(cov, np.cov(z.T, ddof=0))

    def test_n_equals_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            nps.indicator_mean_cov(nps.indicators([9]))

    @pytest.mark.parametrize("ddof", [-1, 2, 0.5])
    def test_bad_ddof(self, ddof: object) -> None:
        with pytest.raises(ValueError, match="ddof"):
            nps.indicator_mean_cov(nps.indicators([9, 3, 7]), ddof=ddof)  # type: ignore[arg-type]

    def test_rejects_bad_shape_and_nan(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            nps.indicator_mean_cov(np.zeros((5, 3)))
        with pytest.raises(ValueError, match="finite"):
            nps.indicator_mean_cov(np.array([[0.0, np.nan], [1.0, 0.0]]))

    @given(scores_strategy)
    def test_permutation_invariant(self, y: np.ndarray) -> None:
        z = nps.indicators(y)
        perm = np.random.default_rng(len(y)).permutation(len(y))
        m1, c1 = nps.indicator_mean_cov(z)
        m2, c2 = nps.indicator_mean_cov(z[perm])
        np.testing.assert_allclose(m1, m2, atol=1e-12)
        np.testing.assert_allclose(c1, c2, atol=1e-12)


class TestNpsVariance:
    @settings(max_examples=200)
    @given(scores_strategy, st.sampled_from([0, 1]))
    def test_two_routes_agree(self, y: np.ndarray, ddof: int) -> None:
        """Joint-covariance delta method equals the variance of the -100/0/100 variable."""
        _, cov = nps.indicator_mean_cov(nps.indicators(y), ddof=ddof)
        via_gradient = nps.nps_variance(cov)
        via_values = float(np.var(nps.nps_values(y), ddof=ddof))
        assert via_gradient == pytest.approx(via_values, rel=1e-9, abs=1e-6)

    def test_all_promoters_zero_variance(self) -> None:
        _, cov = nps.indicator_mean_cov(nps.indicators([9, 10, 10, 9]))
        assert nps.nps_variance(cov) == 0.0

    def test_all_detractors_zero_variance(self) -> None:
        _, cov = nps.indicator_mean_cov(nps.indicators([0, 3, 6, 6]))
        assert nps.nps_variance(cov) == 0.0

    def test_probe_independence_mistake(self) -> None:
        """Half 0s, half 10s: values are +/-100 with mean 0, so Var = 100^2 exactly.

        Treating promoter and detractor shares as independent drops the -2 Cov term
        and gives 100^2 * (0.25 + 0.25) = 5000 instead of 10000. This test fails if the
        cross term is missing or has the wrong sign.
        """
        y = np.array([0, 10] * 50)
        _, cov = nps.indicator_mean_cov(nps.indicators(y), ddof=0)
        assert cov[0, 1] == pytest.approx(-0.25)
        assert nps.nps_variance(cov) == pytest.approx(10_000.0)
        independent = 100.0**2 * (cov[0, 0] + cov[1, 1])
        assert independent == pytest.approx(5_000.0)

    def test_hand_computed(self) -> None:
        cov = np.array([[0.24, -0.08], [-0.08, 0.16]])
        # 1e4 * (0.24 + 0.16 + 2 * 0.08) = 5600
        assert nps.nps_variance(cov) == pytest.approx(5_600.0)

    @pytest.mark.parametrize(
        "bad, match",
        [
            (np.eye(3), r"\(2, 2\)"),
            (np.array([[1.0, 0.1], [0.2, 1.0]]), "symmetric"),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
        ],
    )
    def test_rejects_bad_cov(self, bad: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            nps.nps_variance(bad)


class TestDeltaMethodSe:
    def test_value(self) -> None:
        cov = np.array([[0.24, -0.08], [-0.08, 0.16]])
        assert nps.delta_method_se(cov, 100) == pytest.approx(np.sqrt(5_600.0 / 100))

    @pytest.mark.parametrize("n", [0, -3, 2.5, True])
    def test_bad_n(self, n: object) -> None:
        with pytest.raises(ValueError, match="n"):
            nps.delta_method_se(np.eye(2) * 0.1, n)  # type: ignore[arg-type]
