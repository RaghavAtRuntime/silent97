"""Delta-adjustment sensitivity analysis for nonresponse not explained by covariates."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from silent97.estimators import sensitivity
from silent97.types import Estimate


def _base(point: float = 20.0, se: float = 3.0, alpha: float = 0.1) -> Estimate:
    z = 1.6448536269514722
    return Estimate(
        point=point,
        lower=point - z * se,
        upper=point + z * se,
        alpha=alpha,
        se=se,
        n_labeled=500,
        n_unlabeled=9500,
        method="ppi_pp_cf",
        diagnostics={},
    )


def test_zero_delta_returns_the_base_estimate() -> None:
    base = _base()
    curve = sensitivity.delta_adjust(base, [0.0], nonrespondent_share=0.95)
    assert curve.point[0] == base.point
    assert curve.lower[0] == base.lower
    assert curve.upper[0] == base.upper


def test_shift_is_two_delta_times_nonrespondent_share() -> None:
    """+delta pp on promoters and -delta pp on detractors moves the non-respondents' NPS
    by 2 delta points, and the population NPS by 2 delta w_U. (Probing test: forgetting
    w_U, or moving only one share, gives a different answer.)"""
    base = _base()
    deltas = np.array([-5.0, -1.0, 2.5, 10.0])
    curve = sensitivity.delta_adjust(base, deltas, nonrespondent_share=0.5)
    np.testing.assert_allclose(curve.point, base.point + deltas)
    np.testing.assert_allclose(curve.upper - curve.lower, base.upper - base.lower)
    np.testing.assert_allclose(curve.lower, base.lower + deltas)
    np.testing.assert_array_equal(curve.delta, deltas)
    assert curve.nonrespondent_share == 0.5


def test_agrees_with_recomposing_the_two_shares() -> None:
    """The scalar shift equals recomputing NPS from the adjusted 2-D shares of respondents
    and non-respondents (rule 1: both routes agree)."""
    n, big_n = 400, 9600
    p_r, d_r = 0.52, 0.18
    p_u, d_u = 0.41, 0.27
    m = n + big_n
    pop_p, pop_d = (n * p_r + big_n * p_u) / m, (n * d_r + big_n * d_u) / m
    base = _base(point=100 * (pop_p - pop_d))
    implied = sensitivity.implied_nonrespondent_shares(
        (pop_p, pop_d), (p_r, d_r), n_respondents=n, n_nonrespondents=big_n
    )
    np.testing.assert_allclose(implied, (p_u, d_u))
    for delta in (-7.0, 0.0, 3.0):
        p_adj, d_adj = p_u + delta / 100, d_u - delta / 100
        direct = 100 * ((n * p_r + big_n * p_adj) - (n * d_r + big_n * d_adj)) / m
        curve = sensitivity.delta_adjust(base, [delta], nonrespondent_share=big_n / m)
        assert math.isclose(curve.point[0], direct, abs_tol=1e-9)


def test_true_shift_recovers_the_truth() -> None:
    """If non-respondents truly differ from the implied rates by delta, the adjusted
    point equals the population NPS."""
    n, big_n = 300, 2700
    m = n + big_n
    nps_r, implied_u, true_u = 35.0, 18.0, 10.0
    base = _base(point=(n * nps_r + big_n * implied_u) / m)
    truth = (n * nps_r + big_n * true_u) / m
    delta = (true_u - implied_u) / 2
    curve = sensitivity.delta_adjust(base, [delta], nonrespondent_share=big_n / m)
    assert math.isclose(curve.point[0], truth, abs_tol=1e-9)


def test_feasible_range_and_mask() -> None:
    assert sensitivity.feasible_range((0.4, 0.2)) == pytest.approx((-40.0, 20.0))
    # Promoter share cannot exceed 1, detractor share cannot go below 0.
    assert sensitivity.feasible_range((0.9, 0.05)) == pytest.approx((-90.0, 5.0))
    base = _base()
    curve = sensitivity.delta_adjust(
        base, [-50.0, -40.0, 0.0, 20.0, 25.0], nonrespondent_share=0.9,
        nonrespondent_shares=(0.4, 0.2),
    )  # fmt: skip
    assert curve.delta_range == pytest.approx((-40.0, 20.0))
    np.testing.assert_array_equal(curve.feasible, [False, True, True, True, False])


def test_without_shares_everything_is_marked_feasible() -> None:
    curve = sensitivity.delta_adjust(_base(), [-80.0, 80.0], nonrespondent_share=0.9)
    assert curve.delta_range is None
    assert curve.feasible.all()


def test_breakpoint_above() -> None:
    base = _base(point=20.0, se=3.0)
    w = 0.9
    d_star = sensitivity.breakpoint(base, nonrespondent_share=w, threshold=0.0)
    # "NPS > 0" fails when non-respondents are less satisfied: delta* is negative.
    assert d_star < 0
    at = sensitivity.delta_adjust(base, [d_star], nonrespondent_share=w)
    assert math.isclose(at.lower[0], 0.0, abs_tol=1e-9)
    just_inside = sensitivity.delta_adjust(base, [0.99 * d_star], nonrespondent_share=w)
    assert just_inside.lower[0] > 0.0


def test_breakpoint_below() -> None:
    base = _base(point=-15.0, se=2.0)
    d_star = sensitivity.breakpoint(
        base, nonrespondent_share=0.8, threshold=-5.0, direction="below"
    )
    assert d_star > 0
    at = sensitivity.delta_adjust(base, [d_star], nonrespondent_share=0.8)
    assert math.isclose(at.upper[0], -5.0, abs_tol=1e-9)


def test_breakpoint_when_conclusion_does_not_hold() -> None:
    base = _base(point=2.0, se=3.0)  # lower bound below 0
    assert sensitivity.breakpoint(base, nonrespondent_share=0.9, threshold=0.0) == 0.0


def test_breakpoint_when_everyone_responded() -> None:
    base = _base()
    assert sensitivity.breakpoint(base, nonrespondent_share=0.0, threshold=0.0) == -math.inf
    below = sensitivity.breakpoint(base, nonrespondent_share=0.0, threshold=40.0, direction="below")
    assert below == math.inf


@given(
    point=st.floats(-80, 80),
    se=st.floats(0.1, 10),
    w=st.floats(0.01, 1.0),
    threshold=st.floats(-90, 90),
)
def test_breakpoint_is_the_edge(point: float, se: float, w: float, threshold: float) -> None:
    base = _base(point=point, se=se)
    d_star = sensitivity.breakpoint(base, nonrespondent_share=w, threshold=threshold)
    if base.lower <= threshold:
        assert d_star == 0.0
    else:
        at = sensitivity.delta_adjust(base, [d_star], nonrespondent_share=w)
        assert math.isclose(at.lower[0], threshold, abs_tol=1e-7)


def test_input_validation() -> None:
    base = _base()
    with pytest.raises(ValueError, match="nonrespondent_share"):
        sensitivity.delta_adjust(base, [0.0], nonrespondent_share=1.2)
    with pytest.raises(ValueError, match="deltas"):
        sensitivity.delta_adjust(base, [0.0, math.nan], nonrespondent_share=0.5)
    with pytest.raises(ValueError, match="deltas"):
        sensitivity.delta_adjust(base, [], nonrespondent_share=0.5)
    with pytest.raises(ValueError, match="direction"):
        sensitivity.breakpoint(base, nonrespondent_share=0.5, threshold=0.0, direction="up")
    with pytest.raises(ValueError, match="finite"):
        sensitivity.delta_adjust(base._replace(upper=math.inf), [0.0], nonrespondent_share=0.5)
    with pytest.raises(ValueError, match="shares"):
        sensitivity.feasible_range((0.7, 0.4))
    with pytest.raises(ValueError, match="n_nonrespondents"):
        sensitivity.implied_nonrespondent_shares(
            (0.4, 0.2), (0.4, 0.2), n_respondents=10, n_nonrespondents=0
        )
