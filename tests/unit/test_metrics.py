"""Tests for evaluation metrics."""

import numpy as np
import polars as pl
import pytest

from silent97.eval import metrics


def test_wilson_known_value() -> None:
    lo, hi = metrics.wilson_interval(90, 100)
    assert lo == pytest.approx(0.8256, abs=1e-4)
    assert hi == pytest.approx(0.9448, abs=1e-4)


@pytest.mark.parametrize("k", [0, 1000])
def test_wilson_edges(k: int) -> None:
    lo, hi = metrics.wilson_interval(k, 1000)
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_rejects() -> None:
    with pytest.raises(ValueError):
        metrics.wilson_interval(5, 0)
    with pytest.raises(ValueError):
        metrics.wilson_interval(6, 5)


def _fake() -> pl.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for est, width, bias in (("classical", 20.0, 0.0), ("ppi_pp", 10.0, 1.0)):
        for r in range(200):
            point = 20.0 + bias + rng.normal(scale=3)
            rows.append(
                {
                    "scenario": "mcar",
                    "estimator": est,
                    "n_labeled": 100,
                    "replicate": r,
                    "truth": 20.0,
                    "point": point,
                    "lower": point - width / 2,
                    "upper": point + width / 2,
                    "se": width / 3.29,
                    "width": width,
                    "covered": abs(point - 20.0) <= width / 2,
                    "lambda_hat": float("nan") if est == "classical" else 0.7,
                }
            )
    return pl.DataFrame(rows)


def test_summarize() -> None:
    df = _fake()
    s = metrics.summarize(df).sort("estimator")
    cl = s.filter(pl.col("estimator") == "classical").row(0, named=True)
    pp = s.filter(pl.col("estimator") == "ppi_pp").row(0, named=True)
    sub = df.filter(pl.col("estimator") == "ppi_pp")
    assert pp["replicates"] == 200
    assert pp["coverage"] == pytest.approx(sub["covered"].mean())
    assert pp["bias"] == pytest.approx((sub["point"] - 20.0).mean())
    assert pp["rmse"] == pytest.approx(np.sqrt(((sub["point"] - 20.0) ** 2).mean()))
    assert pp["mean_width"] == pytest.approx(10.0)
    assert pp["effective_n"] == pytest.approx(100 * (20.0 / 10.0) ** 2)
    assert pp["responses_saved"] == pytest.approx(300.0)
    assert cl["effective_n"] == pytest.approx(100.0)
    assert pp["mean_lambda"] == pytest.approx(0.7)
    lo, hi = metrics.wilson_interval(int(sub["covered"].sum()), 200)
    assert (pp["coverage_lo"], pp["coverage_hi"]) == pytest.approx((lo, hi))


@pytest.mark.parametrize(
    "lo, hi, ok",
    [(0.87, 0.89, True), (0.91, 0.93, True), (0.85, 0.879, False), (0.921, 0.95, False)],
)
def test_coverage_ok(lo: float, hi: float, ok: bool) -> None:
    assert metrics.overlaps_target(lo, hi) is ok
