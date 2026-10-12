"""Planning tool: estimate rho from past surveys and turn it into survey sizes."""

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

from silent97.eval import harness, metrics, planning


def _data(n: int, rho_signal: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores 0..10 and predictions whose quality grows with ``rho_signal``."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = np.clip(np.round(7 + 2.2 * latent), 0, 10).astype(np.int64)
    noisy = rho_signal * latent + np.sqrt(1 - rho_signal**2) * rng.normal(size=n)
    p_prom = 1 / (1 + np.exp(-(1.5 * noisy - 0.3)))
    p_det = (1 - p_prom) * 1 / (1 + np.exp(1.5 * noisy + 1.0))
    return y, np.column_stack([p_prom, p_det])


def test_rho_matches_the_harness_definition() -> None:
    y, p = _data(500, 0.7)
    est = planning.estimate_rho(y, p, rng=np.random.default_rng(1))
    assert est.rho == pytest.approx(harness.model_correlation(p, y))
    assert est.lo < est.rho < est.hi
    assert est.n == 500


def test_rho_interval_is_seeded_and_covers_under_resampling() -> None:
    y, p = _data(400, 0.6)
    a = planning.estimate_rho(y, p, rng=np.random.default_rng(5))
    b = planning.estimate_rho(y, p, rng=np.random.default_rng(5))
    assert (a.lo, a.hi) == (b.lo, b.hi)
    # The population rho (from a large sample) lies inside most small-sample intervals.
    y_big, p_big = _data(200_000, 0.6, seed=99)
    truth = harness.model_correlation(p_big, y_big)
    hits = 0
    for seed in range(40):
        ys, ps = _data(400, 0.6, seed=seed + 1)
        e = planning.estimate_rho(ys, ps, rng=np.random.default_rng(seed), n_boot=500)
        hits += e.lo <= truth <= e.hi
    assert hits >= 34  # nominal 95%: 38 of 40 expected


def test_estimate_rho_validates_inputs() -> None:
    y, p = _data(50, 0.5)
    with pytest.raises(ValueError, match="at least 20"):
        planning.estimate_rho(y[:10], p[:10], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="shape"):
        planning.estimate_rho(y, p[:, :1], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="constant"):
        planning.estimate_rho(np.full(50, 10), p, rng=np.random.default_rng(0))


def test_gain_is_the_e2_theory() -> None:
    assert planning.gain(0.0, 100, 10_000) == 1.0
    for rho in (0.3, 0.6, 0.9):
        assert planning.gain(rho, 200, 50_000) == metrics.ppi_pp_theory_ratio(rho, 200, 50_000)
    assert planning.gain(0.8, 100, math.inf) == pytest.approx(1 / (1 - 0.64))


def test_required_n_hits_the_target_margin() -> None:
    sd, h, big_n = 80.0, 5.0, 100_000
    z = stats.norm.ppf(0.95)
    n_classical = planning.required_n(h, sd=sd, rho=0.0, big_n=big_n)
    assert n_classical == math.ceil((z * sd / h) ** 2)
    for rho in (0.5, 0.7, 0.9):
        n = planning.required_n(h, sd=sd, rho=rho, big_n=big_n)
        assert n < n_classical

        def half(m: int, rho: float = rho) -> float:
            return z * sd * math.sqrt((1 - rho**2 / (1 + m / big_n)) / m)

        assert half(n) <= h < half(n - 1)


def test_planning_table_rows() -> None:
    t = planning.planning_table(
        sd=80.0, rhos=(0.0, 0.5, 0.8), margins=(10.0, 5.0), big_n=200_000, alpha=0.1
    )
    assert isinstance(t, pl.DataFrame)
    assert t.height == 3 * 2
    row = t.filter((pl.col("rho") == 0.0) & (pl.col("margin") == 5.0)).row(0, named=True)
    assert row["n_ppi_pp"] == row["n_classical"]
    best = t.filter((pl.col("rho") == 0.8) & (pl.col("margin") == 5.0)).row(0, named=True)
    assert best["n_ppi_pp"] < best["n_classical"] * 0.4
    assert best["responses_saved"] == best["n_classical"] - best["n_ppi_pp"]


def test_report_on_past_survey() -> None:
    y, p = _data(300, 0.75)
    rep = planning.report(y, p, big_n=500_000, margins=(5.0,), rng=np.random.default_rng(2))
    assert rep.rho.n == 300
    assert rep.sd == pytest.approx(np.std(100.0 * ((y >= 9) * 1.0 - (y <= 6) * 1.0), ddof=1))
    t = rep.table
    assert set(t["rho_case"]) == {"low", "estimate", "high"}
    low, mid = (t.filter(pl.col("rho_case") == c).row(0, named=True) for c in ("low", "estimate"))
    assert low["n_ppi_pp"] >= mid["n_ppi_pp"]
    assert "rho" in planning.format_report(rep)


def test_compare_with_results() -> None:
    summary = pl.DataFrame(
        {
            "scenario": ["emb", "emb", "emb", "emb"],
            "estimator": ["classical", "ppi_pp", "classical", "ppi_pp"],
            "n_labeled": [100, 100, 1000, 1000],
            "effective_n": [100.0, 150.0, 1000.0, 1400.0],
        }
    )
    t = planning.compare_with_results(summary, rho={"emb": 0.6}, pool=10_000)
    assert t.columns == ["scenario", "rho", "n_labeled", "predicted_gain", "realized_gain"]
    assert t["realized_gain"].to_list() == [1.5, 1.4]
    assert t["predicted_gain"][0] == pytest.approx(planning.gain(0.6, 100, 9_900))
