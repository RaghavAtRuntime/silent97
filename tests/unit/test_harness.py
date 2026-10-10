"""Tests for the experiment harness grid runner."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.eval import harness

SMALL = {
    "name": "test_grid",
    "seed": 11,
    "population": {"n_customers": 6000},
    "labeler": {"n_train": 800},
    "scenarios": [
        {"name": "mcar", "mechanism": "mcar"},
        {"name": "mar_weak", "mechanism": "mar", "signal_strength": 0.3},
    ],
    "estimators": ["classical", "naive_respondent", "naive_model", "ppi", "ppi_pp", "ppi_crossfit"],
    "n_labeled": [40, 80],
    "replicates": 4,
    "alpha": 0.1,
    "crossfit_folds": 4,
    "n_jobs": 1,
}


@pytest.fixture(scope="module")
def results() -> pl.DataFrame:
    return harness.run_grid(harness.GridConfig.from_dict(SMALL))


def test_shape_and_columns(results: pl.DataFrame) -> None:
    assert results.height == 2 * 2 * 4 * 6
    for col in harness.RESULT_COLUMNS:
        assert col in results.columns
    assert set(results["estimator"]) == set(SMALL["estimators"])


def test_rows_consistent(results: pl.DataFrame) -> None:
    covered = (results["lower"] <= results["truth"]) & (results["truth"] <= results["upper"])
    assert (covered == results["covered"]).all()
    np.testing.assert_allclose(results["width"], results["upper"] - results["lower"])
    assert (
        results["n_unlabeled"].filter(results["estimator"] == "ppi")
        == 6000 - 800 - results.filter(results["estimator"] == "ppi")["n_labeled"]
    ).all()


def test_truth_is_pool_nps(results: pl.DataFrame) -> None:
    cfg = harness.GridConfig.from_dict(SMALL)
    prep = harness.prepare_scenario(cfg, 0)
    v = 100.0 * ((prep.scores >= 9).astype(float) - (prep.scores <= 6).astype(float))
    assert prep.truth == pytest.approx(v[prep.pool].mean())
    assert results.filter(pl.col("scenario") == "mcar")["truth"].unique().to_list() == [prep.truth]
    assert np.intersect1d(prep.pool, prep.train_idx).size == 0
    assert prep.pool.size + prep.train_idx.size == 6000


def test_same_estimates_for_shared_draws(results: pl.DataFrame) -> None:
    """classical and naive_respondent see the same labeled draw in a replicate."""
    a = results.filter(pl.col("estimator") == "classical").sort(
        ["scenario", "n_labeled", "replicate"]
    )
    b = results.filter(pl.col("estimator") == "naive_respondent").sort(
        ["scenario", "n_labeled", "replicate"]
    )
    np.testing.assert_array_equal(a["point"], b["point"])


def test_deterministic_and_parallel_invariant(results: pl.DataFrame) -> None:
    again = harness.run_grid(harness.GridConfig.from_dict({**SMALL, "n_jobs": 2}))
    assert again.equals(results)


def test_leak_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a labeled draw ever overlapped the labeler's training set, the run must fail."""
    cfg = harness.GridConfig.from_dict({**SMALL, "estimators": ["ppi"], "n_labeled": [40]})
    prep = harness.prepare_scenario(cfg, 0)

    def leaky(
        pool: np.ndarray, weights: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        return np.sort(prep.train_idx[:n])

    monkeypatch.setattr(harness, "draw_respondents", leaky)
    with pytest.raises(ValueError, match="leak"):
        harness.run_replicate(prep, cfg, 0, 40, 0)


def test_write_results(results: pl.DataFrame, tmp_path: Path) -> None:
    path = harness.write_results(results, tmp_path / "sub" / "r.parquet")
    assert pl.read_parquet(path).equals(results)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"estimators": ["ppi", "magic"]}, "estimator"),
        ({"scenarios": [{"name": "x", "mechanism": "nmar"}]}, "mechanism"),
        ({"n_labeled": [1]}, "n_labeled"),
        ({"replicates": 0}, "replicates"),
        ({"alpha": 1.5}, "alpha"),
        ({"surprise": 1}, "unknown"),
        ({"labeler": {"n_train": 10, "depth": 3}}, "unknown"),
        (
            {"scenarios": [{"name": "a", "mechanism": "mcar"}, {"name": "a", "mechanism": "mar"}]},
            "unique",
        ),
    ],
)
def test_config_validation(change: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        harness.GridConfig.from_dict({**SMALL, **change})


def test_load_yaml(tmp_path: Path) -> None:
    path = tmp_path / "grid.yaml"
    path.write_text(
        "name: g\nseed: 3\nscenarios:\n  - {name: mcar, mechanism: mcar}\n"
        "estimators: [classical]\nn_labeled: [50]\nreplicates: 2\n"
    )
    cfg = harness.load_grid_config(path)
    assert cfg.seed == 3 and cfg.n_labeled == (50,) and cfg.alpha == 0.1


def test_repo_e01_config_loads() -> None:
    cfg = harness.load_grid_config(Path("configs/e01_coverage.yaml"))
    assert cfg.n_labeled == (50, 100, 200, 500, 1000)
    assert cfg.replicates == 1000
