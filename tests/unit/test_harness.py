"""Tests for the experiment harness grid runner."""

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.estimators import classical, nps, ppi
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
    "estimators": ["classical", "naive_model", "ppi", "ppi_pp", "ppi_crossfit"],
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
    assert results.height == 2 * 2 * 4 * 5
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


def test_rows_match_estimators_on_the_replicate_draw(results: pl.DataFrame) -> None:
    """Rebuild one replicate's survey draw independently; every estimator saw it."""
    cfg = harness.GridConfig.from_dict(SMALL)
    prep = harness.prepare_scenario(cfg, 0)
    rng = np.random.default_rng([cfg.seed, 2, 0, 40, 1])
    lab = harness.draw_respondents(prep.pool, prep.propensity[prep.pool], 40, rng)
    unlab = np.setdiff1d(prep.pool, lab)
    row = results.filter(
        (pl.col("scenario") == "mcar") & (pl.col("n_labeled") == 40) & (pl.col("replicate") == 1)
    )
    expected = {
        "classical": classical.classical(prep.scores[lab]),
        "ppi": ppi.ppi(prep.scores[lab], prep.p_all[lab], prep.p_all[unlab]),
    }
    for name, est in expected.items():
        got = row.filter(pl.col("estimator") == name).row(0, named=True)
        assert got["point"] == pytest.approx(est.point, abs=1e-12)
        assert got["upper"] == pytest.approx(est.upper, abs=1e-12)


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


def test_labeler_trained_on_scenario_mechanism() -> None:
    """train_mechanism='scenario': the labeler learns from respondents of that mechanism."""
    uniform = harness.GridConfig.from_dict(SMALL)
    selected = harness.GridConfig.from_dict(
        {**SMALL, "labeler": {"n_train": 800, "train_mechanism": "scenario"}}
    )
    u_mcar, u_mar = harness.prepare_scenario(uniform, 0), harness.prepare_scenario(uniform, 1)
    s_mcar, s_mar = harness.prepare_scenario(selected, 0), harness.prepare_scenario(selected, 1)
    # Uniform training: the same customers whatever the scenario.
    np.testing.assert_array_equal(u_mcar.train_idx, u_mar.train_idx)
    # Scenario training: MCAR scenario unchanged, MAR scenario draws with MAR propensity.
    np.testing.assert_array_equal(s_mcar.train_idx, u_mcar.train_idx)
    assert not np.array_equal(s_mar.train_idx, u_mar.train_idx)
    assert s_mar.propensity[s_mar.train_idx].mean() > u_mar.propensity[u_mar.train_idx].mean()
    assert np.intersect1d(s_mar.pool, s_mar.train_idx).size == 0


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
        ({"labeler": {"train_mechanism": "respondents"}}, "train_mechanism"),
        ({"labeler": {"kind": "forest"}}, "kind"),
        (
            {"scenarios": [{"name": "a", "mechanism": "mcar", "labeler_features": "some"}]},
            "labeler_features",
        ),
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


def test_tabular_labeler_kind() -> None:
    cfg = harness.GridConfig.from_dict(
        {**SMALL, "labeler": {"n_train": 800, "kind": "tabular"}, "estimators": ["ppi"]}
    )
    prep = harness.prepare_scenario(cfg, 0)
    logistic = harness.prepare_scenario(harness.GridConfig.from_dict(SMALL), 0)
    np.testing.assert_array_equal(prep.train_idx, logistic.train_idx)
    assert not np.allclose(prep.p_all, logistic.p_all)
    rows = harness.run_replicate(prep, cfg, 0, 40, 0)
    assert rows[0]["estimator"] == "ppi" and np.isfinite(rows[0]["point"])


def test_pool_subset_defines_unlabeled_set() -> None:
    """A prepared scenario whose pool is a subset (as for CFPB): unlabeled = pool - draw."""
    base = harness.prepare_scenario(harness.GridConfig.from_dict(SMALL), 0)
    pool = base.pool[:2000]
    prep = dataclasses.replace(
        base, pool=pool, truth=float(nps.nps_values(base.scores[pool]).mean())
    )
    cfg = harness.GridConfig.from_dict({**SMALL, "estimators": ["ppi", "ppi_crossfit"]})
    rows = harness.run_replicate(prep, cfg, 0, 40, 0)
    assert [r["n_unlabeled"] for r in rows] == [1960, 1960]


def test_model_corr_recorded(results: pl.DataFrame) -> None:
    cfg = harness.GridConfig.from_dict(SMALL)
    prep = harness.prepare_scenario(cfg, 0)
    h = prep.p_all[prep.pool] @ nps.NPS_GRADIENT
    v = nps.nps_values(prep.scores[prep.pool])
    assert prep.model_corr == pytest.approx(np.corrcoef(h, v)[0, 1])
    got = results.filter(pl.col("scenario") == "mcar")["model_corr"].unique().to_list()
    assert got == [pytest.approx(prep.model_corr)]


def test_signal_only_labeler_has_no_covariate_signal() -> None:
    """With signal strength 0 and only the signal feature, the model knows nothing."""
    scenario = {
        "name": "s0",
        "mechanism": "mcar",
        "signal_strength": 0.0,
        "labeler_features": "signal_only",
    }
    cfg = harness.GridConfig.from_dict(
        {
            **SMALL,
            "population": {"n_customers": 20_000},
            "labeler": {"n_train": 2000},
            "scenarios": [scenario],
        }
    )
    prep = harness.prepare_scenario(cfg, 0)
    assert prep.x.shape[1] == 1
    assert abs(prep.model_corr) < 0.06


def test_ppi_pp_cf_runs_with_its_own_random_stream(results: pl.DataFrame) -> None:
    """Adding ppi_pp_cf (which draws a fold split) must not change any other estimator."""
    with_cf = harness.run_grid(
        harness.GridConfig.from_dict({**SMALL, "estimators": [*SMALL["estimators"], "ppi_pp_cf"]})
    )
    keys = ["scenario", "n_labeled", "replicate", "estimator"]
    others = with_cf.filter(pl.col("estimator") != "ppi_pp_cf").sort(keys)
    assert others.select(results.columns).equals(results.sort(keys))
    cf = with_cf.filter(pl.col("estimator") == "ppi_pp_cf")
    assert cf.height == 2 * 2 * 4
    assert cf["lambda_hat"].is_finite().all()


# --------------------------------------------------------------------------- sampling designs

DESIGNS = {
    **SMALL,
    "name": "design_grid",
    "scenarios": [
        {"name": "uniform", "mechanism": "mcar", "design": "uniform"},
        {"name": "strat", "mechanism": "mcar", "design": "stratified_neyman", "n_strata": 4},
        {"name": "active", "mechanism": "mcar", "design": "active"},
    ],
    "estimators": ["classical", "ppi", "ppi_active", "ppi_stratified"],
    "n_labeled": [60],
    "replicates": 6,
}


@pytest.fixture(scope="module")
def design_results() -> pl.DataFrame:
    return harness.run_grid(harness.GridConfig.from_dict(DESIGNS))


def test_design_grid_rows(design_results: pl.DataFrame) -> None:
    assert design_results.height == 3 * 6 * 4
    assert (design_results["n_labeled"] == 60).all()  # the target budget
    active_n = design_results.filter(pl.col("scenario") == "active")["n_realized"]
    assert active_n.n_unique() > 1  # Poisson sampling: the realized n varies
    fixed = design_results.filter(pl.col("scenario") != "active")["n_realized"]
    assert (fixed == 60).all()


def test_design_draws_match_estimators(design_results: pl.DataFrame) -> None:
    """Recompute one active replicate by hand from the prepared scenario."""
    cfg = harness.GridConfig.from_dict(DESIGNS)
    prep = harness.prepare_scenario(cfg, 2)
    rng = np.random.default_rng([cfg.seed, 2, 2, 60, 0])
    d = harness.draw_design(prep, cfg, 60, rng)
    assert d.name == "active" and d.pi.shape == prep.pool.shape
    lab = prep.pool[d.idx]
    unl = np.setdiff1d(prep.pool, lab)
    from silent97.estimators.active import ppi_active

    est = ppi_active(
        prep.scores[lab], prep.p_all[lab], prep.p_all[unl], pi_lab=d.pi[d.idx], alpha=0.1
    )
    row = design_results.filter(
        (pl.col("scenario") == "active")
        & (pl.col("estimator") == "ppi_active")
        & (pl.col("replicate") == 0)
    ).row(0, named=True)
    assert row["point"] == pytest.approx(est.point, abs=1e-10)


def test_strata_are_prediction_quantiles() -> None:
    cfg = harness.GridConfig.from_dict(DESIGNS)
    prep = harness.prepare_scenario(cfg, 1)
    assert prep.strata is not None
    s = prep.strata[prep.pool]
    assert set(np.unique(s)) == {0, 1, 2, 3}
    h = prep.p_all[prep.pool] @ np.array([1.0, -1.0])
    means = [h[s == k].mean() for k in range(4)]
    assert means == sorted(means)


@pytest.mark.parametrize(
    "scenario, match",
    [
        ({"name": "x", "mechanism": "mar", "design": "active"}, "mcar"),
        ({"name": "x", "mechanism": "mcar", "design": "lottery"}, "design"),
        (
            {"name": "x", "mechanism": "mcar", "design": "stratified_neyman", "n_strata": 1},
            "n_strata",
        ),
    ],
)
def test_design_scenario_validation(scenario: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        harness.Scenario(**scenario)  # type: ignore[arg-type]


def test_ppi_active_needs_a_design() -> None:
    with pytest.raises(ValueError, match="ppi_active"):
        harness.GridConfig.from_dict({**SMALL, "estimators": ["ppi_active"]})


def test_scenario_estimator_subsets() -> None:
    cfg = harness.GridConfig.from_dict(
        {
            **DESIGNS,
            "scenarios": [
                {
                    "name": "uniform",
                    "mechanism": "mcar",
                    "design": "uniform",
                    "estimators": ["classical", "ppi"],
                },
                {
                    "name": "active",
                    "mechanism": "mcar",
                    "design": "active",
                    "estimators": ["classical", "ppi_active"],
                },
            ],
            "estimators": ["classical", "ppi", "ppi_active"],
            "replicates": 2,
        }
    )
    res = harness.run_grid(cfg)
    got = {s: set(res.filter(pl.col("scenario") == s)["estimator"]) for s in ("uniform", "active")}
    assert got == {"uniform": {"classical", "ppi"}, "active": {"classical", "ppi_active"}}


def test_scenario_estimators_must_be_configured() -> None:
    with pytest.raises(ValueError, match="not in the grid"):
        harness.GridConfig.from_dict(
            {
                **DESIGNS,
                "scenarios": [
                    {
                        "name": "u",
                        "mechanism": "mcar",
                        "design": "uniform",
                        "estimators": ["ppi_pp"],
                    }
                ],
            }
        )


# --------------------------------------------------------------------------- MRP


MRP_GRID = {
    **SMALL,
    "name": "mrp_grid",
    "scenarios": [{"name": "mar", "mechanism": "mar"}],
    "estimators": ["classical", "ppi", "mrp", "mrp_ppi"],
    "n_labeled": [150],
    "replicates": 3,
}


def test_prepared_cells() -> None:
    cfg = harness.GridConfig.from_dict(MRP_GRID)
    prep = harness.prepare_scenario(cfg, 0)
    assert prep.cells is not None and prep.cell is not None and prep.x_cell is not None
    assert prep.cells.factor_names == ("province", "age", "tenure", "segment")
    assert prep.cells.pop_count.sum() == prep.pool.size
    assert (prep.cell[prep.pool] >= 0).all()
    assert prep.x_cell.shape == (prep.cells.n_cells, 2)
    # Each cell's segment equals province * 6 + age.
    f = prep.cells.factors
    assert (f[:, 3] == f[:, 0] * 6 + f[:, 1]).all()
    assert (prep.cells.segment == f[:, 3]).all()


def test_mrp_rows_and_own_stream() -> None:
    cfg = harness.GridConfig.from_dict(MRP_GRID)
    res = harness.run_grid(cfg)
    assert set(res["estimator"]) == {"classical", "ppi", "mrp", "mrp_ppi"}
    m = res.filter(pl.col("estimator").is_in(["mrp", "mrp_ppi"]))
    assert m["point"].is_finite().all() and (m["upper"] > m["lower"]).all()
    base = harness.run_grid(
        harness.GridConfig.from_dict({**MRP_GRID, "estimators": ["classical", "ppi"]})
    )
    keys = ["scenario", "n_labeled", "replicate", "estimator"]
    others = res.filter(pl.col("estimator").is_in(["classical", "ppi"])).sort(keys)
    assert others.select(base.columns).equals(base.sort(keys))


def test_mrp_fit_setting_is_validated() -> None:
    with pytest.raises(ValueError, match="mrp_fit"):
        harness.GridConfig.from_dict({**MRP_GRID, "mrp_fit": "vi"})
