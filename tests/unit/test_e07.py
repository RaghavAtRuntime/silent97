"""E7 (sensitivity curves and breakpoints) on synthetic replicate results."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
Z90 = 1.6448536269514722


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "e07", ROOT / "experiments" / "e07_sensitivity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e07"] = module
    spec.loader.exec_module(module)
    return module


def _results(bias: float, pool: int = 10_000, reps: int = 400, seed: int = 0) -> pl.DataFrame:
    """Replicate rows shaped like the harness output: a biased (MNAR-like) estimator."""
    rng = np.random.default_rng(seed)
    rows = []
    for est, se in (("classical", 4.0), ("ppi_pp_cf", 3.0), ("naive_model", 0.2)):
        for n in (200, 500):
            for r in range(reps):
                point = 20.0 + bias + se * rng.normal()
                rows.append(
                    {"scenario": "mnar", "mechanism": "mnar", "estimator": est,
                     "n_labeled": n, "n_unlabeled": 0 if est == "classical" else pool - n,
                     "replicate": r, "truth": 20.0, "point": point, "se": se,
                     "lower": point - Z90 * se, "upper": point + Z90 * se}
                )  # fmt: skip
    return pl.DataFrame(rows)


def _config(tmp_path: Path, path: Path) -> Path:
    cfg = tmp_path / "e07.yaml"
    cfg.write_text(
        "name: e07_tiny\n"
        "sources:\n"
        f"  - {{name: sim, path: {path}, threshold: 10.0,"
        " estimators: [classical, ppi_pp_cf]}\n"
        "delta_grid: {start: -10, stop: 10, step: 0.5}\n"
        "illustrate: {source: sim, scenario: mnar, n_labeled: 500,"
        " estimators: [classical, ppi_pp_cf], replicate: 1}\n"
    )
    return cfg


def test_nonrespondent_share_from_pool() -> None:
    e07 = _load()
    rows = e07.with_nonrespondent_share(_results(bias=0.0, reps=2))
    w = rows.filter(pl.col("n_labeled") == 200)["w_u"].unique().to_list()
    # Classical rows carry n_unlabeled = 0; the pool size comes from the other estimators.
    assert w == pytest.approx([(10_000 - 200) / 10_000])


def test_true_delta_and_coverage(tmp_path: Path) -> None:
    e07 = _load()
    bias = -6.0
    rows = e07.with_nonrespondent_share(_results(bias=bias))
    t = e07.a6_table(rows, threshold=10.0, estimators=["classical", "ppi_pp_cf"])
    assert set(t["estimator"]) == {"classical", "ppi_pp_cf"}
    for r in t.iter_rows(named=True):
        w = (10_000 - r["n_labeled"]) / 10_000
        # True delta: -bias / (2 w), estimated on the calibration half only.
        assert r["delta_true"] == pytest.approx(-bias / (2 * w), abs=0.8)
        assert r["coverage_at_true"] > r["coverage_at_zero"]
        assert r["coverage_bounds"] >= r["coverage_at_true"]
        assert r["n_eval"] == 200
    for col in ("coverage_at_true_lo", "coverage_at_true_hi", "median_delta_star",
                "share_holding", "a6_ok"):  # fmt: skip
        assert col in t.columns


def test_calibration_and_evaluation_halves_are_disjoint() -> None:
    """delta_true uses even replicates only: changing odd replicates leaves it unchanged."""
    e07 = _load()
    rows = e07.with_nonrespondent_share(_results(bias=-4.0))
    moved = rows.with_columns(
        pl.when(pl.col("replicate") % 2 == 1)
        .then(pl.col("point") + 50)
        .otherwise(pl.col("point"))
        .alias("point")
    )
    a = e07.a6_table(rows, threshold=10.0, estimators=["ppi_pp_cf"])
    b = e07.a6_table(moved, threshold=10.0, estimators=["ppi_pp_cf"])
    np.testing.assert_allclose(a["delta_true"].to_numpy(), b["delta_true"].to_numpy())


def test_unbiased_source_has_zero_true_delta() -> None:
    e07 = _load()
    rows = e07.with_nonrespondent_share(_results(bias=0.0, reps=2000, seed=3))
    t = e07.a6_table(rows, threshold=10.0, estimators=["ppi_pp_cf"])
    assert all(abs(d) < 0.3 for d in t["delta_true"])
    assert all(t["a6_ok"])


def test_main_writes_outputs(tmp_path: Path) -> None:
    e07 = _load()
    src = tmp_path / "sim.parquet"
    _results(bias=-5.0, reps=40).write_parquet(src)
    cfg = _config(tmp_path, src)
    assert e07.main(["--config", str(cfg), "--docs-dir", str(tmp_path)]) == 0
    t = pl.read_csv(tmp_path / "e07_tiny_summary.csv")
    assert {"source", "scenario", "estimator", "n_labeled", "delta_true"} <= set(t.columns)
    curve = pl.read_csv(tmp_path / "e07_tiny_curve.csv")
    assert {"estimator", "delta", "point", "lower", "upper"} <= set(curve.columns)
    assert curve.height == 2 * 41
    text = (tmp_path / "e07_tiny_summary.md").read_text()
    assert "A6" in text and "δ*" in text
    assert (tmp_path / "e07_tiny.png").read_bytes()[:4] == b"\x89PNG"


def test_missing_source_is_reported(tmp_path: Path) -> None:
    e07 = _load()
    cfg = _config(tmp_path, tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match=r"absent\.parquet"):
        e07.main(["--config", str(cfg), "--docs-dir", str(tmp_path)])


def test_breakpoint_matches_library() -> None:
    from silent97.estimators import sensitivity
    from silent97.types import Estimate

    e07 = _load()
    rows = e07.with_nonrespondent_share(_results(bias=0.0, reps=3))
    one = rows.filter(pl.col("estimator") == "ppi_pp_cf").head(1)
    r = one.row(0, named=True)
    base = Estimate(r["point"], r["lower"], r["upper"], 0.1, r["se"], r["n_labeled"],
                    r["n_unlabeled"], "ppi_pp_cf", {})  # fmt: skip
    expected = sensitivity.breakpoint(base, nonrespondent_share=r["w_u"], threshold=10.0)
    got = e07.delta_star(one, threshold=10.0)[0]
    assert math.isclose(got, expected)
