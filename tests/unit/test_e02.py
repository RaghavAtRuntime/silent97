"""E2 script: A3 is judged by the MSE-based effective n (reviewer decision, Gate G3)."""

import importlib.util
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

TINY = """\
name: e02_tiny
seed: 7
population: {n_customers: 4000}
labeler: {n_train: 500, kind: logistic}
scenarios:
  - {name: weak, mechanism: mcar, signal_strength: 0.3}
  - {name: strong, mechanism: mcar, signal_strength: 0.95}
estimators: [classical, ppi, ppi_pp]
n_labeled: [40]
replicates: 30
"""


def _load_e02():  # type: ignore[no-untyped-def]
    path = ROOT / "experiments" / "e02_model_quality.py"
    spec = importlib.util.spec_from_file_location("e02", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path) -> pl.DataFrame:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e02 = _load_e02()
    code = e02.main(
        ["--config", str(cfg), "--results-dir", str(tmp_path / "r"), "--docs-dir", str(tmp_path)]
    )
    assert code == 0
    return pl.read_csv(tmp_path / "e02_tiny_summary.csv")


def test_a3_checks_use_mse_ratio(tmp_path: Path) -> None:
    t = _run(tmp_path).filter(pl.col("estimator") == "ppi_pp")
    assert t.height == 2
    for r in t.iter_rows(named=True):
        within = r["mse_ratio_lo"] <= r["theory_ppi_pp"] <= r["mse_ratio_hi"]
        assert r["a3_within_mc"] == within
        if r["model_corr"] >= 0.6:
            assert r["a3_threshold_met"] == (r["mse_ratio"] >= 1.5)
        else:
            assert r["a3_threshold_met"] is None
    # The width-based check stays as information, under its own name.
    assert "theory_within_mc_widths" in t.columns


def test_a3_threshold_ignores_width_ratio(tmp_path: Path) -> None:
    e02 = _load_e02()
    t = _run(tmp_path)
    # Force a disagreement: widths say 2x, MSE says 1.2x, at rho >= 0.6.
    forced = t.with_columns(
        pl.lit(0.7).alias("model_corr"),
        pl.lit(2.0).alias("ratio"),
        pl.lit(1.2).alias("mse_ratio"),
    )
    flags = e02.a3_flags(forced)["a3_threshold_met"].to_list()
    assert flags == [False] * forced.height


def test_outputs_and_figures(tmp_path: Path) -> None:
    _run(tmp_path)
    md = (tmp_path / "e02_tiny_summary.md").read_text()
    assert "A3" in md and "MSE" in md
    for name in ("e02_tiny.png", "e02_tiny_widths.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
