"""Smoke test for E4 (nonresponse) and its NUTS-vs-Laplace agreement table."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]

TINY = """\
name: e04_tiny
seed: 5
population: {n_customers: 6000}
labeler: {n_train: 600}
scenarios:
  - {name: mar, mechanism: mar, signal_strength: 0.6}
  - {name: mnar, mechanism: mnar, signal_strength: 0.6}
estimators: [classical, naive_model, ppi_pp_cf, mrp, mrp_ppi]
n_labeled: [150]
replicates: 3
"""


def _load():  # type: ignore[no-untyped-def]
    path = ROOT / "experiments" / "e04_nonresponse.py"
    spec = importlib.util.spec_from_file_location("e04", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e04"] = module
    spec.loader.exec_module(module)
    return module


def test_e04_runs(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e04 = _load()
    args = ["--config", str(cfg), "--results-dir", str(tmp_path), "--docs-dir", str(tmp_path)]
    assert e04.main(args) == 0
    s = pl.read_csv(tmp_path / "e04_tiny_summary.csv")
    assert set(s["estimator"]) == {"classical", "naive_model", "ppi_pp_cf", "mrp", "mrp_ppi"}
    for name in ("e04_tiny_mar.png", "e04_tiny_mnar.png"):
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"


def test_nuts_agreement_table() -> None:
    e04 = _load()
    base = {
        "scenario": ["mar"] * 4, "n_labeled": [500] * 4, "replicate": [0, 1, 0, 1],
        "estimator": ["mrp", "mrp", "mrp_ppi", "mrp_ppi"],
        "point": [10.0, 12.0, 11.0, 9.0], "lower": [6.0, 8.0, 7.0, 5.0],
        "upper": [14.0, 16.0, 15.0, 13.0], "se": [2.0, 2.0, 2.0, 2.0],
        "covered": [True, True, True, False],
    }  # fmt: skip
    lap = pl.DataFrame(base)
    nuts = lap.with_columns(pl.col("point") + 1.0, pl.col("se") * 1.1)
    t = e04.nuts_agreement(lap, nuts)
    np.testing.assert_allclose(t["mean_d_point"], [0.5, 0.5])
    np.testing.assert_allclose(t["mean_se_ratio"], [1.1, 1.1])
    assert t["same_coverage_call"].to_list() == [1.0, 1.0]
