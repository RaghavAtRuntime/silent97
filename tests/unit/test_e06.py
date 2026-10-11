"""Smoke tests for E6 on a synthetic CFPB-shaped file (stub encoder, stub LLM scorer)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.data import cfpb
from tests.unit.test_embedding_labeler import StubEncoder
from tests.unit.test_llm_labeler import StubScorer

ROOT = Path(__file__).resolve().parents[2]


def _load_e06():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("e06", ROOT / "experiments" / "e06_cfpb.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e06"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfpb_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("cfpb")
    rng = np.random.default_rng(0)
    path = tmp / "complaints.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(cfpb.COLUMNS))
        for i in range(600):
            disputed = rng.random() < 0.25
            word = "angry" if (disputed or rng.random() < 0.2) else "fine"
            product = rng.choice(["Credit card", "Mortgage", "Debt collection"])
            text = f"{word} {rng.choice(['great', 'fine'])} service, case {i}"
            state = rng.choice(["CA", "TX"])
            w.writerow(
                [str(i), "2016-01-15", product, "Billing", state, text, "Yes" if disputed else "No"]
            )
    parquet, _ = cfpb.build(path, tmp / "processed")
    return parquet


def _cfg(tmp: Path, data: Path, **extra: object) -> Path:
    llm_cfg = tmp / "llm.yaml"
    llm_cfg.write_text("model: stub/model\nmax_items: 1000\n")
    cfg = {
        "name": "e06_tiny",
        "seed": 3,
        "data": str(data),
        "train_fraction": 0.3,
        "pool_size": 300,
        "labelers": ["tabular", "embedding", "llm"],
        "embedding_model": "stub-bow",
        "llm_config": str(llm_cfg),
        "estimators": ["classical", "naive_model", "ppi", "ppi_pp", "ppi_crossfit"],
        "n_labeled": [40],
        "replicates": 3,
        "crossfit_folds": 3,
        "cache_dir": str(tmp / "cache"),
        **extra,
    }
    path = tmp / "e06.yaml"
    path.write_text(json.dumps(cfg))  # JSON is valid YAML
    return path


def test_e06_runs_with_all_labelers(tmp_path: Path, cfpb_parquet: Path) -> None:
    e06 = _load_e06()
    cfg = e06.load_config(_cfg(tmp_path, cfpb_parquet))
    scorer = StubScorer()
    out = e06.run(
        cfg, encoder=StubEncoder(), llm_scorer=scorer, run_llm=True,
        results_dir=tmp_path / "r", docs_dir=tmp_path / "d", log=lambda s: None,
    )  # fmt: skip
    results = pl.read_parquet(out["results"])
    assert set(results["scenario"]) == {"tabular", "embedding", "llm"}
    assert results.height == 3 * 3 * 5
    assert scorer.n_scored == 300  # the pool only, once each
    # Dispute-rate scale: truth equals the pool's dispute rate in percent.
    df = pl.read_parquet(cfpb_parquet)
    truth = results["truth_rate_pct"][0]
    assert 0 < truth < 100
    assert set(results["truth_rate_pct"].to_list()) == {truth}
    summary = pl.read_csv(out["summary"])
    assert {"scenario", "estimator", "coverage", "mean_width_pp"} <= set(summary.columns)
    report = json.loads(Path(out["llm_report"]).read_text())
    assert report["n_scored"] == 300 and report["spent_usd"] == 0.0
    assert df.height == 600


def test_e06_dry_run_skips_llm(tmp_path: Path, cfpb_parquet: Path) -> None:
    e06 = _load_e06()
    cfg = e06.load_config(_cfg(tmp_path, cfpb_parquet))
    scorer = StubScorer()
    lines: list[str] = []
    out = e06.run(
        cfg, encoder=StubEncoder(), llm_scorer=scorer, run_llm=False,
        results_dir=tmp_path / "r", docs_dir=tmp_path / "d", log=lines.append,
    )  # fmt: skip
    assert scorer.n_scored == 0
    results = pl.read_parquet(out["results"])
    assert set(results["scenario"]) == {"tabular", "embedding"}
    assert any("dry run" in line.lower() for line in lines)
    report = json.loads(Path(out["llm_report"]).read_text())
    assert report["dry_run"] is True and report["spent_usd"] == 0.0


def test_e06_rates_are_negated_nps() -> None:
    e06 = _load_e06()
    df = pl.DataFrame(
        {"point": [-20.0], "lower": [-25.0], "upper": [-15.0], "truth": [-22.0], "width": [10.0]}
    )
    out = e06.to_rate_scale(df)
    assert out.row(0, named=True) == {
        "point": 20.0,
        "lower": 15.0,
        "upper": 25.0,
        "truth": 22.0,
        "width": 10.0,
    }
