"""Smoke tests for E6 on Amazon reviews (stub encoder, stub LLM scorer)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.data import amazon
from tests.unit.test_embedding_labeler import StubEncoder
from tests.unit.test_llm_labeler import StubScorer

ROOT = Path(__file__).resolve().parents[2]


def _load():  # type: ignore[no-untyped-def]
    path = ROOT / "experiments" / "e06_amazon.py"
    spec = importlib.util.spec_from_file_location("e06_amazon", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e06_amazon"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reviews_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("amazon")
    rng = np.random.default_rng(0)
    with open(tmp / "Gift_Cards.jsonl", "w") as fh:
        for i in range(600):
            stars = int(rng.choice([1, 3, 4, 5, 5, 5]))
            word = {1: "awful", 3: "meh", 4: "good", 5: "great"}[stars]
            fh.write(
                json.dumps(
                    {
                        "rating": float(stars),
                        "title": word,
                        "text": f"{word} card {i}",
                        "images": [],
                        "asin": f"a{i % 7}",
                        "parent_asin": f"a{i % 7}",
                        "user_id": f"u{i}",
                        "timestamp": 1_500_000_000_000 + i * 86_400_000,
                        "helpful_vote": int(rng.integers(0, 3)),
                        "verified_purchase": True,
                    }
                )
                + "\n"
            )
    parquet, _ = amazon.build(tmp / "Gift_Cards.jsonl", tmp / "processed")
    return parquet


def _cfg(tmp: Path, data: Path) -> Path:
    llm_cfg = tmp / "llm.yaml"
    llm_cfg.write_text("model: stub/model\nmax_items: 1000\n")
    cfg = {
        "name": "e06a_tiny",
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
    }
    path = tmp / "e06a.yaml"
    path.write_text(json.dumps(cfg))
    return path


def test_runs_with_all_labelers(tmp_path: Path, reviews_parquet: Path) -> None:
    mod = _load()
    cfg = mod.load_config(_cfg(tmp_path, reviews_parquet))
    out = mod.run(
        cfg, encoder=StubEncoder(), llm_scorer=StubScorer(), run_llm=True,
        results_dir=tmp_path / "r", docs_dir=tmp_path / "d", log=lambda s: None,
    )  # fmt: skip
    res = pl.read_parquet(out["results"])
    assert set(res["scenario"].unique()) == {"tabular", "embedding", "llm"}
    assert res.height == 3 * 5 * 3
    # The truth is the pool's NPS, on the -100..100 scale.
    truth = res["truth"].unique().to_list()
    assert len(truth) == 1 and -100 <= truth[0] <= 100
    # Both prediction columns are used: the stub LLM says P(promoter)=0.1, P(detractor)=0.6.
    naive = res.filter((pl.col("scenario") == "llm") & (pl.col("estimator") == "naive_model"))
    np.testing.assert_allclose(naive["point"].to_numpy(), 100 * (0.1 - 0.6))
    for name in ("summary", "llm_report"):
        assert Path(out[name]).exists()
    assert (tmp_path / "d" / "e06a_tiny_trap.png").read_bytes()[:4] == b"\x89PNG"


def test_truth_is_pool_nps(tmp_path: Path, reviews_parquet: Path) -> None:
    mod = _load()
    cfg = mod.load_config(_cfg(tmp_path, reviews_parquet))
    df = pl.read_parquet(reviews_parquet)
    preps, _ = mod.build_scenarios(
        cfg, df, encoder=StubEncoder(), llm_scorer=None, run_llm=False, log=lambda s: None
    )
    p = preps[0]
    s = df["score"].to_numpy()[p.pool]
    assert p.truth == pytest.approx(100 * ((s >= 9).mean() - (s <= 6).mean()))
    assert np.intersect1d(p.pool, p.train_idx).size == 0
    assert {q.scenario.name for q in preps} == {"tabular", "embedding"}  # LLM skipped (dry run)


def test_score_only_fills_the_cache_without_a_grid(tmp_path: Path, reviews_parquet: Path) -> None:
    mod = _load()
    cfg = mod.load_config(_cfg(tmp_path, reviews_parquet))
    scorer = StubScorer()
    report = mod.score_llm_only(cfg, llm_scorer=scorer, log=lambda s: None)
    assert report["n_scored"] == scorer.n_scored > 0
    again = mod.score_llm_only(cfg, llm_scorer=StubScorer(), log=lambda s: None)
    assert again["n_scored"] == 0
