"""E6 (reviews): does it work on real text with a real satisfaction rating?

Data: Amazon Reviews 2023, Gift Cards (``python -m silent97.data.amazon``). The
outcome is the customer's own star rating on the NPS scale (score = 2 x stars: 5 stars
promoter, 4 passive, 1-3 detractor), so the full two-dimensional NPS estimators run on
real text. CFPB disputing is close to unpredictable from public data
(``docs/progress/phase-3.md``); this benchmark shows the method where text does carry
the signal.

Design (configs/e06_amazon.yaml), as for the CFPB benchmark:

1. A uniform ``train_fraction`` of reviews trains the tabular and embedding labelers.
   A fixed-size inference pool is drawn uniformly from the rest; the LLM labels only
   the pool. The truth is the pool's NPS.
2. Each replicate draws n reviews uniformly from the pool (MCAR) and runs every
   estimator, with the harness's disjointness guard. ``ppi_crossfit`` cross-fits a
   logistic model on the metadata features.

The LLM labeler runs locally (``configs/llm.yaml``); nothing is sent to an external
API. ``--score-only`` fills the LLM cache without running the grid (scoring takes
hours; it resumes from the cache), and ``--run-llm`` includes the LLM in the grid.

Usage::

    uv run python experiments/e06_amazon.py [--run-llm] [--score-only]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from silent97.data import amazon
from silent97.eval import harness, metrics, planning
from silent97.labelers import embedding, llm
from silent97.labelers.cache import DiskCache
from silent97.labelers.tabular import TabularLabeler
from silent97.viz import figures

LABELERS = ("tabular", "embedding", "llm")
TRAP_N = 200


@dataclass(frozen=True)
class E6AmazonConfig:
    """E6 (reviews) settings; see ``configs/e06_amazon.yaml``."""

    name: str
    seed: int
    data: str
    labelers: tuple[str, ...]
    estimators: tuple[str, ...]
    n_labeled: tuple[int, ...]
    replicates: int
    train_fraction: float = 0.2
    pool_size: int = 10_000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_config: str = "configs/llm.yaml"
    alpha: float = 0.1
    crossfit_folds: int = 10
    n_jobs: int = 1
    cache_dir: str = "cache"

    def __post_init__(self) -> None:
        bad = sorted(set(self.labelers) - set(LABELERS))
        if bad:
            raise ValueError(f"unknown labeler(s) {bad}; choose from {LABELERS}")
        if not 0 < self.train_fraction < 1:
            raise ValueError(f"train_fraction must lie in (0, 1), got {self.train_fraction}")


def load_config(path: str | Path) -> E6AmazonConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(E6AmazonConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown E6 key(s) {unknown}")
    for key in ("labelers", "estimators", "n_labeled"):
        raw[key] = tuple(raw.get(key, ()))
    return E6AmazonConfig(**raw)


def _split(cfg: E6AmazonConfig, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([cfg.seed, 1])
    perm = rng.permutation(n_total)
    n_train = round(cfg.train_fraction * n_total)
    train_idx = np.sort(perm[:n_train]).astype(np.int64)
    rest = perm[n_train:]
    pool = np.sort(rng.choice(rest, size=min(cfg.pool_size, rest.size), replace=False))
    return train_idx, pool.astype(np.int64)


def _llm_items(df: pl.DataFrame, pool: np.ndarray) -> list[llm.LLMItem]:
    amazon.assert_labeler_columns(["title", "text"])
    ids, titles, texts = df["review_id"], df["title"], df["text"]
    return [
        llm.render_review_item(review_id=ids[int(i)], title=titles[int(i)], text=texts[int(i)])
        for i in pool
    ]


def _label_llm(
    cfg: E6AmazonConfig,
    df: pl.DataFrame,
    pool: np.ndarray,
    *,
    llm_scorer: llm.Scorer | None,
    run_llm: bool,
    log: Callable[[str], None],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    llm_cfg = llm.load_config(cfg.llm_config)
    scorer = llm_scorer
    if run_llm and scorer is None:
        scorer = llm.transformers_scorer(llm_cfg, llm.SYSTEM_PROMPTS["amazon_reviews"])
    res = llm.label_items(
        _llm_items(df, pool),
        llm_cfg,
        cache=DiskCache(Path(cfg.cache_dir) / "llm"),
        scorer=scorer,
        dry_run=not run_llm,
        log=log,
    )
    report = {
        "dry_run": res.predictions is None,
        "model": llm_cfg.model_id,
        "device": llm_cfg.device,
        "plan": dataclasses.asdict(res.plan),
        "n_scored": res.n_scored,
        "n_failed": res.n_failed,
        "seconds": res.seconds,
        "spent_usd": 0.0,
    }
    return res.predictions, report


def score_llm_only(
    cfg: E6AmazonConfig,
    *,
    llm_scorer: llm.Scorer | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Score the pool with the LLM (filling the cache) without running the grid."""
    df = pl.read_parquet(cfg.data)
    _, pool = _split(cfg, df.height)
    _, report = _label_llm(cfg, df, pool, llm_scorer=llm_scorer, run_llm=True, log=log)
    return report


def build_scenarios(
    cfg: E6AmazonConfig,
    df: pl.DataFrame,
    *,
    encoder: embedding.Encoder | None,
    llm_scorer: llm.Scorer | None,
    run_llm: bool,
    log: Callable[[str], None],
) -> tuple[list[harness.PreparedScenario], dict[str, Any]]:
    """Split, fit or score each labeler, and return one prepared scenario per labeler."""
    n_total = df.height
    train_idx, pool = _split(cfg, n_total)
    scores = df["score"].to_numpy().astype(np.int64)
    x_tab, _ = amazon.tabular_design(df)
    texts = (df["title"] + "\n\n" + df["text"]).to_list()
    value = np.where(scores >= 9, 100.0, 0.0) - np.where(scores <= 6, 100.0, 0.0)
    truth = float(value[pool].mean())
    cache_root = Path(cfg.cache_dir)
    llm_report: dict[str, Any] = {"dry_run": True, "spent_usd": 0.0, "n_scored": 0}

    preds: dict[str, np.ndarray] = {}
    for name in cfg.labelers:
        if name == "tabular":
            lab = TabularLabeler(random_state=cfg.seed % (2**31 - 1)).fit(
                x_tab[train_idx], scores[train_idx]
            )
            preds[name] = lab.predict(x_tab[pool])
        elif name == "embedding":
            enc = encoder or embedding.sentence_transformer_encoder(cfg.embedding_model)
            ecache = DiskCache(cache_root / "embeddings")
            e_train = embedding.encode_cached(
                [texts[i] for i in train_idx], enc, cache=ecache, model_name=cfg.embedding_model
            )
            e_pool = embedding.encode_cached(
                [texts[i] for i in pool], enc, cache=ecache, model_name=cfg.embedding_model
            )
            preds[name] = (
                embedding.EmbeddingLabeler(enc).fit(e_train, scores[train_idx]).predict(e_pool)
            )
        else:
            p, llm_report = _label_llm(
                cfg, df, pool, llm_scorer=llm_scorer, run_llm=run_llm, log=log
            )
            if p is None:
                log("LLM labeler skipped (dry run). Pass --run-llm to include it.")
                continue
            preds[name] = p

    preps = []
    for name, p_pool in preds.items():
        p_all = np.full((n_total, 2), np.nan)
        p_all[pool] = p_pool
        preps.append(
            harness.PreparedScenario(
                scenario=harness.Scenario(name=name, mechanism="mcar"),
                signal_strength=float("nan"),
                x=x_tab,
                scores=scores,
                p_all=p_all,
                propensity=np.ones(n_total),
                train_idx=train_idx,
                pool=pool,
                truth=truth,
                model_corr=harness.model_correlation(p_all[pool], scores[pool]),
            )
        )
    return preps, llm_report


def planning_lines(
    summary: pl.DataFrame, preps: list[harness.PreparedScenario], seed: int, sample: int = 300
) -> list[str]:
    """Predicted gain from rho against the realized PPI++ gain, per labeler.

    rho is estimated twice: on the whole pool, and, as a team would in practice, on one
    random sample of ``sample`` past responses (with a bootstrap interval).
    """
    pool = preps[0].pool.size
    rho = {p.scenario.name: p.model_corr for p in preps}
    table = planning.compare_with_results(summary, rho=rho, pool=pool)
    lines = [
        "",
        "## Planning check: predicted vs realized gain (PPI++)",
        "",
        "Predicted gain = 1 / (1 - rho^2 / (1 + n/N)) from rho on the pool "
        "(`silent97.eval.planning`); realized gain = effective n / n from interval widths.",
        "",
        "| labeler | rho (pool) | rho from 300 past responses [95%] | n | predicted | realized |",
        "|---|---|---|---|---|---|",
    ]
    rng = np.random.default_rng([seed, 7])
    small = {}
    for p in preps:
        idx = rng.choice(p.pool, size=min(sample, p.pool.size), replace=False)
        try:
            est = planning.estimate_rho(p.scores[idx], p.p_all[idx], rng=rng)
            small[p.scenario.name] = f"{est.rho:.3f} [{est.lo:.3f}, {est.hi:.3f}]"
        except ValueError:  # constant predictions: rho is undefined
            small[p.scenario.name] = "n/a"
    for r in table.iter_rows(named=True):
        lines.append(
            f"| {r['scenario']} | {r['rho']:.3f} | {small[r['scenario']]} | {r['n_labeled']} | "
            f"{r['predicted_gain']:.3f} | {r['realized_gain']:.3f} |"
        )
    return lines


def run(
    cfg: E6AmazonConfig,
    *,
    encoder: embedding.Encoder | None = None,
    llm_scorer: llm.Scorer | None = None,
    run_llm: bool = False,
    results_dir: str | Path = "data/results",
    docs_dir: str | Path = "docs/results",
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Run E6 (reviews) and write results, summary, figures and the LLM report."""
    df = pl.read_parquet(cfg.data)
    preps, llm_report = build_scenarios(
        cfg, df, encoder=encoder, llm_scorer=llm_scorer, run_llm=run_llm, log=log
    )
    if not preps:
        raise RuntimeError("no labeler produced predictions")
    grid = harness.GridConfig(
        name=cfg.name, seed=cfg.seed, scenarios=tuple(p.scenario for p in preps),
        estimators=cfg.estimators, n_labeled=cfg.n_labeled, replicates=cfg.replicates,
        alpha=cfg.alpha, crossfit_folds=cfg.crossfit_folds, n_jobs=cfg.n_jobs,
    )  # fmt: skip
    start = time.perf_counter()
    results = harness.run_grid(grid, prepare=lambda g, i: preps[i])
    log(f"{results.height} rows in {time.perf_counter() - start:.0f} s")
    raw = harness.write_results(results, Path(results_dir) / f"{cfg.name}.parquet")

    summary = metrics.summarize(results)
    corr = results.group_by("scenario").agg(pl.col("model_corr").first())
    summary = summary.join(corr, on="scenario")
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    summary_path = docs / f"{cfg.name}_summary.csv"
    summary.write_csv(summary_path, float_precision=6)
    truth = preps[0].truth
    lines = [
        f"# {cfg.name} summary\n",
        f"Generated by `experiments/e06_amazon.py` ({cfg.replicates} replicates, seed "
        f"{cfg.seed}). Pool of {preps[0].pool.size} reviews; true NPS {truth:+.2f}. Widths "
        "and bias in NPS points.\n",
        "| labeler (ρ) | estimator | n | coverage [Wilson 95%] | within A1 | mean width | bias "
        "| effective n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {r['scenario']} ({r['model_corr']:.3f}) | {r['estimator']} | {r['n_labeled']} | "
            f"{r['coverage']:.3f} [{r['coverage_lo']:.3f}, {r['coverage_hi']:.3f}] | "
            f"{'yes' if r['coverage_ok'] else 'NO'} | {r['mean_width']:.2f} | "
            f"{r['bias']:+.2f} | {r['effective_n']:.0f} |"
        )
    lines += planning_lines(summary, preps, cfg.seed)
    (docs / f"{cfg.name}_summary.md").write_text("\n".join(lines) + "\n")
    for p in preps:
        figures.coverage_figure(
            summary.filter(pl.col("scenario") == p.scenario.name),
            docs / f"{cfg.name}_{p.scenario.name}.png",
            nominal=1 - cfg.alpha,
            title=f"E6, Amazon gift-card reviews (NPS), {p.scenario.name} labeler "
            f"(ρ = {p.model_corr:.2f})",
            width_label="mean interval width (NPS points, log scale)",
        )
    trap_n = TRAP_N if TRAP_N in cfg.n_labeled else cfg.n_labeled[0]
    figures.trap_figure(
        summary, docs / f"{cfg.name}_trap.png", n=trap_n, truth=truth, value_label="NPS",
        title="Model-only NPS vs survey-corrected NPS (Amazon gift-card reviews)",
    )  # fmt: skip
    report_path = docs / f"{cfg.name}_llm_report.json"
    report_path.write_text(json.dumps(llm_report, indent=2, sort_keys=True, default=str) + "\n")
    return {"results": raw, "summary": summary_path, "llm_report": report_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6 (reviews): Amazon gift-card NPS benchmark")
    parser.add_argument("--config", default="configs/e06_amazon.yaml")
    parser.add_argument("--labelers", nargs="*", default=None, help="override the labeler list")
    parser.add_argument("--run-llm", action="store_true", help="include the local LLM")
    parser.add_argument("--score-only", action="store_true", help="fill the LLM cache only")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.labelers is not None:
        cfg = dataclasses.replace(cfg, labelers=tuple(args.labelers))
    if args.score_only:
        print(json.dumps(score_llm_only(cfg), indent=2, default=str))
        return 0
    out = run(cfg, run_llm=args.run_llm)
    print(Path(out["summary"]).with_suffix(".md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
