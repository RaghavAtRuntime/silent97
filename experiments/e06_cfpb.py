"""E6: does it work on real text? CFPB dispute-rate benchmark.

Data: public CFPB complaints from 2015 to April 2017 with a narrative and a
"Consumer disputed?" value (``python -m silent97.data.cfpb`` builds the Parquet).
The outcome is disputed: a proxy for dissatisfaction, not NPS.

Design (configs/e06_cfpb.yaml):

1. A uniform ``train_fraction`` of complaints trains the tabular and embedding
   labelers. A fixed-size inference pool is then drawn uniformly from the rest; the
   LLM labels only this pool, which bounds its cost. The truth is the pool's dispute
   rate.
2. Disputed is coded as score 0 (detractor) and not disputed as 7, and predictions
   as ``(0, P(disputed))``, so every estimator estimates -100 x dispute rate (see
   ``silent97.data.cfpb``). Results are reported on the dispute-rate scale in percent:
   rate% = -NPS, with the bounds swapped. Widths in NPS points equal percentage
   points.
3. Each replicate draws n complaints uniformly from the pool (MCAR) and runs every
   estimator, with the harness's disjointness guard. ``ppi_crossfit`` cross-fits a
   logistic model on the one-hot product, issue and state.

The LLM labeler is a small open-weight model run locally on CPU
(``configs/llm.yaml``); nothing is sent to an external API. Scoring takes hours, so
it runs only with ``--run-llm``. Otherwise it prints its dry-run plan (items,
projected compute) and is skipped. Scores are cached, so later runs replay them.

Usage::

    uv run python experiments/e06_cfpb.py [--run-llm] [--labelers tabular embedding]
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

from silent97.data import cfpb
from silent97.eval import harness, metrics, planning
from silent97.labelers import embedding, llm
from silent97.labelers.cache import DiskCache
from silent97.labelers.tabular import TabularLabeler
from silent97.viz import figures

LABELERS = ("tabular", "embedding", "llm")


@dataclass(frozen=True)
class E6Config:
    """E6 settings; see ``configs/e06_cfpb.yaml``."""

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


def load_config(path: str | Path) -> E6Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(E6Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown E6 key(s) {unknown}")
    for key in ("labelers", "estimators", "n_labeled"):
        raw[key] = tuple(raw.get(key, ()))
    return E6Config(**raw)


def to_rate_scale(df: pl.DataFrame) -> pl.DataFrame:
    """NPS-coded results -> dispute rate in percent (rate% = -NPS, bounds swapped)."""
    return df.with_columns(
        (-pl.col("point")).alias("point"),
        (-pl.col("upper")).alias("lower"),
        (-pl.col("lower")).alias("upper"),
        (-pl.col("truth")).alias("truth"),
    )


def build_scenarios(
    cfg: E6Config,
    df: pl.DataFrame,
    *,
    encoder: embedding.Encoder | None,
    llm_scorer: llm.Scorer | None,
    run_llm: bool,
    log: Callable[[str], None],
) -> tuple[list[harness.PreparedScenario], dict[str, Any]]:
    """Split, fit or call each labeler, and return one prepared scenario per labeler."""
    n_total = df.height
    rng = np.random.default_rng([cfg.seed, 1])
    perm = rng.permutation(n_total)
    n_train = round(cfg.train_fraction * n_total)
    train_idx = np.sort(perm[:n_train]).astype(np.int64)
    rest = perm[n_train:]
    pool = np.sort(rng.choice(rest, size=min(cfg.pool_size, rest.size), replace=False)).astype(
        np.int64
    )

    scores = cfpb.dispute_scores(df["disputed"].to_numpy())
    x_tab, _ = cfpb.tabular_design(df)
    texts = df["narrative"].to_list()
    value = np.where(scores == 0, -100.0, 0.0)
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
            items = [
                llm.render_cfpb_item(
                    complaint_id=df["complaint_id"][int(i)],
                    narrative=texts[int(i)],
                    product=df["product"][int(i)],
                    issue=df["issue"][int(i)],
                    state=df["state"][int(i)],
                )
                for i in pool
            ]
            llm_cfg = llm.load_config(cfg.llm_config)
            scorer = llm_scorer
            if run_llm and scorer is None:
                scorer = llm.transformers_scorer(llm_cfg)
            res = llm.label_items(
                items,
                llm_cfg,
                cache=DiskCache(cache_root / "llm"),
                scorer=scorer,
                dry_run=not run_llm,
                log=log,
            )
            llm_report = {
                "dry_run": res.predictions is None,
                "model": llm_cfg.model_id,
                "device": llm_cfg.device,
                "plan": dataclasses.asdict(res.plan),
                "n_scored": res.n_scored,
                "n_failed": res.n_failed,
                "seconds": res.seconds,
                "spent_usd": 0.0,
            }
            if res.predictions is None:
                log("LLM labeler skipped (dry run). Pass --run-llm to score the pool locally.")
                continue
            preds[name] = res.predictions

    preps = []
    for name, p_pool in preds.items():
        p_all = np.full((n_total, 2), np.nan)
        # CFPB coding has no promoters: the promoter column is identically zero.
        p_all[pool] = np.column_stack([np.zeros(pool.size), p_pool[:, 1]])
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
    cfg: E6Config,
    *,
    encoder: embedding.Encoder | None = None,
    llm_scorer: llm.Scorer | None = None,
    run_llm: bool = False,
    results_dir: str | Path = "data/results",
    docs_dir: str | Path = "docs/results",
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Run E6 and write results, summary, figures and the LLM report."""
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
    rate = to_rate_scale(results).with_columns(pl.col("truth").alias("truth_rate_pct"))
    raw = harness.write_results(rate, Path(results_dir) / f"{cfg.name}.parquet")

    summary = metrics.summarize(rate).rename(
        {"mean_width": "mean_width_pp", "bias": "bias_pp", "rmse": "rmse_pp"}
    )
    corr = rate.group_by("scenario").agg(pl.col("model_corr").first())
    summary = summary.join(corr, on="scenario")
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    summary_path = docs / f"{cfg.name}_summary.csv"
    summary.write_csv(summary_path, float_precision=6)
    lines = [
        f"# {cfg.name} summary\n",
        f"Generated by `experiments/e06_cfpb.py` ({cfg.replicates} replicates, seed {cfg.seed}). "
        f"Pool of {preps[0].pool.size} complaints; true dispute rate {preps[0].truth * -1:.3f}%. "
        "Widths and bias in percentage points.\n",
        "| labeler (ρ) | estimator | n | coverage [Wilson 95%] | within A1 | mean width | bias "
        "| effective n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {r['scenario']} ({r['model_corr']:.3f}) | {r['estimator']} | {r['n_labeled']} | "
            f"{r['coverage']:.3f} [{r['coverage_lo']:.3f}, {r['coverage_hi']:.3f}] | "
            f"{'yes' if r['coverage_ok'] else 'NO'} | {r['mean_width_pp']:.2f} | "
            f"{r['bias_pp']:+.2f} | {r['effective_n']:.0f} |"
        )
    lines += planning_lines(summary, preps, cfg.seed)
    (docs / f"{cfg.name}_summary.md").write_text("\n".join(lines) + "\n")
    for p in preps:
        figures.coverage_figure(
            summary.rename({"mean_width_pp": "mean_width"}).filter(
                pl.col("scenario") == p.scenario.name
            ),
            docs / f"{cfg.name}_{p.scenario.name}.png",
            nominal=1 - cfg.alpha,
            title=f"E6, CFPB dispute rate, {p.scenario.name} labeler (ρ = {p.model_corr:.2f})",
            width_label="mean interval width (percentage points, log scale)",
        )
    trap_n = 200 if 200 in cfg.n_labeled else cfg.n_labeled[0]
    figures.trap_figure(
        summary.rename({"mean_width_pp": "mean_width", "bias_pp": "bias"}),
        docs / f"{cfg.name}_trap.png", n=trap_n, truth=-preps[0].truth,
        value_label="dispute rate (%)",
        title="Model-only dispute rate vs survey-corrected (CFPB complaints)",
    )  # fmt: skip
    report_path = docs / f"{cfg.name}_llm_report.json"
    report_path.write_text(json.dumps(llm_report, indent=2, sort_keys=True, default=str) + "\n")
    return {"results": raw, "summary": summary_path, "llm_report": report_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6: CFPB dispute-rate benchmark")
    parser.add_argument("--config", default="configs/e06_cfpb.yaml")
    parser.add_argument("--labelers", nargs="*", default=None, help="override the labeler list")
    parser.add_argument("--run-llm", action="store_true", help="score the pool with the local LLM")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.labelers is not None:
        cfg = dataclasses.replace(cfg, labelers=tuple(args.labelers))
    out = run(cfg, run_llm=args.run_llm)
    print(Path(out["summary"]).with_suffix(".md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
