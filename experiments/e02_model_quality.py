"""E2: how much does model quality buy?

Simulator, MCAR, the labeler signal strength swept so the model-outcome correlation
rho runs from 0 to about 0.9 (configs/e02_model_quality.yaml). For each rho and n:

- **effective n / n from widths**, (mean classical width / mean method width)², the
  docs/PLAN.md definition. It measures the precision each method *reports*;
- **MSE ratio**, MSE_classical / MSE_method. It measures the precision actually
  *achieved*;

both with paired-bootstrap 95% intervals over replicates, against the theory curve
for PPI++, 1 / (1 - rho^2 / (1 + n/N)).

Checks criterion A3 (docs/PLAN.md): within Monte Carlo error of the theory, and at
least 1.5x when rho >= 0.6.

Usage::

    uv run python experiments/e02_model_quality.py [--replicates R] [--n-jobs J]
    uv run python experiments/e02_model_quality.py \
        --from-results data/results/e02_model_quality.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from silent97.eval import harness, metrics
from silent97.viz import figures

A3_RHO, A3_RATIO = 0.6, 1.5


def mse_ratios(results: pl.DataFrame, rng: np.random.Generator, n_boot: int = 2000) -> pl.DataFrame:
    """MSE_classical / MSE_method per (scenario, n, estimator) with a paired bootstrap."""
    rows: list[dict[str, Any]] = []
    for (scenario, n), part in results.group_by(["scenario", "n_labeled"], maintain_order=True):
        err = part.with_columns((pl.col("point") - pl.col("truth")).alias("err"))
        wide = err.pivot(on="estimator", index="replicate", values="err").sort("replicate")
        ref = wide["classical"].to_numpy() ** 2
        idx = rng.integers(0, ref.size, size=(n_boot, ref.size))
        for est in (c for c in wide.columns if c not in ("replicate", "classical")):
            sq = wide[est].to_numpy() ** 2
            boot = ref[idx].mean(axis=1) / sq[idx].mean(axis=1)
            rows.append(
                {
                    "scenario": scenario,
                    "n_labeled": n,
                    "estimator": est,
                    "mse_ratio": float(ref.mean() / sq.mean()),
                    "mse_ratio_lo": float(np.quantile(boot, 0.025)),
                    "mse_ratio_hi": float(np.quantile(boot, 0.975)),
                }
            )
    return pl.DataFrame(rows)


def _within(lo: str, hi: str) -> pl.Expr:
    return (pl.col(lo) <= pl.col("theory_ppi_pp")) & (pl.col("theory_ppi_pp") <= pl.col(hi))


def a3_table(results: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Efficiency ratios with bootstrap intervals, theory values and the A3 checks."""
    boot = metrics.effective_n_bootstrap(results, rng=np.random.default_rng([seed, 99]))
    mse = mse_ratios(results, np.random.default_rng([seed, 98]))
    unlabeled = pl.col("n_unlabeled").filter(pl.col("estimator") == "ppi").first()
    info = results.group_by("scenario").agg(
        pl.col("model_corr").first(),
        pl.col("signal_strength").first(),
        (unlabeled + pl.col("n_labeled").first()).alias("pool"),
    )
    keys = ["scenario", "estimator", "n_labeled"]
    cov = metrics.summarize(results).select(*keys, "coverage", "coverage_ok")
    t = boot.join(info, on="scenario").join(cov, on=keys).join(mse, on=keys, how="left")
    theory = [
        metrics.ppi_pp_theory_ratio(r["model_corr"], r["n_labeled"], r["pool"] - r["n_labeled"])
        for r in t.iter_rows(named=True)
    ]
    t = t.with_columns(pl.Series("theory_ppi_pp", theory))
    t = t.with_columns(
        _within("ratio_lo", "ratio_hi").alias("theory_within_mc"),
        _within("mse_ratio_lo", "mse_ratio_hi").alias("theory_within_mc_mse"),
        pl.when(pl.col("model_corr") >= A3_RHO)
        .then(pl.col("ratio") >= A3_RATIO)
        .otherwise(None)
        .alias("a3_threshold_met"),
    )
    return t.sort(["n_labeled", "model_corr", "estimator"])


def _yes(flag: bool | None) -> str:
    return "" if flag is None else ("yes" if flag else "NO")


def markdown(t: pl.DataFrame) -> str:
    head = (
        "| n | signal | ρ | estimator | effective n / n (widths) [95%] | MSE ratio [95%] | "
        "theory (PPI++) | theory within MC: widths / MSE | ≥1.5× (ρ ≥ 0.6) | coverage |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in t.filter(pl.col("estimator") != "classical").iter_rows(named=True):
        pp = r["estimator"] == "ppi_pp"
        within = f"{_yes(r['theory_within_mc'])} / {_yes(r['theory_within_mc_mse'])}" if pp else ""
        lines.append(
            f"| {r['n_labeled']} | {r['signal_strength']:.2f} | {r['model_corr']:.3f} | "
            f"{r['estimator']} | {r['ratio']:.3f} [{r['ratio_lo']:.3f}, {r['ratio_hi']:.3f}] | "
            f"{r['mse_ratio']:.3f} [{r['mse_ratio_lo']:.3f}, {r['mse_ratio_hi']:.3f}] | "
            f"{r['theory_ppi_pp']:.3f} | {within} | "
            f"{_yes(r['a3_threshold_met']) if pp else ''} | {r['coverage']:.3f} |"
        )
    return head + "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2: effective n against model quality")
    parser.add_argument("--config", default="configs/e02_model_quality.yaml")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--docs-dir", default="docs/results")
    parser.add_argument("--from-results", default=None, help="summarize saved replicates")
    args = parser.parse_args(argv)

    cfg = harness.load_grid_config(args.config)
    if args.replicates is not None:
        cfg = dataclasses.replace(cfg, replicates=args.replicates)
    if args.n_jobs is not None:
        cfg = dataclasses.replace(cfg, n_jobs=args.n_jobs)
    suffix = "" if args.replicates is None else f"_r{cfg.replicates}"

    start = time.perf_counter()
    if args.from_results is not None:
        raw = Path(args.from_results)
        results = pl.read_parquet(raw)
    else:
        results = harness.run_grid(cfg)
        raw = harness.write_results(results, Path(args.results_dir) / f"{cfg.name}{suffix}.parquet")
    elapsed = time.perf_counter() - start

    t = a3_table(results, cfg.seed)
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    t.write_csv(docs / f"{cfg.name}{suffix}_summary.csv", float_precision=6)
    (docs / f"{cfg.name}{suffix}_summary.md").write_text(
        f"# {cfg.name} summary\n\nGenerated by `experiments/e02_model_quality.py` from "
        f"`{args.config}` ({cfg.replicates} replicates, seed {cfg.seed}). Effective n / n "
        "(widths) = (mean classical width / mean method width)², the docs/PLAN.md "
        "definition: the precision each method reports. MSE ratio = MSE_classical / "
        "MSE_method: the precision achieved. 95% paired-bootstrap intervals over "
        "replicates.\n\n" + markdown(t)
    )
    pool = int(t["pool"][0])

    def curve(n: int) -> Any:
        return lambda r: metrics.ppi_pp_theory_ratio(r, n, pool - n)

    fig = figures.efficiency_figure(
        t,
        docs / f"{cfg.name}{suffix}.png",
        theory={n: curve(n) for n in cfg.n_labeled},
        threshold=(A3_RHO, A3_RATIO),
        title="E2: responses saved by model quality (simulator, MCAR, tabular labeler)",
    )
    print(markdown(t))
    print(f"{len(results)} rows in {elapsed:.0f} s -> {raw}, {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
