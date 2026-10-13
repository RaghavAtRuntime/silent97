"""E5: can we report thin segments?

Simulator, uniform (MCAR) surveys of the pool (configs/e05_segments.yaml). For every
replicate and each of the 60 segments (province x age band), four segment estimates
are compared with the segment's true pool NPS:

- ``raw``: classical NPS of the segment's respondents (Wald interval);
- ``ppi``: PPI within the segment (its respondents plus its unsurveyed customers);
- ``mrp`` and ``mrp_ppi``: the segment poststratification of the MRP fit (Laplace).

Segments are grouped by their number of respondents (``bins``), and RMSE and coverage
are reported per group. Criterion A5 (docs/PLAN.md): for segments with fewer than 30
responses, MRP has a lower RMSE than the raw segment means.

Usage::

    uv run python experiments/e05_segments.py [--replicates R]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from joblib import Parallel, delayed

from silent97.estimators import classical, mrp, ppi
from silent97.eval import harness, metrics
from silent97.viz import figures

METHODS = ("raw", "ppi", "mrp", "mrp_ppi")


@dataclasses.dataclass(frozen=True)
class E5Config:
    name: str
    seed: int
    n_labeled: tuple[int, ...]
    replicates: int
    bins: tuple[int, ...]
    signal_strength: float = 0.6
    population: dict[str, Any] = dataclasses.field(default_factory=dict)
    labeler: dict[str, Any] = dataclasses.field(default_factory=dict)
    alpha: float = 0.1
    n_jobs: int = 1


def load_config(path: str | Path) -> E5Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("n_labeled", "bins"):
        raw[key] = tuple(int(v) for v in raw[key])
    return E5Config(**raw)


def bin_labels(edges: Sequence[int]) -> list[str]:
    return [f"{a}-{b - 1}" for a, b in itertools.pairwise(edges)]


def assign_bin(counts: Sequence[int], edges: Sequence[int]) -> list[str | None]:
    labels = bin_labels(edges)
    out: list[str | None] = []
    for c in counts:
        k = int(np.searchsorted(edges, c, side="right")) - 1
        out.append(labels[k] if 0 <= k < len(labels) else None)
    return out


def _replicate(
    prep: harness.PreparedScenario, cfg: E5Config, n: int, rep: int
) -> list[dict[str, Any]]:
    assert prep.cells is not None and prep.cell is not None and prep.cells.segment is not None
    rng = np.random.default_rng([cfg.seed, 5, n, rep])
    lab = harness.draw_respondents(prep.pool, np.ones(prep.pool.size), n, rng)
    harness.check_disjoint(prep.train_idx, lab)
    seg_all = np.full(prep.cell.size, -1)
    in_cells = prep.cell >= 0
    seg_all[in_cells] = prep.cells.segment[prep.cell[in_cells]]
    unl = np.setdiff1d(prep.pool, lab)
    y = prep.scores
    value = 100.0 * ((y >= 9).astype(float) - (y <= 6).astype(float))
    fits = {
        "mrp": mrp.mrp(y[lab], prep.cell[lab], prep.cells, rng=rng, alpha=cfg.alpha),
        "mrp_ppi": mrp.mrp_ppi(
            y[lab], prep.cell[lab], prep.cells, x_cell=prep.x_cell, rng=rng, alpha=cfg.alpha
        ),
    }
    seg_fit = {m: dict(zip(e.diagnostics["segments"]["label"], range(60), strict=False))
               for m, e in fits.items()}  # fmt: skip
    rows: list[dict[str, Any]] = []
    for s in np.unique(prep.cells.segment):
        lab_s, unl_s = lab[seg_all[lab] == s], unl[seg_all[unl] == s]
        if lab_s.size < 2:
            continue
        pool_s = prep.pool[seg_all[prep.pool] == s]
        truth = float(value[pool_s].mean())
        ests = {
            "raw": classical.classical(y[lab_s], alpha=cfg.alpha),
            "ppi": ppi.ppi(y[lab_s], prep.p_all[lab_s], prep.p_all[unl_s], alpha=cfg.alpha),
        }
        out = {k: (e.point, e.lower, e.upper) for k, e in ests.items()}
        for m, e in fits.items():
            seg = e.diagnostics["segments"]
            i = seg_fit[m][int(s)]
            out[m] = (seg["point"][i], seg["lower"][i], seg["upper"][i])
        for method, (pt, lo, hi) in out.items():
            rows.append(
                {"n_labeled": n, "replicate": rep, "segment": int(s), "n_seg": int(lab_s.size),
                 "method": method, "truth": truth, "point": pt, "lower": lo, "upper": hi}
            )  # fmt: skip
    return rows


def summarize(rows: pl.DataFrame, edges: Sequence[int]) -> pl.DataFrame:
    binned = rows.with_columns(
        pl.Series("bin", assign_bin(rows["n_seg"].to_list(), edges), dtype=pl.String)
    ).filter(pl.col("bin").is_not_null())
    err = pl.col("point") - pl.col("truth")
    covered = (pl.col("lower") <= pl.col("truth")) & (pl.col("truth") <= pl.col("upper"))
    s = binned.group_by("bin", "method").agg(
        pl.len().alias("segments"),
        covered.sum().alias("n_covered"),
        (err**2).mean().sqrt().alias("rmse"),
        err.mean().alias("bias"),
        (pl.col("upper") - pl.col("lower")).mean().alias("mean_width"),
    )
    bounds = [
        metrics.wilson_interval(int(k), int(r))
        for k, r in zip(s["n_covered"], s["segments"], strict=True)
    ]
    order = {b: i for i, b in enumerate(bin_labels(edges))}
    return (
        s.with_columns(
            (pl.col("n_covered") / pl.col("segments")).alias("coverage"),
            pl.Series("coverage_lo", [b[0] for b in bounds]),
            pl.Series("coverage_hi", [b[1] for b in bounds]),
            pl.col("bin").replace_strict(order).alias("_o"),
            pl.col("method").replace_strict({m: i for i, m in enumerate(METHODS)}).alias("_m"),
        )
        .sort("_o", "_m")
        .drop("_o", "_m")
    )


def markdown(s: pl.DataFrame, cfg: E5Config) -> str:
    lines = [
        f"# {cfg.name} summary\n",
        f"Generated by `experiments/e05_segments.py` ({cfg.replicates} replicates per n, "
        f"n = {list(cfg.n_labeled)}, seed {cfg.seed}). Segment estimates vs the segment's true "
        "pool NPS, grouped by respondents per segment. A5: MRP RMSE below raw for segments "
        "with fewer than 30 responses.\n",
        "| respondents in segment | method | segment estimates | RMSE | bias | mean width | "
        "coverage [Wilson 95%] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s.iter_rows(named=True):
        lines.append(
            f"| {r['bin']} | {r['method']} | {r['segments']} | {r['rmse']:.2f} | "
            f"{r['bias']:+.2f} | {r['mean_width']:.2f} | {r['coverage']:.3f} "
            f"[{r['coverage_lo']:.3f}, {r['coverage_hi']:.3f}] |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E5: thin segments")
    parser.add_argument("--config", default="configs/e05_segments.yaml")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--docs-dir", default="docs/results")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.replicates is not None:
        cfg = dataclasses.replace(cfg, replicates=args.replicates)
    start = time.perf_counter()
    scenario = {"name": "mcar", "mechanism": "mcar", "signal_strength": cfg.signal_strength}
    grid = harness.GridConfig.from_dict(
        {
            "name": cfg.name,
            "seed": cfg.seed,
            "population": cfg.population,
            "labeler": cfg.labeler,
            "estimators": ["classical"],
            "n_labeled": list(cfg.n_labeled),
            "replicates": cfg.replicates,
            "scenarios": [scenario],
        }
    )
    prep = harness.prepare_scenario(grid, 0)
    tasks = [(n, r) for n in cfg.n_labeled for r in range(cfg.replicates)]
    chunks = Parallel(n_jobs=cfg.n_jobs)(delayed(_replicate)(prep, cfg, n, r) for n, r in tasks)
    rows = pl.DataFrame([row for chunk in chunks for row in chunk])
    raw = Path(args.results_dir) / f"{cfg.name}.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(raw)
    s = summarize(rows, cfg.bins)
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    s.write_csv(docs / f"{cfg.name}_summary.csv", float_precision=6)
    text = markdown(s, cfg)
    (docs / f"{cfg.name}_summary.md").write_text(text)
    figures.segment_figure(
        s, docs / f"{cfg.name}.png", bins=bin_labels(cfg.bins), nominal=1 - cfg.alpha,
        title="E5: thin segments (simulator, uniform survey)",
    )  # fmt: skip
    print(text)
    print(f"{rows.height} segment estimates in {time.perf_counter() - start:.0f} s -> {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
