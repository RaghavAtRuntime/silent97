"""E7: how fragile is a conclusion? Sensitivity curves and breakpoints on E4 and E6.

Reads stored replicate results (no refitting) and applies the delta adjustment of
:mod:`silent97.estimators.sensitivity`: non-respondents' promoter share +δ pp and
detractor share -δ pp relative to the estimator's implied rates, so the NPS estimate and
its interval move by 2 δ w_U, with w_U = (M - n) / M the non-respondent share of the
pool of M customers.

For each source, scenario, estimator and n:

- **True δ.** The simulator knows the truth, so the δ that the estimator's MAR
  correction misses is δ_true = -bias / (2 w_U), the population shift between the
  non-respondents' true NPS and the NPS the estimator implies for them (in expectation
  over surveys). It is estimated on the even-numbered replicates (calibration half) and
  evaluated on the odd-numbered ones, so no evaluation replicate informs it.
- **Coverage** of the truth on the evaluation half: at δ = 0 (the base interval), at
  δ_true (the interval under the true assumption), and of the sensitivity bounds over
  |δ| <= |δ_true| (lower edge at -|δ_true|, upper edge at +|δ_true|).
- **Breakpoint δ*** of the conclusion "NPS > threshold" per replicate
  (:func:`silent97.estimators.sensitivity.breakpoint`), and the share of replicates in
  which the conclusion holds at δ = 0.

Criterion A6 (docs/PLAN.md): "sensitivity bounds contain the truth at the stated
assumption". Operationalised here as: at δ_true, the interval's empirical coverage is not
significantly below the lower edge of the A1 band (Wilson 95% upper bound >= 0.88). The
bounds over |δ| <= |δ_true| contain that interval, so they cover at least as often.

``naive_model`` is excluded: its interval ignores model error, so no shift of its centre
can make it valid (E4, E6).

Usage::

    uv run python experiments/e07_sensitivity.py [--config configs/e07_sensitivity.yaml]
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
import yaml

from silent97.estimators import sensitivity
from silent97.eval import metrics
from silent97.types import Estimate
from silent97.viz import figures

A1_LOW = 0.88
A1_HIGH = 0.92


@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    path: str
    threshold: float
    estimators: tuple[str, ...]
    label: str = ""


@dataclasses.dataclass(frozen=True)
class E7Config:
    name: str
    sources: tuple[Source, ...]
    delta_grid: dict[str, float]
    illustrate: dict[str, Any]
    alpha: float = 0.1


def load_config(path: str | Path) -> E7Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["sources"] = tuple(
        Source(**{**s, "estimators": tuple(s["estimators"])}) for s in raw["sources"]
    )
    return E7Config(**raw)


def with_nonrespondent_share(rows: pl.DataFrame) -> pl.DataFrame:
    """Add ``w_u`` = (M - n) / M. The pool size M per scenario and n is taken over all
    estimators (``classical`` rows carry ``n_unlabeled = 0``)."""
    keys = ["scenario", "n_labeled"]
    pool = rows.group_by(keys).agg(
        (pl.col("n_labeled") + pl.col("n_unlabeled")).max().alias("pool_size")
    )
    out = rows.join(pool, on=keys, how="left")
    bad = out.filter(pl.col("pool_size") <= pl.col("n_labeled"))
    if bad.height:
        raise ValueError("cannot infer the pool size: no rows with n_unlabeled > 0")
    return out.with_columns(
        ((pl.col("pool_size") - pl.col("n_labeled")) / pl.col("pool_size")).alias("w_u")
    )


def _estimate(r: dict[str, Any]) -> Estimate:
    return Estimate(
        point=r["point"], lower=r["lower"], upper=r["upper"], alpha=0.1, se=r["se"],
        n_labeled=r["n_labeled"], n_unlabeled=r["n_unlabeled"], method=r["estimator"],
        diagnostics={},
    )  # fmt: skip


def delta_star(rows: pl.DataFrame, *, threshold: float) -> list[float]:
    """Per-row breakpoint of "NPS > threshold" (pp; negative = flips if non-respondents
    are less satisfied)."""
    return [
        sensitivity.breakpoint(_estimate(r), nonrespondent_share=r["w_u"], threshold=threshold)
        for r in rows.iter_rows(named=True)
    ]


def _covers(rows: pl.DataFrame, delta: float, *, bounds: bool = False) -> int:
    hits = 0
    for r in rows.iter_rows(named=True):
        if bounds:
            lo = sensitivity.delta_adjust(
                _estimate(r), [-abs(delta)], nonrespondent_share=r["w_u"]
            ).lower[0]
            hi = sensitivity.delta_adjust(
                _estimate(r), [abs(delta)], nonrespondent_share=r["w_u"]
            ).upper[0]
        else:
            c = sensitivity.delta_adjust(_estimate(r), [delta], nonrespondent_share=r["w_u"])
            lo, hi = c.lower[0], c.upper[0]
        hits += bool(lo <= r["truth"] <= hi)
    return hits


def a6_table(rows: pl.DataFrame, *, threshold: float, estimators: Sequence[str]) -> pl.DataFrame:
    """Per (scenario, estimator, n): true δ (calibration half), coverage at δ = 0, at
    δ_true and of the bounds |δ| <= |δ_true| (evaluation half), and breakpoints."""
    out: list[dict[str, Any]] = []
    keys = ["scenario", "estimator", "n_labeled"]
    rows = rows.filter(pl.col("estimator").is_in(list(estimators)))
    for key, g in rows.group_by(keys, maintain_order=True):
        scenario, estimator, n = key
        cal = g.filter(pl.col("replicate") % 2 == 0)
        ev = g.filter(pl.col("replicate") % 2 == 1)
        w = float(g["w_u"][0])
        bias_cal = float((cal["point"] - cal["truth"]).mean())  # type: ignore[operator]
        d_true = -bias_cal / (2.0 * w)
        m = ev.height
        k0, kt, kb = _covers(ev, 0.0), _covers(ev, d_true), _covers(ev, d_true, bounds=True)
        lo_t, hi_t = metrics.wilson_interval(kt, m)
        lo_0, hi_0 = metrics.wilson_interval(k0, m)
        lo_b, hi_b = metrics.wilson_interval(kb, m)
        stars = np.array(delta_star(ev, threshold=threshold))
        holds = stars != 0.0
        out.append(
            {
                "scenario": scenario,
                "estimator": estimator,
                "n_labeled": int(n),
                "truth": float(g["truth"][0]),
                "w_u": w,
                "bias": float((g["point"] - g["truth"]).mean()),  # type: ignore[operator]
                "delta_true": d_true,
                "n_eval": m,
                "coverage_at_zero": k0 / m,
                "coverage_at_zero_lo": lo_0,
                "coverage_at_zero_hi": hi_0,
                "coverage_at_true": kt / m,
                "coverage_at_true_lo": lo_t,
                "coverage_at_true_hi": hi_t,
                "coverage_bounds": kb / m,
                "coverage_bounds_lo": lo_b,
                "coverage_bounds_hi": hi_b,
                "share_holding": float(holds.mean()),
                "median_delta_star": float(np.median(stars[holds])) if holds.any() else 0.0,
                "a6_ok": hi_t >= A1_LOW,
                "in_a1_band": lo_t <= A1_HIGH and hi_t >= A1_LOW,
            }
        )
    order = {e: i for i, e in enumerate(estimators)}
    return (
        pl.DataFrame(out)
        .with_columns(pl.col("estimator").replace_strict(order).alias("_e"))
        .sort("scenario", "_e", "n_labeled")
        .drop("_e")
    )


def curve(
    rows: pl.DataFrame, *, scenario: str, n_labeled: int, estimators: Sequence[str],
    replicate: int, deltas: np.ndarray,
) -> pl.DataFrame:  # fmt: skip
    """Sensitivity curve of one survey (an evaluation replicate) per estimator."""
    if replicate % 2 == 0:
        raise ValueError("illustrate an evaluation (odd-numbered) replicate")
    frames = []
    for est in estimators:
        sel = rows.filter(
            (pl.col("scenario") == scenario) & (pl.col("n_labeled") == n_labeled)
            & (pl.col("estimator") == est) & (pl.col("replicate") == replicate)
        )  # fmt: skip
        if sel.height != 1:
            raise ValueError(f"no unique row for {scenario}/{est}/n={n_labeled}/rep={replicate}")
        r = sel.row(0, named=True)
        c = sensitivity.delta_adjust(_estimate(r), deltas, nonrespondent_share=r["w_u"])
        frames.append(
            pl.DataFrame(
                {
                    "estimator": est,
                    "delta": c.delta,
                    "point": c.point,
                    "lower": c.lower,
                    "upper": c.upper,
                    "truth": r["truth"],
                }
            )
        )
    return pl.concat(frames)


def markdown(tables: dict[str, pl.DataFrame], cfg: E7Config, curve_note: str) -> str:
    lines = [
        f"# {cfg.name} summary\n",
        "Generated by `experiments/e07_sensitivity.py` from stored replicate results. "
        "δ in percentage points: non-respondents' promoter share +δ, detractor share -δ "
        "relative to the estimator's implied rates (NPS moves by 2 δ w_U). δ_true = "
        "-bias / (2 w_U), estimated on even replicates; coverage and δ* on odd replicates. "
        "A6: at δ_true, coverage's Wilson 95% upper bound >= 0.88. δ*: median breakpoint "
        'of the conclusion "NPS > threshold" among replicates where it holds at δ = 0.\n',
    ]
    for s in cfg.sources:
        t = tables[s.name]
        lines += [
            f"## {s.label or s.name} (conclusion: NPS > {s.threshold:+g})\n",
            "| scenario | estimator | n | bias | δ_true | coverage at δ=0 | "
            "coverage at δ_true [Wilson 95%] | bounds |δ|<=|δ_true| | A6 | "
            "holds at δ=0 | median δ* |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in t.iter_rows(named=True):
            lines.append(
                f"| {r['scenario']} | {r['estimator']} | {r['n_labeled']} | {r['bias']:+.2f} | "
                f"{r['delta_true']:+.2f} | {r['coverage_at_zero']:.3f} | "
                f"{r['coverage_at_true']:.3f} [{r['coverage_at_true_lo']:.3f}, "
                f"{r['coverage_at_true_hi']:.3f}] | {r['coverage_bounds']:.3f} | "
                f"{'yes' if r['a6_ok'] else 'NO'} | {r['share_holding']:.2f} | "
                f"{r['median_delta_star']:+.2f} |"
            )
        lines.append("")
    lines.append(curve_note)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7: sensitivity curves and breakpoints")
    parser.add_argument("--config", default="configs/e07_sensitivity.yaml")
    parser.add_argument("--docs-dir", default="docs/results")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    start = time.perf_counter()
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pl.DataFrame] = {}
    rows_by_source: dict[str, pl.DataFrame] = {}
    for s in cfg.sources:
        path = Path(s.path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run its experiment first")
        rows = with_nonrespondent_share(pl.read_parquet(path))
        rows_by_source[s.name] = rows
        tables[s.name] = a6_table(rows, threshold=s.threshold, estimators=s.estimators)
    summary = pl.concat(
        [t.with_columns(pl.lit(name).alias("source")) for name, t in tables.items()]
    ).select("source", pl.all().exclude("source"))
    summary.write_csv(docs / f"{cfg.name}_summary.csv", float_precision=6)

    ill = cfg.illustrate
    src = next(s for s in cfg.sources if s.name == ill["source"])
    g = cfg.delta_grid
    deltas = np.arange(g["start"], g["stop"] + g["step"] / 2, g["step"])
    c = curve(
        rows_by_source[src.name], scenario=ill["scenario"], n_labeled=int(ill["n_labeled"]),
        estimators=ill["estimators"], replicate=int(ill["replicate"]), deltas=deltas,
    )  # fmt: skip
    c.write_csv(docs / f"{cfg.name}_curve.csv", float_precision=6)
    t_ill = tables[src.name]
    marks = {}
    for est in ill["estimators"]:
        row = t_ill.filter(
            (pl.col("scenario") == ill["scenario"]) & (pl.col("estimator") == est)
            & (pl.col("n_labeled") == int(ill["n_labeled"]))
        ).row(0, named=True)  # fmt: skip
        one = rows_by_source[src.name].filter(
            (pl.col("scenario") == ill["scenario"]) & (pl.col("estimator") == est)
            & (pl.col("n_labeled") == int(ill["n_labeled"]))
            & (pl.col("replicate") == int(ill["replicate"]))
        )  # fmt: skip
        marks[est] = {"delta_true": row["delta_true"],
                      "delta_star": delta_star(one, threshold=src.threshold)[0]}  # fmt: skip
    note = (
        f"Curve (`{cfg.name}_curve.csv`): {src.name}, scenario {ill['scenario']}, "
        f"n = {ill['n_labeled']}, replicate {ill['replicate']}; "
        + "; ".join(
            f"{e}: δ_true {m['delta_true']:+.2f}, δ* {m['delta_star']:+.2f}"
            for e, m in marks.items()
        )
        + "."
    )
    figures.sensitivity_figure(
        c, t_ill, docs / f"{cfg.name}.png", marks=marks, threshold=src.threshold,
        nominal=1 - cfg.alpha,
        title=f"E7: sensitivity to nonresponse not explained by covariates ({src.name})",
    )  # fmt: skip
    text = markdown(tables, cfg, note)
    (docs / f"{cfg.name}_summary.md").write_text(text)
    print(text)
    print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
