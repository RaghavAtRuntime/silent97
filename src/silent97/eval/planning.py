"""Survey planning: how many responses does a model save, before running a new survey?

PPI++'s gain over a plain survey is governed by one number, the correlation rho
between each customer's NPS value v = 100 (1[y >= 9] - 1[y <= 6]) and the model's
predicted value h = 100 (P(promoter) - P(detractor)). With N unlabeled customers and n
survey responses,

.. math::

    \\frac{\\operatorname{Var}(\\hat\\theta_{\\text{PPI++}})}{\\operatorname{Var}(\\hat\\theta_{\\text{classical}})}
    = 1 - \\frac{\\rho^2}{1 + n/N},

so one response is worth ``gain = 1 / (1 - rho^2 / (1 + n/N))`` classical responses
(Angelopoulos, Duchi & Zrnic, PPI++, arXiv 2311.01453; the variance-optimal lambda for
the NPS contrast). E2 checked this curve by simulation (A3: 18 of 20 points within
Monte Carlo error, judged on MSE).

rho can be estimated from data a team already has: past survey responses plus the
model's predictions for those same customers. :func:`report` gives rho with a bootstrap
interval and turns it into the responses needed for a target margin of error, with
and without the model.

Assumptions and caveats.

- The predictions must be **out of sample** for the surveyed customers (the model was
  not trained on them). Otherwise rho, and every number derived from it, is inflated.
- Past respondents are self-selected. rho among them can differ from rho in a future
  random sample, so plan with the low end of the interval.
- The gain is asymptotic. At small n, estimating lambda costs a little (E2: about 2%
  at n = 100 with a useless model), so treat gains below about 1.05 as none.
- Validity never depends on rho: a poor model only costs width (A1, E1, E6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from silent97.eval.metrics import ppi_pp_theory_ratio

__all__ = [
    "PlanningReport",
    "RhoEstimate",
    "compare_with_results",
    "estimate_rho",
    "format_report",
    "gain",
    "planning_table",
    "report",
    "required_n",
]

_MIN_N: Final = 20
_CHUNK: Final = 200


@dataclass(frozen=True)
class RhoEstimate:
    """Correlation between predicted and actual NPS values, with a bootstrap interval."""

    rho: float
    lo: float
    hi: float
    n: int
    conf: float


def _values(
    y: NDArray[np.integer], p: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] != y.shape[0]:
        raise ValueError(f"p must have shape (n, 2) matching y; got {p.shape} for n = {y.size}")
    if y.size < _MIN_N:
        raise ValueError(f"need at least {_MIN_N} past responses, got {y.size}")
    if not np.isfinite(p).all() or not np.isin(y, np.arange(11)).all():
        raise ValueError("y must be integer scores 0..10 and p finite probabilities")
    v = 100.0 * ((y >= 9).astype(np.float64) - (y <= 6).astype(np.float64))
    h = p @ np.array([100.0, -100.0])
    if v.std() == 0 or h.std() == 0:
        raise ValueError("NPS values or predictions are constant; rho is undefined")
    return v, h


def _corr_rows(v: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise Pearson correlation of two (b, n) arrays."""
    vc = v - v.mean(axis=1, keepdims=True)
    hc = h - h.mean(axis=1, keepdims=True)
    num = (vc * hc).sum(axis=1)
    den = np.sqrt((vc**2).sum(axis=1) * (hc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out: NDArray[np.float64] = num / den
    return out


def estimate_rho(
    y: NDArray[np.integer],
    p: NDArray[np.floating],
    *,
    rng: np.random.Generator,
    conf: float = 0.95,
    n_boot: int = 2000,
) -> RhoEstimate:
    """rho = corr(v, h) on past responses, with a percentile bootstrap interval.

    ``y`` are the 0..10 answers, ``p`` the model's [P(promoter), P(detractor)] for the
    same customers (out of sample). Resampling customers keeps the interval honest for
    the discrete NPS values, where Fisher's z interval does not apply.
    """
    v, h = _values(y, p)
    rho = float(np.corrcoef(v, h)[0, 1])
    boots: list[NDArray[np.float64]] = []
    for start in range(0, n_boot, _CHUNK):
        idx = rng.integers(0, v.size, size=(min(_CHUNK, n_boot - start), v.size))
        boots.append(_corr_rows(v[idx], h[idx]))
    b = np.concatenate(boots)
    b = b[np.isfinite(b)]
    a = (1 - conf) / 2
    lo, hi = np.quantile(b, [a, 1 - a])
    return RhoEstimate(rho=rho, lo=float(lo), hi=float(hi), n=int(v.size), conf=conf)


def gain(rho: float, n: float, big_n: float) -> float:
    """Classical responses each response is worth under PPI++ (``inf`` N allowed)."""
    if math.isinf(big_n):
        return 1.0 / (1.0 - rho**2)
    return ppi_pp_theory_ratio(rho, n, big_n)


def required_n(margin: float, *, sd: float, rho: float, big_n: float, alpha: float = 0.1) -> int:
    """Smallest n whose (1 - alpha) interval half-width is at most ``margin`` NPS points.

    Half-width ``z sd sqrt((1 - rho^2 / (1 + n/N)) / n)``; ``rho = 0`` is the classical
    survey. ``sd`` is the standard deviation of the per-customer NPS value (at most 100).
    """
    if margin <= 0 or sd <= 0 or not 0 < alpha < 1 or abs(rho) > 1:
        raise ValueError("need margin > 0, sd > 0, 0 < alpha < 1 and |rho| <= 1")
    z = float(stats.norm.ppf(1 - alpha / 2))

    def half(n: int) -> float:
        return z * sd * math.sqrt((1 - rho**2 / (1 + n / big_n)) / n)

    hi = max(math.ceil((z * sd / margin) ** 2), 1)  # classical; PPI++ never needs more
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if half(mid) <= margin:
            hi = mid
        else:
            lo = mid + 1
    return lo


def planning_table(
    *,
    sd: float,
    rhos: tuple[float, ...],
    margins: tuple[float, ...],
    big_n: float,
    alpha: float = 0.1,
) -> pl.DataFrame:
    """Responses needed for each (rho, margin), with and without the model."""
    rows = []
    for rho in rhos:
        for margin in margins:
            n_c = required_n(margin, sd=sd, rho=0.0, big_n=big_n, alpha=alpha)
            n_p = required_n(margin, sd=sd, rho=rho, big_n=big_n, alpha=alpha)
            rows.append(
                {
                    "rho": float(rho),
                    "margin": float(margin),
                    "n_classical": n_c,
                    "n_ppi_pp": n_p,
                    "responses_saved": n_c - n_p,
                    "saving_pct": 100.0 * (n_c - n_p) / n_c,
                    "gain": gain(rho, n_p, big_n),
                }
            )
    return pl.DataFrame(rows)


@dataclass(frozen=True)
class PlanningReport:
    """rho from past responses and the survey sizes it implies."""

    rho: RhoEstimate
    sd: float
    big_n: float
    alpha: float
    table: pl.DataFrame


def report(
    y: NDArray[np.integer],
    p: NDArray[np.floating],
    *,
    big_n: float,
    margins: tuple[float, ...] = (10.0, 5.0, 3.0),
    alpha: float = 0.1,
    rng: np.random.Generator,
    conf: float = 0.95,
) -> PlanningReport:
    """Estimate rho on past responses and plan survey sizes at its low, point and high values.

    The "low" case uses the end of the rho interval closest to 0 (0 if the interval
    spans 0): the value to plan with.
    """
    v, _ = _values(y, p)
    est = estimate_rho(y, p, rng=rng, conf=conf)
    sd = float(v.std(ddof=1))
    low = 0.0 if est.lo <= 0 <= est.hi else min(abs(est.lo), abs(est.hi))
    cases = {"low": low, "estimate": abs(est.rho), "high": max(abs(est.lo), abs(est.hi))}
    parts = []
    for case, rho in cases.items():
        t = planning_table(sd=sd, rhos=(rho,), margins=margins, big_n=big_n, alpha=alpha)
        parts.append(t.with_columns(pl.lit(case).alias("rho_case")))
    return PlanningReport(rho=est, sd=sd, big_n=big_n, alpha=alpha, table=pl.concat(parts))


def format_report(rep: PlanningReport) -> str:
    """Markdown summary of a :class:`PlanningReport`."""
    r = rep.rho
    lines = [
        f"rho = {r.rho:.3f} ({r.conf:.0%} bootstrap interval {r.lo:.3f} to {r.hi:.3f}) "
        f"from {r.n} past responses; NPS value sd = {rep.sd:.1f}; N = {rep.big_n:,.0f}; "
        f"{1 - rep.alpha:.0%} intervals.",
        "",
        "| rho case | rho | margin (± NPS points) | classical n | PPI++ n | saved | gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rep.table.iter_rows(named=True):
        lines.append(
            f"| {row['rho_case']} | {row['rho']:.3f} | {row['margin']:.1f} | "
            f"{row['n_classical']:,} | {row['n_ppi_pp']:,} | {row['responses_saved']:,} "
            f"({row['saving_pct']:.0f}%) | {row['gain']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def compare_with_results(
    summary: pl.DataFrame, *, rho: dict[str, float], pool: int, estimator: str = "ppi_pp"
) -> pl.DataFrame:
    """Predicted gain (from rho) against the gain an experiment actually realized.

    ``summary`` is an experiment summary (``scenario``, ``estimator``, ``n_labeled``,
    ``effective_n``); realized gain = effective n / n for ``estimator``. The unlabeled
    set is the pool minus the survey, N = pool - n.
    """
    rows = summary.filter(
        (pl.col("estimator") == estimator) & pl.col("scenario").is_in(list(rho))
    ).sort("scenario", "n_labeled")
    out = [
        {
            "scenario": r["scenario"],
            "rho": float(rho[r["scenario"]]),
            "n_labeled": int(r["n_labeled"]),
            "predicted_gain": gain(rho[r["scenario"]], r["n_labeled"], pool - r["n_labeled"]),
            "realized_gain": float(r["effective_n"]) / r["n_labeled"],
        }
        for r in rows.iter_rows(named=True)
    ]
    return pl.DataFrame(out)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """``python -m silent97.eval.planning {table,estimate}``."""
    import argparse

    parser = argparse.ArgumentParser(description="Plan survey sizes from model quality")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="responses needed by rho and margin")
    t.add_argument("--sd", type=float, default=80.0, help="sd of the NPS value (<= 100)")
    t.add_argument("--big-n", type=float, default=1_000_000)
    t.add_argument("--alpha", type=float, default=0.1)
    t.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    t.add_argument("--margins", type=float, nargs="+", default=[10.0, 5.0, 3.0])
    t.add_argument("--out", default=None, help="write Markdown here")
    e = sub.add_parser("estimate", help="estimate rho from past responses (Parquet/CSV)")
    e.add_argument("--data", required=True)
    e.add_argument("--score-col", default="score")
    e.add_argument("--promoter-col", default="p_promoter")
    e.add_argument("--detractor-col", default="p_detractor")
    e.add_argument("--big-n", type=float, required=True, help="customers to report on")
    e.add_argument("--margins", type=float, nargs="+", default=[10.0, 5.0, 3.0])
    e.add_argument("--alpha", type=float, default=0.1)
    e.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "table":
        tab = planning_table(
            sd=args.sd, rhos=tuple(args.rhos), margins=tuple(args.margins),
            big_n=args.big_n, alpha=args.alpha,
        )  # fmt: skip
        lines = [
            f"Responses needed for a {1 - args.alpha:.0%} interval of ± margin NPS points "
            f"(NPS value sd {args.sd:g}, N = {args.big_n:,.0f}). Generated by "
            "`python -m silent97.eval.planning table`.",
            "",
            "| rho | margin | classical n | PPI++ n | saved | gain |",
            "|---|---|---|---|---|---|",
        ]
        for row in tab.iter_rows(named=True):
            lines.append(
                f"| {row['rho']:.1f} | ±{row['margin']:g} | {row['n_classical']:,} | "
                f"{row['n_ppi_pp']:,} | {row['responses_saved']:,} ({row['saving_pct']:.0f}%) "
                f"| {row['gain']:.2f}x |"
            )
        text = "\n".join(lines) + "\n"
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(text)
        print(text)
    else:
        df = (
            pl.read_parquet(args.data) if args.data.endswith(".parquet") else pl.read_csv(args.data)
        )
        y = df[args.score_col].to_numpy()
        p = df.select(args.promoter_col, args.detractor_col).to_numpy()
        rep = report(
            y, p, big_n=args.big_n, margins=tuple(args.margins), alpha=args.alpha,
            rng=np.random.default_rng(args.seed),
        )  # fmt: skip
        print(format_report(rep))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
