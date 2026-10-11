"""Evaluation metrics over replicate results.

Per (scenario, estimator, n): coverage with a Wilson 95% interval, mean interval
width, bias, RMSE, mean lambda, effective sample size and responses saved.

Effective n follows docs/PLAN.md: the classical-survey n that gives the same mean
width, ``n_eff = n * (mean_width_classical / mean_width_method) ** 2``.
Responses saved is ``n_eff - n``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import norm

__all__ = [
    "TARGET_BAND",
    "effective_n_bootstrap",
    "overlaps_target",
    "ppi_pp_theory_ratio",
    "summarize",
    "wilson_interval",
]

TARGET_BAND = (0.88, 0.92)
"""Acceptance band for nominal 90% coverage (A1)."""


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k / n."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    z = float(norm.ppf(0.5 + conf / 2))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def overlaps_target(lo: float, hi: float, band: tuple[float, float] = TARGET_BAND) -> bool:
    """Whether the interval [lo, hi] overlaps the acceptance band."""
    return lo <= band[1] and hi >= band[0]


def summarize(results: pl.DataFrame, *, reference: str = "classical") -> pl.DataFrame:
    """Summarize replicate rows (see :data:`silent97.eval.harness.RESULT_COLUMNS`)."""
    keys = ["scenario", "estimator", "n_labeled"]
    err = pl.col("point") - pl.col("truth")
    s = results.group_by(keys).agg(
        pl.len().alias("replicates"),
        pl.col("covered").sum().alias("n_covered"),
        pl.col("covered").mean().alias("coverage"),
        pl.col("width").mean().alias("mean_width"),
        err.mean().alias("bias"),
        (err**2).mean().sqrt().alias("rmse"),
        pl.col("se").mean().alias("mean_se"),
        pl.col("lambda_hat").fill_nan(None).mean().alias("mean_lambda"),
    )
    bounds = [
        wilson_interval(int(k), int(r))
        for k, r in zip(s["n_covered"], s["replicates"], strict=True)
    ]
    s = s.with_columns(
        pl.Series("coverage_lo", [b[0] for b in bounds]),
        pl.Series("coverage_hi", [b[1] for b in bounds]),
    )
    s = s.with_columns(
        pl.struct("coverage_lo", "coverage_hi")
        .map_elements(
            lambda r: overlaps_target(r["coverage_lo"], r["coverage_hi"]), return_dtype=pl.Boolean
        )
        .alias("coverage_ok")
    )
    ref = s.filter(pl.col("estimator") == reference).select(
        "scenario", "n_labeled", pl.col("mean_width").alias("_ref_width")
    )
    s = s.join(ref, on=["scenario", "n_labeled"], how="left")
    s = s.with_columns(
        (pl.col("n_labeled") * (pl.col("_ref_width") / pl.col("mean_width")) ** 2).alias(
            "effective_n"
        )
    ).with_columns((pl.col("effective_n") - pl.col("n_labeled")).alias("responses_saved"))
    return s.drop("_ref_width").sort(keys)


def ppi_pp_theory_ratio(rho: float, n: float, big_n: float) -> float:
    """Theoretical effective n / n for PPI++ with the optimal lambda.

    With v the per-customer NPS value and h the predicted one,
    Var(PPI++) / Var(classical) = 1 - rho^2 / (1 + n/N) at
    lambda* = Cov(v, h) / ((1 + n/N) Var(h)), so effective n / n is its inverse.
    For N >> n this is 1 / (1 - rho^2).
    """
    return 1.0 / (1.0 - rho**2 / (1.0 + n / big_n))


def effective_n_bootstrap(
    results: pl.DataFrame,
    *,
    reference: str = "classical",
    n_boot: int = 2000,
    rng: np.random.Generator,
    conf: float = 0.95,
) -> pl.DataFrame:
    """Effective n / n per (scenario, estimator, n) with a paired bootstrap interval.

    ``ratio = (mean width_reference / mean width_method) ** 2``, the docs/PLAN.md
    definition divided by n. Replicates are resampled jointly across estimators,
    because every estimator in a replicate sees the same survey draw.
    """
    rows = []
    lo_q, hi_q = (1 - conf) / 2, 1 - (1 - conf) / 2
    for (scenario, n), part in results.group_by(["scenario", "n_labeled"], maintain_order=True):
        wide = part.pivot(on="estimator", index="replicate", values="width").sort("replicate")
        ref = wide[reference].to_numpy()
        idx = rng.integers(0, ref.size, size=(n_boot, ref.size))
        ref_boot = ref[idx].mean(axis=1)
        for est in (c for c in wide.columns if c != "replicate"):
            w = wide[est].to_numpy()
            ratio = (ref.mean() / w.mean()) ** 2
            boot = (ref_boot / w[idx].mean(axis=1)) ** 2
            rows.append(
                {
                    "scenario": scenario,
                    "n_labeled": n,
                    "estimator": est,
                    "ratio": float(ratio),
                    "ratio_lo": float(np.quantile(boot, lo_q)),
                    "ratio_hi": float(np.quantile(boot, hi_q)),
                }
            )
    return pl.DataFrame(rows).sort(["scenario", "n_labeled", "estimator"])
