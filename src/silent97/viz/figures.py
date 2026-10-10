"""Shared figure style and experiment figures (matplotlib, static PNG).

Style rules (CLAUDE.md section 10 and the project's data-viz method):

- Categorical colours come from a fixed, validated order and follow the estimator,
  never its rank, so a method has the same colour in every figure.
- Every coverage figure draws the nominal line, the Monte Carlo band around it and
  the A1 acceptance band. Axes are labeled with units.
- Marker shapes are a secondary encoding, so colour is never the only cue. Three
  of the light-surface colours are below 3:1 contrast, so every figure ships with a
  table view (the summary CSV/Markdown written next to it).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

__all__ = ["ESTIMATOR_STYLE", "apply_style", "coverage_figure"]

TEXT_PRIMARY: Final = "#0b0b0b"
TEXT_SECONDARY: Final = "#52514e"
GRID: Final = "#e4e3df"
SURFACE: Final = "#fcfcfb"
REFERENCE: Final = "#52514e"

# Fixed categorical order (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6).
ESTIMATOR_STYLE: Final[dict[str, dict[str, Any]]] = {
    "classical": {"color": "#2a78d6", "marker": "o", "label": "classical"},
    "naive_respondent": {"color": "#eb6834", "marker": "s", "label": "naive respondent"},
    "ppi": {"color": "#1baf7a", "marker": "^", "label": "PPI"},
    "ppi_pp": {"color": "#eda100", "marker": "D", "label": "PPI++"},
    "ppi_crossfit": {"color": "#e87ba4", "marker": "v", "label": "cross-fit PPI"},
    "naive_model": {"color": "#008300", "marker": "X", "label": "naive model"},
}


def apply_style() -> None:
    """Set the shared rcParams (recessive grid and axes, neutral text)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # No software/date metadata, so the PNG is reproducible.
    fig.savefig(out, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def coverage_figure(
    summary: pl.DataFrame,
    path: str | Path,
    *,
    nominal: float = 0.9,
    band: tuple[float, float] = (0.88, 0.92),
    coverage_estimators: tuple[str, ...] = (
        "classical",
        "naive_respondent",
        "ppi",
        "ppi_pp",
        "ppi_crossfit",
    ),
    title: str = "",
) -> Path:
    """Coverage (left) and mean interval width (right) against survey size n.

    ``summary`` is the output of :func:`silent97.eval.metrics.summarize` for one
    scenario. Coverage points carry Wilson 95% error bars. The grey band is the
    Monte Carlo band ``nominal -/+ 1.96 * sqrt(nominal (1 - nominal) / R)`` and the
    dashed lines mark the acceptance band. Estimators whose coverage falls far
    below the axis are listed in a note instead of compressing the scale.
    """
    apply_style()
    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ns = sorted(summary["n_labeled"].unique().to_list())
    reps = int(summary["replicates"].to_numpy().min())
    mc = 1.96 * math.sqrt(nominal * (1 - nominal) / reps)

    ax_c.axhspan(
        nominal - mc,
        nominal + mc,
        color=GRID,
        alpha=0.8,
        lw=0,
        label=f"Monte Carlo band (R = {reps})",
    )
    ax_c.axhline(nominal, color=REFERENCE, lw=1.2, label=f"nominal {nominal:.0%}")
    for j, edge in enumerate(band):
        ax_c.axhline(
            edge, color=REFERENCE, lw=0.8, ls="--",
            label=f"A1 band {band[0]:.0%}-{band[1]:.0%}" if j == 0 else None,
        )  # fmt: skip
    k = len(coverage_estimators)
    for i, name in enumerate(coverage_estimators):
        rows = summary.filter(pl.col("estimator") == name).sort("n_labeled")
        if rows.is_empty():
            continue
        style = ESTIMATOR_STYLE[name]
        dodge = 1 + 0.035 * (i - (k - 1) / 2)  # multiplicative offset on the log axis
        x = [n * dodge for n in rows["n_labeled"]]
        cov = rows["coverage"].to_numpy()
        err = [cov - rows["coverage_lo"].to_numpy(), rows["coverage_hi"].to_numpy() - cov]
        ax_c.errorbar(
            x, cov, yerr=err, color=style["color"], marker=style["marker"], lw=1.4,
            capsize=2.5, label=style["label"], markeredgecolor=SURFACE, markeredgewidth=0.8,
        )  # fmt: skip
    off_scale = [
        f"{ESTIMATOR_STYLE[e]['label']}: {c:.0%}"
        for e, c in summary.filter(~pl.col("estimator").is_in(coverage_estimators))
        .group_by("estimator")
        .agg(pl.col("coverage").max())
        .iter_rows()
    ]
    ax_c.set_xscale("log")
    ax_c.set_xticks(ns, [str(n) for n in ns])
    ax_c.minorticks_off()
    ax_c.set_ylim(0.80, 1.0)
    ax_c.set_yticks([0.80 + 0.02 * i for i in range(11)])
    ax_c.set_xlabel("surveyed customers n (log scale)")
    ax_c.set_ylabel("empirical coverage (share of replicates)")
    ax_c.set_title("Coverage of nominal 90% intervals")
    ax_c.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if off_scale:
        ax_c.text(
            0.02, 0.03, "Off scale (max over n): " + "; ".join(off_scale),
            transform=ax_c.transAxes, color=TEXT_SECONDARY, fontsize=8.5,
        )  # fmt: skip

    for name, style in ESTIMATOR_STYLE.items():
        rows = summary.filter(pl.col("estimator") == name).sort("n_labeled")
        if rows.is_empty():
            continue
        ax_w.plot(
            rows["n_labeled"], rows["mean_width"], color=style["color"], marker=style["marker"],
            label=style["label"], markeredgecolor=SURFACE, markeredgewidth=0.8,
        )  # fmt: skip
    ax_w.set_xscale("log")
    ax_w.set_yscale("log")
    ax_w.set_xticks(ns, [str(n) for n in ns])
    ax_w.minorticks_off()
    lo_w = float(summary["mean_width"].to_numpy().min())
    hi_w = float(summary["mean_width"].to_numpy().max())
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100) if lo_w / 1.5 <= t <= hi_w * 1.5]
    ax_w.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax_w.set_xlabel("surveyed customers n (log scale)")
    ax_w.set_ylabel("mean interval width (NPS points, log scale)")
    ax_w.set_title("Interval width")
    # One shared legend below both panels: estimators (from the width panel, which
    # has all of them) followed by the coverage reference marks.
    handles, labels = ax_w.get_legend_handles_labels()
    ref_h, ref_l = ax_c.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(ref_l) if lab.startswith(("Monte", "nominal", "A1"))]
    fig.legend(
        handles + [ref_h[i] for i in keep], labels + [ref_l[i] for i in keep],
        loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01),
    )  # fmt: skip
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, path)
