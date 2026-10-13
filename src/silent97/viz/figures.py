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
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

__all__ = [
    "DESIGN_STYLE",
    "ESTIMATOR_STYLE",
    "apply_style",
    "coverage_figure",
    "design_figure",
    "efficiency_figure",
    "segment_figure",
    "sensitivity_figure",
    "trap_figure",
]

TEXT_PRIMARY: Final = "#0b0b0b"
TEXT_SECONDARY: Final = "#52514e"
GRID: Final = "#e4e3df"
SURFACE: Final = "#fcfcfb"
REFERENCE: Final = "#52514e"

# Fixed categorical order, slots 1-5 of the reference palette (validated: adjacent
# CVD dE >= 9.1, normal-vision dE >= 19.6).
ESTIMATOR_STYLE: Final[dict[str, dict[str, Any]]] = {
    "classical": {"color": "#2a78d6", "marker": "o", "label": "classical"},
    "ppi": {"color": "#eb6834", "marker": "^", "label": "PPI"},
    "ppi_pp": {"color": "#1baf7a", "marker": "D", "label": "PPI++"},
    "ppi_crossfit": {"color": "#eda100", "marker": "v", "label": "cross-fit PPI"},
    "naive_model": {"color": "#e87ba4", "marker": "X", "label": "naive model"},
    "ppi_pp_cf": {"color": "#7a5cc9", "marker": "s", "label": "PPI++ (cross-fit λ)"},
    "ppi_active": {"color": "#0f8a8a", "marker": "P", "label": "active PPI (1/π)"},
    "ppi_stratified": {"color": "#8a5a2b", "marker": "h", "label": "stratified PPI"},
    "mrp": {"color": "#c0392b", "marker": "*", "label": "MRP"},
    "mrp_ppi": {"color": "#5b6b7a", "marker": "p", "label": "MRP + predictions"},
}

# E5 segment methods: "raw" is the classical segment mean.
SEGMENT_STYLE: Final[dict[str, dict[str, Any]]] = {
    "raw": {**ESTIMATOR_STYLE["classical"], "label": "raw segment mean"},
    "ppi": {**ESTIMATOR_STYLE["ppi"], "label": "PPI within segment"},
    "mrp": ESTIMATOR_STYLE["mrp"],
    "mrp_ppi": ESTIMATOR_STYLE["mrp_ppi"],
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
        "ppi",
        "ppi_pp",
        "ppi_crossfit",
    ),
    title: str = "",
    width_label: str = "mean interval width (NPS points, log scale)",
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
    ax_w.set_ylabel(width_label)
    ax_w.set_title("Interval width")
    # One shared legend below both panels: estimators (from the width panel, which
    # has all of them) followed by the coverage reference marks.
    handles, labels = ax_w.get_legend_handles_labels()
    ref_h, ref_l = ax_c.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(ref_l) if lab.startswith(("Monte", "nominal", "A1"))]
    fig.legend(
        handles + [ref_h[i] for i in keep], labels + [ref_l[i] for i in keep],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
    )  # fmt: skip
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, path)


_EFFICIENCY_METRICS: dict[str, tuple[str, str]] = {
    "mse": ("mse_ratio", "MSE ratio, classical / method (classical responses per response)"),
    "widths": ("ratio", "effective n / n from widths (classical responses per response)"),
}


def efficiency_figure(
    table: pl.DataFrame,
    path: str | Path,
    *,
    theory: dict[int, Any],
    threshold: tuple[float, float] = (0.6, 1.5),
    estimators: tuple[str, ...] = ("ppi", "ppi_pp"),
    metric: str = "mse",
    title: str = "",
) -> Path:
    """Efficiency against model-outcome correlation, one panel per survey size n.

    ``table`` has columns ``n_labeled``, ``estimator``, ``model_corr`` and, for
    ``metric="mse"``, ``mse_ratio``/``mse_ratio_lo``/``mse_ratio_hi`` (MSE_classical /
    MSE_method, the measure A3 is judged on) or, for ``metric="widths"``,
    ``ratio``/``ratio_lo``/``ratio_hi`` ((classical width / method width)²), each with a
    paired-bootstrap 95% interval. ``theory[n]`` is a callable rho -> ratio drawn as the
    reference curve. The dashed segment marks the A3 requirement (ratio >= threshold[1]
    for rho >= threshold[0]); it is drawn on the MSE figure only.
    """
    if metric not in _EFFICIENCY_METRICS:
        raise ValueError(f"metric must be one of {sorted(_EFFICIENCY_METRICS)}, got {metric!r}")
    col, ylabel = _EFFICIENCY_METRICS[metric]
    apply_style()
    ns = sorted(table["n_labeled"].unique().to_list())
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 4.4), sharey=True, squeeze=False)
    rho_max = float(table["model_corr"].to_numpy().max())
    grid = [i * (rho_max + 0.05) / 200 for i in range(201)]
    for ax, n in zip(axes[0], ns, strict=True):
        curve = theory[n]
        ax.plot(grid, [curve(r) for r in grid], color=REFERENCE, lw=1.2,
                label="theory, PPI++: 1 / (1 - ρ² / (1 + n/N))")  # fmt: skip
        if metric == "mse":
            ax.plot([threshold[0], rho_max + 0.05], [threshold[1]] * 2, color=REFERENCE, lw=0.8,
                    ls="--", label=f"A3: ≥ {threshold[1]}× when ρ ≥ {threshold[0]}")  # fmt: skip
        ax.axhline(1.0, color=GRID, lw=1.0)
        k = len(estimators)
        for i, name in enumerate(estimators):
            rows = table.filter((pl.col("n_labeled") == n) & (pl.col("estimator") == name))
            rows = rows.sort("model_corr")
            if rows.is_empty():
                continue
            style = ESTIMATOR_STYLE[name]
            x = rows["model_corr"].to_numpy() + 0.014 * (i - (k - 1) / 2)
            y = rows[col].to_numpy()
            err = [y - rows[f"{col}_lo"].to_numpy(), rows[f"{col}_hi"].to_numpy() - y]
            # The first estimator is drawn hollow so near-identical points stay visible.
            face = "none" if i == 0 and k > 1 else style["color"]
            edge = style["color"] if i == 0 and k > 1 else SURFACE
            ax.errorbar(x, y, yerr=err, color=style["color"], marker=style["marker"], lw=0,
                        elinewidth=1.4, capsize=2.5, label=style["label"], markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=1.2)  # fmt: skip
        ax.set_title(f"n = {n} surveyed customers")
        ax.set_xlabel("model-outcome correlation ρ (predicted vs true NPS value)")
        ax.set_xlim(-0.02, rho_max + 0.05)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


TRAP_ESTIMATORS: Final = ("naive_model", "classical", "ppi_pp")


def trap_figure(
    summary: pl.DataFrame,
    path: str | Path,
    *,
    n: int,
    truth: float,
    value_label: str,
    estimators: tuple[str, ...] = TRAP_ESTIMATORS,
    title: str = "",
) -> Path:
    """The model-only trap at one survey size: average intervals against the truth.

    One row per (labeler, estimator): the dot is the mean point estimate (truth + bias)
    and the bar spans the mean interval width around it. The coverage is printed at the
    right. ``summary`` needs ``scenario``, ``estimator``, ``n_labeled``, ``bias``,
    ``mean_width`` and ``coverage``.
    """
    rows = summary.filter((pl.col("n_labeled") == n) & pl.col("estimator").is_in(list(estimators)))
    if rows.is_empty():
        raise ValueError(f"no rows for n = {n}")
    apply_style()
    scenarios = sorted(rows["scenario"].unique().to_list())
    labels: list[str] = []
    ypos: list[float] = []
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(scenarios) * len(estimators) + 1.4))
    y = 0.0
    for scenario in scenarios:
        for est in estimators:
            r = rows.filter((pl.col("scenario") == scenario) & (pl.col("estimator") == est))
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            point, half = truth + row["bias"], row["mean_width"] / 2
            style = ESTIMATOR_STYLE[est]
            ax.errorbar([point], [y], xerr=[[half], [half]], color=style["color"],
                        marker=style["marker"], lw=0, elinewidth=3, capsize=0)  # fmt: skip
            ax.annotate(f"coverage {row['coverage']:.0%}", (1.0, y), xycoords=("axes fraction",
                        "data"), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=TEXT_SECONDARY, annotation_clip=False)  # fmt: skip
            labels.append(f"{scenario}: {style['label']}")
            ypos.append(y)
            y += 1
        y += 0.5
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.axvline(truth, color=REFERENCE, lw=1.2, ls="--")
    ax.annotate("truth", (truth, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
    ax.set_xlabel(f"{value_label}: mean estimate and mean interval (n = {n} surveyed)")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


DESIGN_STYLE: Final[dict[str, dict[str, Any]]] = {
    "uniform": {"color": "#2a78d6", "marker": "o", "label": "uniform → PPI"},
    "stratified_proportional": {
        "color": "#eda100", "marker": "v", "label": "stratified (proportional) → stratified PPI"
    },
    "stratified_neyman": {
        "color": "#8a5a2b", "marker": "h", "label": "stratified (Neyman) → stratified PPI"
    },
    "active": {"color": "#0f8a8a", "marker": "P", "label": "active (1/π) → active PPI"},
}  # fmt: skip


def design_figure(
    table: pl.DataFrame, path: str | Path, *, nominal: float, title: str = ""
) -> Path:
    """E3: mean width and coverage against n, one line per design, one row per signal.

    ``table`` has ``signal``, ``model_corr``, ``n_labeled``, ``design``, ``mean_width``,
    ``coverage``, ``coverage_lo`` and ``coverage_hi``. The coverage panels draw the nominal
    line, the A1 band and Wilson 95% bars.
    """
    apply_style()
    signals = sorted(table["signal"].unique().to_list())
    fig, axes = plt.subplots(len(signals), 2, figsize=(11, 3.9 * len(signals)), squeeze=False)
    for row, sig in zip(axes, signals, strict=True):
        part = table.filter(pl.col("signal") == sig)
        rho = float(part["model_corr"][0])
        ax_w, ax_c = row
        ax_c.axhspan(0.88, 0.92, color=GRID, alpha=0.6, lw=0, label="A1 band 88%-92%")
        ax_c.axhline(nominal, color=REFERENCE, lw=1.2, label=f"nominal {nominal:.0%}")
        for design, style in DESIGN_STYLE.items():
            d = part.filter(pl.col("design") == design).sort("n_labeled")
            if d.is_empty():
                continue
            n = d["n_labeled"].to_numpy()
            ax_w.plot(n, d["mean_width"].to_numpy(), color=style["color"],
                      marker=style["marker"], label=style["label"])  # fmt: skip
            cov = d["coverage"].to_numpy()
            err = [cov - d["coverage_lo"].to_numpy(), d["coverage_hi"].to_numpy() - cov]
            ax_c.errorbar(n, cov, yerr=err, color=style["color"], marker=style["marker"],
                          capsize=2.5, lw=1.4)  # fmt: skip
        for ax in (ax_w, ax_c):
            ax.set_xscale("log")
            ax.set_xlabel("survey budget n (log scale)")
        ax_w.set_yscale("log")
        ax_w.set_ylabel("mean interval width (NPS points, log scale)")
        ax_w.set_title(f"Interval width, signal {sig:.2f} (ρ = {rho:.2f})")
        ax_c.set_ylabel("empirical coverage")
        ax_c.set_ylim(0.8, 1.0)
        ax_c.set_title(f"Coverage of nominal {nominal:.0%} intervals (Wilson 95% bars)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[0][1].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, path)


def segment_figure(
    table: pl.DataFrame,
    path: str | Path,
    *,
    bins: list[str],
    nominal: float,
    title: str = "",
) -> Path:
    """E5: segment RMSE and coverage by number of respondents in the segment.

    ``table`` has ``bin``, ``method``, ``rmse``, ``coverage``, ``coverage_lo`` and
    ``coverage_hi``; ``bins`` gives the x-axis order. The coverage panel draws the
    nominal line, the A1 band and Wilson 95% bars.
    """
    apply_style()
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_c.axhspan(0.88, 0.92, color=GRID, alpha=0.6, lw=0, label="A1 band 88%-92%")
    ax_c.axhline(nominal, color=REFERENCE, lw=1.2, label=f"nominal {nominal:.0%}")
    pos = {b: i for i, b in enumerate(bins)}
    offsets = {m: (k - 1.5) * 0.08 for k, m in enumerate(SEGMENT_STYLE)}
    for method, style in SEGMENT_STYLE.items():
        d = table.filter(pl.col("method") == method)
        d = d.filter(pl.col("bin").is_in(bins))
        if d.is_empty():
            continue
        x = [pos[b] for b in d["bin"].to_list()]
        order = sorted(range(len(x)), key=x.__getitem__)
        xs = [x[i] for i in order]
        rmse = d["rmse"].to_numpy()[order]
        cov = d["coverage"].to_numpy()[order]
        lo = d["coverage_lo"].to_numpy()[order]
        hi = d["coverage_hi"].to_numpy()[order]
        ax_r.plot(xs, rmse, color=style["color"], marker=style["marker"], label=style["label"])
        yerr = [np.clip(cov - lo, 0, None), np.clip(hi - cov, 0, None)]
        shifted = [v + offsets[method] for v in xs]
        ax_c.errorbar(shifted, cov, yerr=yerr, color=style["color"], marker=style["marker"],
                      capsize=2.5, lw=1.4)  # fmt: skip
    for ax in (ax_r, ax_c):
        ax.set_xticks(range(len(bins)), bins)
        ax.set_xlabel("respondents in segment")
    ax_r.set_ylabel("RMSE vs true segment NPS (NPS points)")
    ax_r.set_title("Segment estimate error")
    ax_c.set_ylabel("empirical coverage")
    ax_c.set_ylim(0.5, 1.0)
    ax_c.set_title(f"Coverage of nominal {nominal:.0%} intervals (Wilson 95% bars)")
    handles, labels = ax_r.get_legend_handles_labels()
    h2, l2 = ax_c.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, path)


def sensitivity_figure(
    curve: pl.DataFrame,
    table: pl.DataFrame,
    path: str | Path,
    *,
    marks: dict[str, dict[str, float]],
    threshold: float,
    nominal: float,
    title: str = "",
) -> Path:
    """E7: one survey's sensitivity curve, breakpoints by n, and coverage by n.

    ``curve`` has ``estimator``, ``delta``, ``point``, ``lower``, ``upper`` and
    ``truth``; ``marks`` gives each curve estimator's ``delta_true`` and ``delta_star``.
    ``table`` has ``scenario``, ``estimator``, ``n_labeled``, ``median_delta_star``,
    ``coverage_at_zero`` and ``coverage_at_true`` with their Wilson bounds. Coverage
    panels draw the nominal line, the A1 band and Wilson 95% bars: hollow markers at
    δ = 0 (the base interval), filled at δ_true.
    """
    apply_style()
    scenarios = sorted(table["scenario"].unique().to_list())
    n_panels = 2 + len(scenarios)
    n_rows = math.ceil(n_panels / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.1 * n_rows), squeeze=False)
    flat = axes.ravel()
    ax = flat[0]
    truth = float(curve["truth"][0])
    ax.axhline(truth, color=TEXT_PRIMARY, lw=1.2, ls="--", label=f"true NPS {truth:+.1f}")
    ax.axhline(threshold, color=REFERENCE, lw=1.0, ls=":", label=f"threshold {threshold:+g}")
    for est in curve["estimator"].unique(maintain_order=True).to_list():
        style = ESTIMATOR_STYLE[est]
        c = curve.filter(pl.col("estimator") == est)
        dx = c["delta"].to_numpy()
        ax.fill_between(dx, c["lower"].to_numpy(), c["upper"].to_numpy(), color=style["color"],
                        alpha=0.15, lw=0)  # fmt: skip
        ax.plot(dx, c["point"].to_numpy(), color=style["color"], label=style["label"])
        m = marks.get(est, {})
        if "delta_true" in m:
            ax.axvline(m["delta_true"], color=style["color"], lw=1.0, ls="--")
        if m.get("delta_star", 0.0) != 0.0 and math.isfinite(m["delta_star"]):
            ax.plot([m["delta_star"]], [threshold], color=style["color"], marker="o", ms=7)
    ax.set_xlabel("δ (pp): non-respondents' promoter share +δ, detractor share −δ")
    ax.set_ylabel("NPS (points)")
    ax.set_title("One survey: estimate and 90% interval vs δ\n(dashed: δ_true; dot: δ*)")
    ax.legend(loc="upper left", fontsize=8)

    ax = flat[1]
    for i, scen in enumerate(scenarios):
        part = table.filter(pl.col("scenario") == scen)
        for est in part["estimator"].unique(maintain_order=True).to_list():
            style = ESTIMATOR_STYLE[est]
            d = part.filter(pl.col("estimator") == est).sort("n_labeled")
            ax.plot(d["n_labeled"].to_numpy(), d["median_delta_star"].to_numpy(),
                    color=style["color"], marker=style["marker"], ls=("-", "--", ":")[i % 3],
                    label=f"{style['label']} ({scen})")  # fmt: skip
    ax.axhline(0, color=REFERENCE, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("survey responses n (log scale)")
    ax.set_ylabel("median δ* (pp)")
    ax.set_title(f'Breakpoint of "NPS > {threshold:+g}"')
    ax.legend(fontsize=7)

    for ax, scen in zip(flat[2:], scenarios, strict=False):
        ax.axhspan(0.88, 0.92, color=GRID, alpha=0.6, lw=0, label="A1 band 88%-92%")
        ax.axhline(nominal, color=REFERENCE, lw=1.2, label=f"nominal {nominal:.0%}")
        part = table.filter(pl.col("scenario") == scen)
        for est in part["estimator"].unique(maintain_order=True).to_list():
            style = ESTIMATOR_STYLE[est]
            d = part.filter(pl.col("estimator") == est).sort("n_labeled")
            n = d["n_labeled"].to_numpy().astype(float)
            for col, fill, shift in (("coverage_at_zero", "none", 0.97),
                                     ("coverage_at_true", style["color"], 1.03)):  # fmt: skip
                cov = d[col].to_numpy()
                yerr = [np.clip(cov - d[f"{col}_lo"].to_numpy(), 0, None),
                        np.clip(d[f"{col}_hi"].to_numpy() - cov, 0, None)]  # fmt: skip
                ax.errorbar(n * shift, cov, yerr=yerr, color=style["color"],
                            marker=style["marker"], mfc=fill, capsize=2.5, lw=1.2,
                            ls="--" if fill == "none" else "-")  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("survey responses n (log scale)")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"{scen}: coverage at δ = 0 (hollow) and at δ_true (filled)")
        ax.legend(loc="lower left", fontsize=8)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)
