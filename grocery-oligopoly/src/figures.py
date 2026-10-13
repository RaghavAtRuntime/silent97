"""Publication figures for the memo and deck (PNG, 200 dpi) from outputs/tables/*.csv."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from config import FIGURES, TABLES

# Palette (validated categorical slots; light surface). Same parent = same hue family.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BANNER_COLORS = {
    "Loblaws": "#2a78d6",
    "No Frills": "#184f95",
    "T&T": "#86b6ef",
    "Metro": "#eb6834",
    "Voilà": "#1baf7a",
    "Walmart": "#6250d6",
    "Galleria": "#e87ba4",
    "Save-On-Foods": "#8a8984",
}
GROUP_COLORS = {
    "Same parent (Loblaw)": "#2a78d6",
    "Rival parents, Toronto": "#eb6834",
    "Out-of-market control": "#8a8984",
    "Independent specialty": "#e87ba4",
}
SOURCE = "Source: Project Hammer (J. Filipp), Toronto online shelf prices, Feb 2024–Oct 2026; analysis by author."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def finish(fig: plt.Figure, name: str, title: str, subtitle: str | None = None, source: str = SOURCE) -> None:
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=10, color=INK2)
    fig.text(0.01, -0.01, source, ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def pct(x: float, _pos: int | None = None) -> str:
    return f"{x:+.0%}" if x != 0 else "0%"


# ---------------------------------------------------------------------------
def fig_price_ladder() -> None:
    d = t("03b_banner_index_overall").sort_values("regular_index")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.78, bottom=0.2)
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        c = BANNER_COLORS[r.banner]
        sh = r.shelf_vs_regular_base
        ax.plot([sh, r.regular_index], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter(r.regular_index, i, s=110, color=c, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter(sh, i, s=110, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=3)
        ax.text(r.regular_index + 1.0, i, f"{r.regular_index:.0f} regular · {sh:.0f} after promotions",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{b}\n({'Loblaw' if b in ('Loblaws', 'No Frills') else 'Metro Inc.' if b == 'Metro' else 'Walmart Canada'})"
                      for b in d["banner"]])
    ax.axvline(100, color=MUTED, lw=1, ls="--")
    ax.text(100.3, len(d) - 0.55, "4-banner regular-price average = 100", fontsize=8, color=MUTED)
    ax.set_xlim(80, 128)
    ax.set_xlabel("Price index on an identical basket (geometric mean; 4-banner regular-price average = 100)")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=60, color=INK2, label="Regular price")
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK2, label="Shelf price incl. promotions")
    ax.legend(loc="lower right", fontsize=8.5)
    finish(fig, "fig01_price_ladder", "One basket, two price tiers — and Loblaw owns a rung in each",
           f"~{d['avg_basket_items'].mean():,.0f} identical UPCs stocked by all four banners each month, Aug 2024–Oct 2026")


def fig_identical_gaps() -> None:
    d = t("02_identical_item_gaps")
    keep = [("Metro", "Walmart"), ("Loblaws", "Walmart"), ("Loblaws", "No Frills"), ("No Frills", "Walmart"),
            ("Loblaws", "Metro"), ("Metro", "Voilà"), ("Loblaws", "Voilà"), ("No Frills", "Metro")]
    d = d.set_index(["banner_a", "banner_b"]).loc[keep].reset_index()
    d["pair"] = d["banner_a"] + " vs " + d["banner_b"]
    d = d.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(top=0.8, bottom=0.17)
    cols = [GROUP_COLORS[g] for g in d["group"]]
    ax = axes[0]
    ax.barh(d["pair"], d["median_regular_premium_a_vs_b"], color=cols, height=0.62)
    for i, v in enumerate(d["median_regular_premium_a_vs_b"]):
        ax.text(v + (0.006 if v >= 0 else -0.006), i, pct(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlim(-0.24, 0.32)
    ax.set_title("Median regular-price premium (1st vs 2nd banner)", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(d["pair"], d["share_days_identical_regular"], color=cols, height=0.62)
    for i, v in enumerate(d["share_days_identical_regular"]):
        ax.text(v + 0.008, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, 0.5)
    ax.set_title("Item-days at the identical regular price", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in ("Same parent (Loblaw)", "Rival parents, Toronto")]
    fig.legend(handles, ["Same parent (Loblaw)", "Rival parents"], loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.94),
               fontsize=9)
    finish(fig, "fig02_identical_item_gaps", "Same product, same day: conventional banners match each other; discount banners undercut",
           "Matched on identical SKU (Loblaw banners) or UPC (across owners); regular (non-promotional) prices")


def fig_loblaw_siblings() -> None:
    d = t("02c_gap_by_month")
    d = d[(d.banner_a == "Loblaws") & (d.banner_b == "No Frills")].copy()
    d["month"] = pd.to_datetime(d["month"])
    # Feb-Jul 2024 was a small fixed basket in the scrape; start at the full extract
    d = d[(d["month"] >= "2024-07-01") & (d["month"] < "2026-10-01")]
    cat = t("02b_gap_by_category_brand")
    cat = cat[(cat.banner_a == "Loblaws") & (cat.banner_b == "No Frills") & (cat.category != "Other")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.35})
    fig.subplots_adjust(top=0.8, bottom=0.18)
    ax = axes[0]
    ax.plot(d["month"], d["premium"], color=BANNER_COLORS["Loblaws"], marker="o", ms=4)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_ylim(0, 0.16)
    ax.set_title("Median Loblaws premium over No Frills, identical SKU", fontsize=10.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    last = d.iloc[-1]
    ax.annotate(f"{last['premium']:+.1%}", (last["month"], last["premium"]), xytext=(-10, 10), textcoords="offset points",
                fontsize=9, color=INK, ha="right")
    ax = axes[1]
    piv = cat.pivot(index="category", columns="private_label", values="median_regular_premium").dropna()
    piv = piv.sort_values(False)
    y = np.arange(len(piv))
    ax.barh(y + 0.2, piv[False], height=0.38, color=BANNER_COLORS["Loblaws"], label="National brands")
    ax.barh(y - 0.2, piv[True], height=0.38, color=BANNER_COLORS["T&T"], label="Loblaw private label (PC, No Name)")
    ax.set_yticks(y, piv.index)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("Premium by category and brand type", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2)
    finish(fig, "fig03_loblaw_banner_premium", "Inside one portfolio: the same item costs ~11% more under the Loblaws banner",
           "The premium is stable since the full scrape began (Jul 2024) and present in every category, including Loblaw's own brands")


def fig_sync() -> None:
    d = t("04_synchronisation")
    # average the two directions per pair
    d["pair"] = d.apply(lambda r: " – ".join(sorted([r.banner_a, r.banner_b])), axis=1)
    g = d.groupby(["pair", "pair_group"], as_index=False).agg(
        lift=("lift_1d", "mean"), events=("events_a", "sum"), obs=("observed_1d", "mean"), plc=("placebo_1d", "mean"))
    # approximate 95% CI on the observed rate (normal approx), carried to the lift
    se = np.sqrt(g["obs"] * (1 - g["obs"]) / g["events"])
    g["lo"] = (g["obs"] - 1.96 * se) / g["plc"]
    g["hi"] = (g["obs"] + 1.96 * se) / g["plc"]
    order = ["Same parent (Loblaw)", "Rival parents, Toronto", "Out-of-market control", "Independent specialty"]
    g["o"] = g["pair_group"].map(order.index)
    g = g.sort_values(["o", "lift"], ascending=[False, True])  # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.27)
    y = np.arange(len(g))
    ax.barh(y, g["lift"] - 1, left=1, color=[GROUP_COLORS[x] for x in g["pair_group"]], height=0.6)
    ax.errorbar(g["lift"], y, xerr=[g["lift"] - g["lo"], g["hi"] - g["lift"]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    for i, r in enumerate(g.itertuples()):
        ax.text(max(r.hi, 1) + 0.04, i, f"{r.lift:.2f}×", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, g["pair"])
    ax.axvline(1, color=INK, lw=1)
    ax.text(1.01, len(g) - 0.4, "chance (placebo) = 1.0×", fontsize=8, color=INK2)
    ax.set_xlim(0.5, 3.0)
    ax.set_xlabel("Synchronisation lift: share of regular-price changes matched by the other banner within ±1 day,\n"
                  "relative to a time-shifted placebo (same items, dates moved ±2–6 months)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[k]) for k in order]
    ax.legend(handles, order, loc="lower right", fontsize=8.5)
    finish(fig, "fig04_synchronisation", "Price changes cluster inside the Loblaw portfolio; across owners, no more than an out-of-market control",
           "Rival-owner synchronisation is no higher than with Save-On-Foods, a Western grocer outside the Toronto market")


def fig_freeze() -> None:
    d = t("06_weekly_change_rates")
    d["week"] = pd.to_datetime(d["week"])
    banners = ["Loblaws", "No Frills", "Metro", "Walmart"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.08})
    fig.subplots_adjust(top=0.84, bottom=0.1)
    for ax, b in zip(axes.flat, banners):
        g = d[d.banner == b].sort_values("week")
        for yr in (2024, 2025):
            ax.axvspan(pd.Timestamp(f"{yr}-11-01"), pd.Timestamp(f"{yr + 1}-02-05"), color=GRID, alpha=0.9, lw=0)
        ax.plot(g["week"], g["ups_per_1000"], color=BANNER_COLORS[b], lw=1.6)
        w = g["freeze_window"]
        ups_in = g.loc[w, "ups"].sum() / g.loc[w, "active"].sum()
        ups_out = g.loc[~w, "ups"].sum() / g.loc[~w, "active"].sum()
        ax.set_title(f"{b}: in-window vs outside rate ratio {ups_in / ups_out:.2f}", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_ylim(0, 130)
        peaks = g[g["ups_per_1000"] > 130]
        for _, pk in peaks.iterrows():
            ax.text(pk["week"], 127, f"{pk['ups_per_1000']:.0f}", fontsize=7, color=INK2, ha="center", va="top")
    axes[0, 0].text(pd.Timestamp("2024-11-05"), axes[0, 0].get_ylim()[1] * 0.9, "Nov 1–Feb 5\n“freeze” window",
                    fontsize=8, color=INK2, va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("Regular-price increases\nper 1,000 products / week")
    finish(fig, "fig05_price_freeze_test", "No visible holiday price freeze: increases continue through Nov–Feb",
           "Weekly persistent regular-price increases; shaded = the Nov 1–Feb 5 window described as an industry practice; off-scale peaks labelled")


def fig_promotions() -> None:
    d = t("07_promotions_by_banner")
    banners = ["Metro", "Loblaws", "Voilà", "Walmart", "No Frills"]
    agg = d[d.banner.isin(banners)].groupby("banner").apply(
        lambda g: pd.Series({"promo_share": np.average(g.promo_share, weights=g.product_days),
                             "depth": np.average(g.median_depth, weights=g.product_days)}), include_groups=False)
    agg = agg.loc[banners]
    # promotion share over all products (matches 01_coverage and the memo); depth from categorised items
    agg["promo_share"] = t("01_coverage").set_index("banner").loc[banners, "promo_share"]
    idx = t("03b_banner_index_overall").set_index("banner")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(top=0.8, bottom=0.17)
    ax = axes[0]
    ax.bar(agg.index, agg["promo_share"], color=[BANNER_COLORS[b] for b in agg.index], width=0.6)
    for i, (v, dep) in enumerate(zip(agg["promo_share"], agg["depth"])):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=9.5, color=INK)
        ax.text(i, v / 2, f"depth\n{dep:.0%}", ha="center", va="center", fontsize=8, color="white")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("Product-days on promotion (median depth)", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    core = ["Metro", "Loblaws", "No Frills", "Walmart"]
    for i, b in enumerate(core):
        r, s = idx.loc[b, "regular_index"], idx.loc[b, "shelf_vs_regular_base"]
        ax.annotate("", xy=(i, s), xytext=(i, r), arrowprops=dict(arrowstyle="->", color=BANNER_COLORS[b], lw=2))
        ax.scatter([i], [r], color=BANNER_COLORS[b], s=60, zorder=3)
        ax.text(i - 0.12, r, f"{r:.1f}", va="center", ha="right", fontsize=9, color=INK2)
        ax.text(i + 0.12, s, f"{s:.1f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_xticks(range(len(core)), core)
    ax.set_xlim(-0.6, len(core) - 0.4)
    ax.axhline(100, color=MUTED, ls="--", lw=1)
    ax.set_ylim(78, 118)
    ax.set_title("Basket index: regular → after promotions", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    finish(fig, "fig06_promotions", "High-low banners promote far more often, which narrows but never closes the gap",
           "Promotions = struck-through 'was' price, multi-buy or sale flag on the shelf listing")


def fig_private_label() -> None:
    d = t("08b_brand_architecture")
    d = d[d.banner.isin(["Loblaws", "No Frills", "Metro", "Voilà", "Walmart", "T&T"])].sort_values("pl_share_assortment")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.8, bottom=0.17)
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["pl_share_assortment"], height=0.38, color="#2a78d6", label="Share of assortment (products)")
    ax.barh(y - 0.2, d["pl_share_promo_days"], height=0.38, color="#eb6834", label="Share of promotion-days")
    for i, (a, p) in enumerate(zip(d["pl_share_assortment"], d["pl_share_promo_days"])):
        ax.text(a + 0.003, i + 0.2, f"{a:.0%}", va="center", fontsize=8.5, color=INK)
        ax.text(p + 0.003, i - 0.2, f"{p:.0%}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, d["banner"])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    finish(fig, "fig07_private_label", "Store brands are a steering tool: 13–20% of the shelf at the big three's banners",
           "Private label identified from brand / product name (e.g. PC, No Name, Selection, Irresistibles, Compliments, Great Value)")


def fig_inflation() -> None:
    d = t("09_inflation_vs_cpi")
    d["month"] = pd.to_datetime(d["month"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.82, bottom=0.15, right=0.8)
    series = ["Loblaws", "No Frills", "Metro", "Walmart", "Voilà", "CPI food from stores — Ontario"]
    for s in series:
        g = d[d.banner == s].sort_values("month")
        c = BANNER_COLORS.get(s, INK)
        ax.plot(g["month"], g["index"], color=c, lw=2.4 if s.startswith("CPI") else 1.8,
                ls="--" if s.startswith("CPI") else "-")
        ax.text(g["month"].iloc[-1] + pd.Timedelta(days=12), g["index"].iloc[-1],
                ("StatCan CPI, food from stores (ON)" if s.startswith("CPI") else s) + f"  {g['index'].iloc[-1]:.1f}",
                va="center", fontsize=8.5, color=INK2 if not s.startswith("CPI") else INK)
    ax.axhline(100, color=MUTED, lw=1)
    ax.set_ylabel("Index, Aug 2024 = 100")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    finish(fig, "fig08_inflation_vs_cpi", "Scraped shelf prices track official grocery inflation",
           "Chained matched-model index of regular prices (same products month to month) vs Statistics Canada CPI",
           source="Sources: Project Hammer; Statistics Canada Table 18-10-0004-01 (CPI, not seasonally adjusted).")


def fig_statcan_context() -> None:
    d = t("09b_statcan_cpi")
    d["month"] = pd.to_datetime(d["month"])
    d = d[(d.geo == "Canada") & d["product"].isin(["All-items", "Food purchased from stores"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.25})
    fig.subplots_adjust(top=0.8, bottom=0.17)
    colors = {"All-items": MUTED, "Food purchased from stores": "#e34948"}
    for p, g in d.groupby("product"):
        g = g.sort_values("month")
        base = g.loc[g.month == "2020-01-01", "value"].iloc[0]
        axes[0].plot(g["month"], 100 * g["value"] / base, color=colors[p], label=p)
        gg = g[g.month >= "2019-01-01"]
        axes[1].plot(gg["month"], gg["yoy"], color=colors[p], label=p)
    axes[0].set_title("Price level, Jan 2020 = 100", fontsize=10.5)
    axes[1].set_title("Year-over-year change", fontsize=10.5)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].axhline(0, color=INK2, lw=1)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(fontsize=8.5, loc="upper left")
    cum = t("09c_statcan_cumulative").set_index("product")
    finish(fig, "fig09_statcan_context",
           f"Grocery prices are up {cum.loc['Food purchased from stores', 'cum_change_since_jan2020']:.0%} since Jan 2020 vs "
           f"{cum.loc['All-items', 'cum_change_since_jan2020']:.0%} for all items",
           "Canada, CPI not seasonally adjusted", source="Source: Statistics Canada Table 18-10-0004-01.")


def fig_kvi() -> None:
    d = t("10_kvi_entry_prices")
    banners = ["No Frills", "Walmart", "Loblaws", "Metro", "Voilà"]
    d = d[d.banner.isin(banners)].copy()
    d["rel"] = d["entry_shelf_price"] / d["statcan_ontario_avg"] - 1
    kv = d.groupby("kvi")["rel"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.24)
    offs = np.linspace(-0.24, 0.24, len(banners))
    for o, b in zip(offs, banners):
        g = d[d.banner == b].set_index("kvi").reindex(kv)
        ax.scatter(g["rel"], np.arange(len(kv)) + o, color=BANNER_COLORS[b], s=36, label=b, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(kv)), [k.replace(", ", "\n") for k in kv], fontsize=8.5)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("Cheapest option on the shelf vs StatCan Ontario average retail price (Mar–Aug 2026)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.08), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    finish(fig, "fig10_kvi_entry_prices", "Entry price points on staples: the discount banners set the floor",
           "Median daily lowest shelf price among matching items (any brand, same pack size)",
           source="Sources: Project Hammer; Statistics Canada Table 18-10-0245-01 (monthly average retail prices).")


def main() -> None:
    fig_price_ladder()
    fig_identical_gaps()
    fig_loblaw_siblings()
    fig_sync()
    fig_freeze()
    fig_promotions()
    fig_private_label()
    fig_inflation()
    fig_statcan_context()
    fig_kvi()


if __name__ == "__main__":
    main()
