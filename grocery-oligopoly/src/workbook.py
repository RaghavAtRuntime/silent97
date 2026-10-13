"""Build the Excel analysis workbook from outputs/tables/*.csv.

Summary cells are live Excel formulas (AVERAGEIFS / SUMIFS / ratios) over the data
sheets, written with cached values so they display correctly before recalculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell

from config import ROOT, SOURCES, TABLES

OUT = ROOT / "outputs" / "grocery_oligopoly_analysis.xlsx"
NAVY, ORANGE, BLUE, GREY = "#1F3A5F", "#EB6834", "#2A78D6", "#8A8984"
BANNER_COLORS = {"Loblaws": "#2A78D6", "No Frills": "#184F95", "Metro": "#EB6834", "Walmart": "#6250D6",
                 "Voilà": "#1BAF7A", "CPI food from stores — Ontario": "#0B0B0B"}


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


class Book:
    def __init__(self) -> None:
        self.wb = xlsxwriter.Workbook(str(OUT), {"nan_inf_to_errors": True})
        wb = self.wb
        self.f_title = wb.add_format({"bold": True, "font_size": 16, "font_color": NAVY})
        self.f_sub = wb.add_format({"italic": True, "font_color": GREY})
        self.f_head = wb.add_format({"bold": True, "bg_color": NAVY, "font_color": "white", "text_wrap": True,
                                     "valign": "vcenter", "border": 1, "border_color": "#D9D9D9"})
        self.f_pct = wb.add_format({"num_format": "0.0%"})
        self.f_pct_b = wb.add_format({"num_format": "0.0%", "bold": True})
        self.f_num = wb.add_format({"num_format": "#,##0"})
        self.f_dec = wb.add_format({"num_format": "0.00"})
        self.f_dec1 = wb.add_format({"num_format": "0.0"})
        self.f_money = wb.add_format({"num_format": "$0.00"})
        self.f_date = wb.add_format({"num_format": "yyyy-mm-dd"})
        self.f_month = wb.add_format({"num_format": "mmm yyyy"})
        self.f_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        self.f_bold = wb.add_format({"bold": True})
        self.f_kpi = wb.add_format({"bold": True, "font_size": 20, "font_color": NAVY, "num_format": "0.0%"})
        self.f_kpi_x = wb.add_format({"bold": True, "font_size": 20, "font_color": NAVY, "num_format": '0.00"×"'})
        self.f_kpi_n = wb.add_format({"bold": True, "font_size": 20, "font_color": NAVY, "num_format": "0.0"})

    def table(self, ws, df: pd.DataFrame, row: int, col: int = 0, formats: dict | None = None,
              widths: dict | None = None, name: str | None = None) -> tuple[int, int]:
        """Write a DataFrame as an Excel table; returns (first data row, last data row) 0-indexed."""
        formats = formats or {}
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.Series(df[c].dt.to_pydatetime(), index=df.index, dtype=object)
            elif pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype(object)
        df = df.astype(object).where(pd.notna(df), None)
        data = df.values.tolist()
        cols = [{"header": str(c), "header_format": self.f_head, "format": formats.get(c)} for c in df.columns]
        ws.add_table(row, col, row + len(df), col + len(df.columns) - 1,
                     {"data": data, "columns": cols, "style": "Table Style Light 1", "name": name})
        for i, c in enumerate(df.columns):
            ws.set_column(col + i, col + i, (widths or {}).get(c, max(11, min(34, len(str(c)) + 2))))
        ws.set_row(row, 32)
        return row + 1, row + len(df)

    def header(self, ws, title: str, subtitle: str) -> None:
        ws.write(0, 0, title, self.f_title)
        ws.write(1, 0, subtitle, self.f_sub)
        ws.hide_gridlines(2)


def main() -> None:
    b = Book()
    wb = b.wb

    # ---------------- README ----------------
    ws = wb.add_worksheet("README")
    b.header(ws, "Canada's Grocery Oligopoly — consumer-behaviour price analysis",
             "Toronto banner shelf prices (Project Hammer, Feb 2024–Oct 2026) with Statistics Canada context")
    ws.set_column(0, 0, 28)
    ws.set_column(1, 1, 110)
    rows = [
        ("Question", "Do Canada's grocery parents compete banner-against-banner, or manage a portfolio of price tiers?"),
        ("Unit of analysis", "Banner × product × day. Identical products matched on SKU (Loblaws ↔ No Frills share one "
                             "catalogue) or on UPC/GTIN across owners."),
        ("Regular price", "The shelf price when no promotion is running; on promotion it is the struck-through 'was' "
                          "price (or the single-unit price for multi-buy deals)."),
        ("Persistent change", "A regular-price change that is not reversed within 14 days (temporary cuts are excluded)."),
        ("Synchronisation lift", "Share of A's regular-price changes matched by a same-direction change at B within ±k days, "
                                 "divided by the same share when A's dates are shifted ±2–6 months (placebo). 1.0 = chance."),
        ("Banner price index", "Geometric mean price of an identical basket (UPCs stocked by Loblaws, No Frills, Metro and "
                               "Walmart in the same month) relative to the four-banner average = 100."),
        ("Caveats", "Online prices for one Toronto neighbourhood; not a census of stores. No cost or volume data, so "
                    "margins are not observed. Synchronised or identical prices are consistent with coordination but "
                    "also with common supplier costs and manufacturer-suggested prices — this is descriptive evidence, "
                    "not a finding of collusion. Save-On-Foods prices are not Toronto prices (out-of-market control)."),
    ]
    r = 3
    for k, v in rows:
        ws.write(r, 0, k, b.f_bold)
        ws.write(r, 1, v, b.f_wrap)
        ws.set_row(r, 32 if len(v) > 110 else 18)
        r += 1
    r += 1
    ws.write(r, 0, "Sources", b.f_bold)
    for s in SOURCES.values():
        ws.write(r, 1, s["citation"], b.f_wrap)
        r += 1
    r += 1
    ws.write(r, 0, "Sheets", b.f_bold)
    guide = [
        ("Dashboard", "Headline KPIs (live formulas referencing the analysis sheets)."),
        ("Coverage", "What the cleaned panel contains, by banner."),
        ("Banner_Index", "Monthly identical-basket price index; summary via AVERAGEIFS; line chart."),
        ("Identical_Gaps", "Same item, same day, across banner pairs."),
        ("Loblaw_Siblings", "Loblaws vs No Frills monthly premium on identical SKUs; by category & brand type."),
        ("Synchronisation", "Timing of regular-price changes vs placebo, by pair; lift = observed / placebo (formula)."),
        ("Lead_Lag", "Who changes first, and how often rivals land on the identical new price."),
        ("Freeze_Test", "Weekly increase rates; Nov 1–Feb 5 window vs rest via SUMIFS."),
        ("Promotions", "Promotion frequency, depth and co-timing."),
        ("Brand_Architecture", "Private-label share of assortment and of promotion-days."),
        ("Inflation", "Banner matched-model indices vs StatCan CPI (food purchased from stores)."),
        ("StatCan_CPI", "CPI series, Canada / Ontario, 2018+."),
        ("KVI_Prices", "Entry price for 10 staples by banner vs StatCan Ontario average retail price."),
        ("Cleaning_Log", "Row counts at every cleaning step."),
    ]
    for k, v in guide:
        ws.write_url(r, 0, f"internal:'{k}'!A1", string=k)
        ws.write(r, 1, v)
        r += 1

    dash = wb.add_worksheet("Dashboard")  # filled after data sheets exist

    # ---------------- Coverage ----------------
    ws = wb.add_worksheet("Coverage")
    b.header(ws, "Panel coverage by banner", "Cleaned Project Hammer data; Sobeys (4 days) excluded")
    cov = t("01_coverage")
    b.table(ws, cov, 3, formats={"products": b.f_num, "product_days": b.f_num, "days": b.f_num,
                                 "promo_share": b.f_pct, "products_with_gtin": b.f_num},
            widths={"role": 40, "parent": 20}, name="Coverage")

    # ---------------- Banner index ----------------
    ws = wb.add_worksheet("Banner_Index")
    b.header(ws, "Identical-basket price index by banner (4-banner average = 100)",
             "UPCs stocked by all of Loblaws, No Frills, Metro and Walmart in the month")
    bi = t("03_banner_index_monthly")
    bi["month"] = pd.to_datetime(bi["month"])
    bi = bi[["month", "banner", "basket_items", "regular_index", "shelf_index", "shelf_vs_regular_base"]]
    first, last = b.table(ws, bi, 3, col=8, formats={"month": b.f_month, "basket_items": b.f_num,
                                                     "regular_index": b.f_dec1, "shelf_index": b.f_dec1,
                                                     "shelf_vs_regular_base": b.f_dec1}, name="BannerIndex")
    rng = lambda c: f"{xl_col_to_name(8 + c)}{first + 1}:{xl_col_to_name(8 + c)}{last + 1}"  # noqa: E731
    ws.write_row(3, 0, ["Banner", "Avg regular index (Aug 2024+)", "Avg shelf index on regular base",
                        "Realised promo discount"], b.f_head)
    ws.set_row(3, 45)
    ws.set_column(0, 0, 14)
    ws.set_column(1, 3, 16)
    summ = t("03b_banner_index_overall")
    for i, row in enumerate(summ.itertuples()):
        rr = 4 + i
        ws.write(rr, 0, row.banner, b.f_bold)
        cond = f'{rng(1)},A{rr + 1},{rng(0)},">="&DATE(2024,8,1)'
        ws.write_formula(rr, 1, f"=AVERAGEIFS({rng(3)},{cond})", b.f_dec1, row.regular_index)
        ws.write_formula(rr, 2, f"=AVERAGEIFS({rng(5)},{cond})", b.f_dec1, row.shelf_vs_regular_base)
        ws.write_formula(rr, 3, f"=1-C{rr + 1}/B{rr + 1}", b.f_pct, row.realised_promo_discount)
    ch = wb.add_chart({"type": "line"})
    piv = bi.pivot(index="month", columns="banner", values="regular_index")[["Metro", "Loblaws", "No Frills", "Walmart"]]
    hr = 10
    ws.write(hr - 1, 0, "Chart data (regular index, pivoted)", b.f_bold)
    ws.write_row(hr, 0, ["month", *piv.columns], b.f_head)
    for i, (m, vals) in enumerate(piv.iterrows()):
        ws.write_datetime(hr + 1 + i, 0, m.to_pydatetime(), b.f_month)
        for j, v in enumerate(vals):
            ws.write_number(hr + 1 + i, 1 + j, float(v), b.f_dec1)
    n = len(piv)
    for j, banner in enumerate(piv.columns):
        ch.add_series({"name": banner, "categories": ["Banner_Index", hr + 1, 0, hr + n, 0],
                       "values": ["Banner_Index", hr + 1, 1 + j, hr + n, 1 + j],
                       "line": {"color": BANNER_COLORS[banner], "width": 2}})
    ch.set_title({"name": "Regular-price index on an identical basket"})
    ch.set_y_axis({"min": 80, "max": 120, "major_gridlines": {"visible": True, "line": {"color": "#E6E5E1"}}})
    ch.set_x_axis({"date_axis": True, "num_format": "mmm yy"})
    ch.set_legend({"position": "bottom"})
    ch.set_size({"width": 620, "height": 320})
    ws.insert_chart(hr + n + 3, 0, ch)

    # ---------------- Identical gaps ----------------
    ws = wb.add_worksheet("Identical_Gaps")
    b.header(ws, "Same product, same day: price gaps across banner pairs",
             "Median of each product's daily log price ratio, then summarised across products")
    g = t("02_identical_item_gaps")
    pct_cols = [c for c in g.columns if "premium" in c or "share" in c]
    f0, f1 = b.table(ws, g, 3, formats={**{c: b.f_pct for c in pct_cols}, "matched_products": b.f_num,
                                        "matched_product_days": b.f_num}, widths={"group": 24}, name="IdenticalGaps")
    col = list(g.columns).index("median_regular_premium_a_vs_b")
    ws.conditional_format(f0, col, f1, col, {"type": "3_color_scale", "min_color": "#2A78D6", "mid_color": "#F0EFEC",
                                             "max_color": "#E34948", "mid_type": "num", "mid_value": 0})
    col = list(g.columns).index("share_days_identical_regular")
    ws.conditional_format(f0, col, f1, col, {"type": "data_bar", "bar_color": ORANGE, "bar_solid": True})

    # ---------------- Loblaw siblings ----------------
    ws = wb.add_worksheet("Loblaw_Siblings")
    b.header(ws, "Inside one portfolio: Loblaws vs No Frills on identical SKUs",
             "Median regular-price premium by month, and by category and brand type")
    m = t("02c_gap_by_month")
    m = m[(m.banner_a == "Loblaws") & (m.banner_b == "No Frills")].copy()
    m["month"] = pd.to_datetime(m["month"])
    m = m[["month", "n", "premium", "same_share"]].rename(columns={"n": "matched_items",
                                                                    "premium": "median_premium",
                                                                    "same_share": "share_identical_price"})
    f0, f1 = b.table(ws, m, 3, formats={"month": b.f_month, "matched_items": b.f_num, "median_premium": b.f_pct,
                                        "share_identical_price": b.f_pct}, name="LoblawMonthly")
    ch = wb.add_chart({"type": "line"})
    ch.add_series({"name": "Median Loblaws premium over No Frills", "categories": ["Loblaw_Siblings", f0, 0, f1, 0],
                   "values": ["Loblaw_Siblings", f0, 2, f1, 2], "line": {"color": BLUE, "width": 2.25},
                   "marker": {"type": "circle", "size": 4, "fill": {"color": BLUE}, "border": {"color": BLUE}}})
    ch.set_title({"name": "Loblaws premium over No Frills (identical SKU)"})
    ch.set_y_axis({"num_format": "0%", "min": 0, "major_gridlines": {"visible": True, "line": {"color": "#E6E5E1"}}})
    ch.set_x_axis({"date_axis": True, "num_format": "mmm yy"})
    ch.set_legend({"none": True})
    ch.set_size({"width": 620, "height": 300})
    ws.insert_chart(3, 6, ch)
    c = t("02b_gap_by_category_brand")
    c = c[(c.banner_a == "Loblaws") & (c.banner_b == "No Frills")].drop(columns=["banner_a", "banner_b"])
    b.table(ws, c, f1 + 4, formats={"median_regular_premium": b.f_pct, "products": b.f_num},
            widths={"category": 26}, name="LoblawByCategory")

    # ---------------- Synchronisation ----------------
    ws = wb.add_worksheet("Synchronisation")
    b.header(ws, "Do regular-price changes coincide more than chance?",
             "Observed = share of A's persistent changes matched by B (same direction) within ±k days; placebo shifts dates ±2–6 months")
    s = t("04_synchronisation")
    keep = ["banner_a", "banner_b", "pair_group", "events_a"] + [f"{p}_{k}d" for k in (0, 1, 3, 7) for p in ("observed", "placebo")]
    s = s[keep]
    f0, f1 = b.table(ws, s, 3, formats={**{c: b.f_pct for c in keep[4:]}, "events_a": b.f_num},
                     widths={"pair_group": 24}, name="Sync")
    base = len(keep)
    ws.write_row(3, base, ["lift_0d", "lift_1d", "lift_3d", "lift_7d"], b.f_head)
    s_full = t("04_synchronisation")
    for i in range(len(s)):
        for j, k in enumerate((0, 1, 3, 7)):
            oc = keep.index(f"observed_{k}d")
            pc = keep.index(f"placebo_{k}d")
            ws.write_formula(f0 + i, base + j, f"=IFERROR({xl_rowcol_to_cell(f0 + i, oc)}/{xl_rowcol_to_cell(f0 + i, pc)},\"\")",
                             b.f_dec, float(s_full.iloc[i][f"lift_{k}d"]))
    ws.set_column(base, base + 3, 10)
    ws.conditional_format(f0, base, f1, base + 3, {"type": "3_color_scale", "min_color": "#F0EFEC", "mid_color": "#F0EFEC",
                                                   "max_color": "#2A78D6", "min_type": "num", "min_value": 0.5,
                                                   "mid_type": "num", "mid_value": 1, "max_type": "num", "max_value": 2.5})

    # ---------------- Lead / lag ----------------
    ws = wb.add_worksheet("Lead_Lag")
    b.header(ws, "Who moves first, and do rivals land on the same price?",
             "Nearest same-direction persistent change at the other banner within ±14 days")
    ll = t("05_lead_lag").merge(t("05b_paired_change_same_new_price")[["banner_a", "banner_b", "share_new_price_identical"]],
                                on=["banner_a", "banner_b"])
    f0, f1 = b.table(ws, ll, 3, formats={"paired_changes": b.f_num, "share_same_day": b.f_pct, "share_a_first": b.f_pct,
                                         "share_b_first": b.f_pct, "median_abs_lag_days": b.f_dec1,
                                         "share_new_price_identical": b.f_pct}, widths={"pair_group": 24}, name="LeadLag")
    col = list(ll.columns).index("share_new_price_identical")
    ws.conditional_format(f0, col, f1, col, {"type": "data_bar", "bar_color": ORANGE, "bar_solid": True})

    # ---------------- Freeze test ----------------
    ws = wb.add_worksheet("Freeze_Test")
    b.header(ws, "Is there a Nov 1–Feb 5 price freeze?",
             "Persistent regular-price increases per 1,000 active products per week; window flag on the week midpoint")
    fw = t("06_weekly_change_rates")
    fw["week"] = pd.to_datetime(fw["week"])
    fw = fw[["banner", "week", "freeze_window", "active", "ups", "downs", "ups_per_1000", "downs_per_1000"]]
    f0, f1 = b.table(ws, fw, 3, col=11, formats={"week": b.f_date, "active": b.f_num, "ups": b.f_num, "downs": b.f_num,
                                                "ups_per_1000": b.f_dec1, "downs_per_1000": b.f_dec1}, name="Weekly")
    R = lambda c: f"{xl_col_to_name(11 + c)}{f0 + 1}:{xl_col_to_name(11 + c)}{f1 + 1}"  # noqa: E731
    ws.write_row(3, 0, ["Banner", "Increases /1,000 in window", "Increases /1,000 outside", "Ratio in / out",
                        "Mann-Whitney p (weekly rates)"], b.f_head)
    ws.set_row(3, 45)
    ws.set_column(0, 0, 14)
    ws.set_column(1, 4, 15)
    sig = t("06c_freeze_window_significance").set_index("banner")
    for i, banner in enumerate(["Loblaws", "No Frills", "Metro", "Walmart", "Voilà", "T&T", "Galleria", "Save-On-Foods"]):
        rr = 4 + i
        g_ = fw[fw.banner == banner]
        vin = 1000 * g_.loc[g_.freeze_window, "ups"].sum() / g_.loc[g_.freeze_window, "active"].sum()
        vout = 1000 * g_.loc[~g_.freeze_window, "ups"].sum() / g_.loc[~g_.freeze_window, "active"].sum()
        ws.write(rr, 0, banner, b.f_bold)
        ws.write_formula(rr, 1, f"=1000*SUMIFS({R(4)},{R(0)},A{rr + 1},{R(2)},TRUE)/SUMIFS({R(3)},{R(0)},A{rr + 1},{R(2)},TRUE)",
                         b.f_dec1, vin)
        ws.write_formula(rr, 2, f"=1000*SUMIFS({R(4)},{R(0)},A{rr + 1},{R(2)},FALSE)/SUMIFS({R(3)},{R(0)},A{rr + 1},{R(2)},FALSE)",
                         b.f_dec1, vout)
        ws.write_formula(rr, 3, f"=B{rr + 1}/C{rr + 1}", b.f_dec, vin / vout)
        ws.write_number(rr, 4, float(sig.loc[banner, "mann_whitney_p"]), b.f_dec)
    ws.conditional_format(4, 3, 11, 3, {"type": "3_color_scale", "min_color": "#2A78D6", "mid_color": "#F0EFEC",
                                        "max_color": "#E34948", "mid_type": "num", "mid_value": 1})
    ws.write(13, 0, "By brand type", b.f_bold)
    b.table(ws, t("06b_freeze_window_test"), 14, formats={c: b.f_dec1 for c in t("06b_freeze_window_test").columns
                                                         if "per_1000" in c} | {"increase_rate_ratio_in_vs_out": b.f_dec},
            name="FreezeByBrand")

    # ---------------- Promotions ----------------
    ws = wb.add_worksheet("Promotions")
    b.header(ws, "Promotion strategy by banner", "Share of product-days on promotion, depth, and co-timing across banners")
    p = t("07_promotions_by_banner")[["banner", "brand_type", "product_days", "promo_share", "median_depth", "mean_depth",
                                      "share_deep_25pct", "multibuy_share_of_promos"]]
    f0, f1 = b.table(ws, p, 3, formats={"product_days": b.f_num, "promo_share": b.f_pct, "median_depth": b.f_pct,
                                        "mean_depth": b.f_pct, "share_deep_25pct": b.f_pct,
                                        "multibuy_share_of_promos": b.f_pct}, name="Promos")
    pc = t("07b_promo_cotiming")
    b.table(ws, pc, f1 + 4, formats={"matched_days": b.f_num, "p_b_promo": b.f_pct, "p_b_promo_given_a_promo": b.f_pct,
                                     "promo_cotiming_lift": b.f_dec}, widths={"pair_group": 24}, name="PromoCotiming")

    # ---------------- Brand architecture ----------------
    ws = wb.add_worksheet("Brand_Architecture")
    b.header(ws, "Private label as a steering tool", "Store-brand share of assortment and of promotion-days (categorised food & household items)")
    ba = t("08b_brand_architecture")
    f0, f1 = b.table(ws, ba[["banner", "products", "pl_products", "promo_days", "pl_promo_days"]], 3,
                     formats={"products": b.f_num, "pl_products": b.f_num, "promo_days": b.f_num, "pl_promo_days": b.f_num},
                     name="BrandArch")
    ws.write_row(3, 5, ["PL share of assortment", "PL share of promo-days"], b.f_head)
    for i, row in enumerate(ba.itertuples()):
        rr = f0 + i
        ws.write_formula(rr, 5, f"=C{rr + 1}/B{rr + 1}", b.f_pct, row.pl_share_assortment)
        ws.write_formula(rr, 6, f"=IFERROR(E{rr + 1}/D{rr + 1},0)", b.f_pct, row.pl_share_promo_days)
    ws.set_column(5, 6, 16)
    bc = t("08_brand_architecture_by_category")[["banner", "category", "products", "pl_products", "promo_days", "pl_promo_days"]]
    b.table(ws, bc, f1 + 4, formats={"products": b.f_num, "pl_products": b.f_num, "promo_days": b.f_num,
                                     "pl_promo_days": b.f_num}, widths={"category": 26}, name="BrandArchCat")

    # ---------------- Inflation ----------------
    ws = wb.add_worksheet("Inflation")
    b.header(ws, "Banner shelf-price inflation vs official CPI (Aug 2024 = 100)",
             "Chained matched-model index of regular prices vs StatCan CPI, food purchased from stores")
    inf = t("09_inflation_vs_cpi")
    inf["month"] = pd.to_datetime(inf["month"])
    piv = inf.pivot_table(index="month", columns="banner", values="index").reset_index()
    order = ["month", "Loblaws", "No Frills", "Metro", "Walmart", "Voilà", "T&T", "Galleria", "Save-On-Foods",
             "CPI food from stores — Ontario", "CPI food from stores — Canada"]
    piv = piv[[c for c in order if c in piv.columns]]
    f0, f1 = b.table(ws, piv, 3, formats={c: (b.f_month if c == "month" else b.f_dec1) for c in piv.columns},
                     widths={c: 14 for c in piv.columns}, name="Inflation")
    ch = wb.add_chart({"type": "line"})
    for name in ["Loblaws", "No Frills", "Metro", "Walmart", "Voilà", "CPI food from stores — Ontario"]:
        j = list(piv.columns).index(name)
        ch.add_series({"name": name, "categories": ["Inflation", f0, 0, f1, 0], "values": ["Inflation", f0, j, f1, j],
                       "line": {"color": BANNER_COLORS[name], "width": 2.5 if name.startswith("CPI") else 1.75,
                                "dash_type": "dash" if name.startswith("CPI") else "solid"}})
    ch.set_title({"name": "Regular-price index vs CPI (Aug 2024 = 100)"})
    ch.set_y_axis({"min": 95, "max": 110, "major_gridlines": {"visible": True, "line": {"color": "#E6E5E1"}}})
    ch.set_x_axis({"date_axis": True, "num_format": "mmm yy"})
    ch.set_legend({"position": "bottom"})
    ch.set_size({"width": 720, "height": 340})
    ws.insert_chart(f1 + 3, 0, ch)

    # ---------------- StatCan CPI ----------------
    ws = wb.add_worksheet("StatCan_CPI")
    b.header(ws, "Statistics Canada CPI (Table 18-10-0004-01)", "Not seasonally adjusted, 2002 = 100; yoy = 12-month change")
    cpi = t("09b_statcan_cpi")
    cpi["month"] = pd.to_datetime(cpi["month"])
    b.table(ws, cpi, 3, formats={"month": b.f_month, "value": b.f_dec1, "yoy": b.f_pct},
            widths={"product": 44, "geo": 18}, name="CPI")
    cum = t("09c_statcan_cumulative")
    ws.write(3, 7, "Cumulative change since Jan 2020 (Canada)", b.f_bold)
    b.table(ws, cum, 4, col=7, formats={"jan2020": b.f_dec1, "latest": b.f_dec1, "cum_change_since_jan2020": b.f_pct},
            widths={"product": 44}, name="CPICum")

    # ---------------- KVI ----------------
    ws = wb.add_worksheet("KVI_Prices")
    b.header(ws, "Entry price points on 10 staples vs StatCan average retail price",
             "Median daily lowest shelf price among matching items, Mar–Aug 2026; StatCan Table 18-10-0245-01")
    k = t("10_kvi_entry_prices")
    kp = k.pivot(index="kvi", columns="banner", values="entry_shelf_price")
    cols_ = [c for c in ["No Frills", "Walmart", "Loblaws", "Metro", "Voilà", "T&T", "Galleria", "Save-On-Foods"] if c in kp]
    kp = kp[cols_].join(k.groupby("kvi")[["statcan_ontario_avg", "statcan_canada_avg"]].first()).reset_index()
    f0, f1 = b.table(ws, kp, 3, formats={c: b.f_money for c in kp.columns if c != "kvi"}, widths={"kvi": 28},
                     name="KVI")
    ncol = len(kp.columns)
    ws.write_row(3, ncol, ["No Frills vs ON avg", "Loblaws vs ON avg", "Metro vs ON avg", "Walmart vs ON avg"], b.f_head)
    on = list(kp.columns).index("statcan_ontario_avg")
    for i in range(len(kp)):
        for j, bn in enumerate(["No Frills", "Loblaws", "Metro", "Walmart"]):
            bc_ = list(kp.columns).index(bn)
            v = kp.iloc[i][bn] / kp.iloc[i]["statcan_ontario_avg"] - 1
            ws.write_formula(f0 + i, ncol + j,
                             f'=IFERROR({xl_rowcol_to_cell(f0 + i, bc_)}/{xl_rowcol_to_cell(f0 + i, on)}-1,"")',
                             b.f_pct, None if pd.isna(v) else float(v))
    ws.set_column(ncol, ncol + 3, 13)
    ws.conditional_format(f0, ncol, f1, ncol + 3, {"type": "3_color_scale", "min_color": "#2A78D6", "mid_color": "#F0EFEC",
                                                   "max_color": "#E34948", "mid_type": "num", "mid_value": 0})
    mt = t("10a_kvi_item_matches")
    ws.write(f1 + 3, 0, "Items matched to each staple (for audit)", b.f_bold)
    b.table(ws, mt, f1 + 4, widths={"product_name": 60, "kvi": 26}, name="KVIMatches")

    # ---------------- Cleaning log ----------------
    ws = wb.add_worksheet("Cleaning_Log")
    b.header(ws, "Data cleaning log", "Row counts after each step of src/build_panel.py")
    b.table(ws, t("00_cleaning_log"), 3, formats={"rows": b.f_num}, widths={"step": 60}, name="CleaningLog")

    # ---------------- Dashboard ----------------
    ws = dash
    b.header(ws, "Headline findings", "Each figure is a live formula pointing at the analysis sheets")
    ws.set_column(0, 0, 4)
    ws.set_column(1, 1, 18)
    ws.set_column(2, 2, 80)
    g = t("02_identical_item_gaps")
    gi = {(r.banner_a, r.banner_b): i for i, r in enumerate(g.itertuples())}
    gcol = lambda c: xl_col_to_name(list(g.columns).index(c))  # noqa: E731
    sync = t("04_synchronisation")
    si = {(r.banner_a, r.banner_b): i for i, r in enumerate(sync.itertuples())}
    lift1_col = xl_col_to_name(len(keep) + 1)
    summ = t("03b_banner_index_overall").set_index("banner")
    kpis = [
        (f"=Identical_Gaps!{gcol('median_regular_premium_a_vs_b')}{5 + gi[('Loblaws', 'No Frills')]}", b.f_kpi,
         g.iloc[gi[("Loblaws", "No Frills")]]["median_regular_premium_a_vs_b"],
         "Median premium for the identical SKU under the Loblaws banner vs No Frills (same parent, same day)."),
        (f"=Identical_Gaps!{gcol('share_days_identical_regular')}{5 + gi[('Loblaws', 'Metro')]}", b.f_kpi,
         g.iloc[gi[("Loblaws", "Metro")]]["share_days_identical_regular"],
         "Share of matched item-days on which rival conventional banners Loblaws and Metro charge the identical regular price."),
        (f"=Identical_Gaps!{gcol('share_days_identical_regular')}{5 + gi[('Loblaws', 'Walmart')]}", b.f_kpi,
         g.iloc[gi[("Loblaws", "Walmart")]]["share_days_identical_regular"],
         "…versus Loblaws and Walmart — conventional and discount tiers almost never share a price point."),
        ("=Banner_Index!B5-Banner_Index!B7", b.f_kpi_n,
         summ.loc["Metro", "regular_index"] - summ.loc["No Frills", "regular_index"],
         "Index-point gap between the dearest (Metro) and a discount banner (No Frills) on an identical basket."),
        (f"=Synchronisation!{lift1_col}{5 + si[('Loblaws', 'No Frills')]}", b.f_kpi_x,
         sync.iloc[si[("Loblaws", "No Frills")]]["lift_1d"],
         "Synchronisation lift (±1 day) of regular-price changes inside the Loblaw portfolio."),
        (f"=Synchronisation!{lift1_col}{5 + si[('Loblaws', 'Metro')]}", b.f_kpi_x,
         sync.iloc[si[("Loblaws", "Metro")]]["lift_1d"],
         "Same measure across rival owners (Loblaws → Metro): indistinguishable from chance."),
        ("=Freeze_Test!D5", b.f_kpi_x, None,
         "Loblaws' rate of price increases inside vs outside the Nov 1–Feb 5 'freeze' window (≈1 = no freeze)."),
    ]
    for i, (f, fmt, val, txt) in enumerate(kpis):
        rr = 3 + i * 2
        if val is None:
            fw_ = t("06_weekly_change_rates")
            g_ = fw_[fw_.banner == "Loblaws"]
            val = (g_.loc[g_.freeze_window, "ups"].sum() / g_.loc[g_.freeze_window, "active"].sum()) / (
                g_.loc[~g_.freeze_window, "ups"].sum() / g_.loc[~g_.freeze_window, "active"].sum())
        ws.write_formula(rr, 1, f, fmt, float(val))
        ws.write(rr, 2, txt, b.f_wrap)
        ws.set_row(rr, 30)
    ws.activate()
    wb.close()
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
