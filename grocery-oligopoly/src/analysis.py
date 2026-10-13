"""Core analysis of the Toronto banner price panel (Project Hammer) and StatCan context.

Each function writes one or more tidy CSVs to outputs/tables/. The numbered
prefixes match the sections of the research memo.

  01 coverage                  what data we have, by banner
  02 identical-item price gaps same product (SKU / GTIN), same day, across banners
  03 banner price index        balanced basket of items stocked by all four core banners
  04 synchronisation           do regular-price changes coincide more than chance?
  05 lead / lag                who moves first when rivals change the same item
  06 price-freeze test         Nov 1 - Feb 5 "industry practice" freeze, tested
  07 promotions                frequency, depth, co-timing
  08 brand architecture        private-label share of assortment and promotions
  09 inflation                 matched-model banner index vs StatCan CPI
  10 KVI entry prices          cheapest option by banner vs StatCan average price
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from config import (
    CORE_TORONTO,
    DB_PATH,
    FREEZE_END,
    FREEZE_START,
    PLACEBO_SHIFTS,
    RAW,
    SYNC_WINDOWS,
    TABLES,
    VENDORS,
)

LABEL = {k: v["label"] for k, v in VENDORS.items()}

# Pair universe for synchronisation / promo co-timing. (A, B, match type, group)
PAIRS = [
    ("Loblaws", "NoFrills", "sku", "Same parent (Loblaw)"),
    ("Loblaws", "Metro", "gtin", "Rival parents, Toronto"),
    ("NoFrills", "Metro", "gtin", "Rival parents, Toronto"),
    ("Loblaws", "Walmart", "gtin", "Rival parents, Toronto"),
    ("NoFrills", "Walmart", "gtin", "Rival parents, Toronto"),
    ("Metro", "Walmart", "gtin", "Rival parents, Toronto"),
    ("Loblaws", "Voila", "gtin", "Rival parents, Toronto"),
    ("Metro", "Voila", "gtin", "Rival parents, Toronto"),
    ("Loblaws", "SaveOnFoods", "gtin", "Out-of-market control"),
    ("NoFrills", "SaveOnFoods", "gtin", "Out-of-market control"),
    ("Metro", "SaveOnFoods", "gtin", "Out-of-market control"),
    ("Walmart", "SaveOnFoods", "gtin", "Out-of-market control"),
    ("Metro", "Galleria", "gtin", "Independent specialty"),
]


def con_() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    con.execute("set preserve_insertion_order=false")
    con.execute("set enable_progress_bar=false")
    return con


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    print(f"  wrote {name}.csv ({len(df):,} rows)")
    return df


def build_keys(con: duckdb.DuckDBPyConnection) -> None:
    """One product per (vendor, match key). GTIN duplicates within a vendor keep the
    longest-observed product. Variable-weight items are excluded from matching."""
    con.execute(
        """
        create or replace temp table pdays as
        select product_key, count(*) as n_days from daily group by product_key
        """
    )
    con.execute(
        """
        create or replace temp table keymap as
        with g as (
            select p.vendor, 'gtin' as mtype, p.gtin as mkey, p.product_key, d.n_days,
                   row_number() over (partition by p.vendor, p.gtin order by d.n_days desc) as rk
            from product_dim p join pdays d using (product_key)
            where p.gtin is not null and not p.var_weight
        ), s as (
            select p.vendor, 'sku' as mtype, p.sku as mkey, p.product_key, d.n_days, 1 as rk
            from product_dim p join pdays d using (product_key)
            where p.vendor in ('Loblaws', 'NoFrills') and not p.var_weight
        )
        select vendor, mtype, mkey, product_key from g where rk = 1
        union all
        select vendor, mtype, mkey, product_key from s
        """
    )


# ---------------------------------------------------------------------------
def coverage(con: duckdb.DuckDBPyConnection) -> None:
    df = con.sql(
        """
        select p.vendor, count(distinct d.product_key) as products, count(*) as product_days,
               count(distinct d.d) as days, min(d.d) as first_day, max(d.d) as last_day,
               avg(d.on_promo::int) as promo_share,
               count(distinct case when p.gtin is not null then p.product_key end) as products_with_gtin
        from daily d join product_dim p using (product_key) group by p.vendor
        """
    ).df()
    df["banner"] = df["vendor"].map(LABEL)
    df["parent"] = df["vendor"].map(lambda v: VENDORS[v]["parent"])
    df["role"] = df["vendor"].map(lambda v: VENDORS[v]["role"])
    order = list(VENDORS)
    df = df.sort_values("vendor", key=lambda s: s.map(order.index))
    save(df[["banner", "parent", "role", "products", "product_days", "days", "first_day", "last_day",
             "promo_share", "products_with_gtin"]], "01_coverage")


# ---------------------------------------------------------------------------
def pair_daily(con: duckdb.DuckDBPyConnection, a: str, b: str, mtype: str) -> str:
    """Temp view of same-day matched observations for a banner pair."""
    view = f"pd_{a}_{b}"
    con.execute(
        f"""
        create or replace temp table {view} as
        select ka.mkey, da.d, da.regular as reg_a, db.regular as reg_b, da.cur as cur_a, db.cur as cur_b,
               da.on_promo as promo_a, db.on_promo as promo_b, pa.private_label, pa.category
        from keymap ka
        join keymap kb on kb.mtype = ka.mtype and kb.mkey = ka.mkey and kb.vendor = '{b}'
        join daily da on da.product_key = ka.product_key
        join daily db on db.product_key = kb.product_key and db.d = da.d
        join product_dim pa on pa.product_key = ka.product_key
        where ka.vendor = '{a}' and ka.mtype = '{mtype}'
        """
    )
    return view


def identical_item_gaps(con: duckdb.DuckDBPyConnection) -> None:
    rows, cat_rows, month_rows = [], [], []
    for a, b, mtype, group in PAIRS:
        v = pair_daily(con, a, b, mtype)
        # product-level median log ratio (regular and shelf), then summarise across products
        prod = con.sql(
            f"""
            select mkey, any_value(private_label) as private_label, any_value(category) as category,
                   median(ln(reg_a / reg_b)) as lr_reg, median(ln(cur_a / cur_b)) as lr_shelf,
                   avg((abs(reg_a - reg_b) < 0.005)::int) as same_reg_share, count(*) as n_days
            from {v} where reg_a is not null and reg_b is not null
            group by mkey having count(*) >= 14
            """
        ).df()
        prod = prod[np.isfinite(prod["lr_reg"])]
        if prod.empty:
            continue
        rows.append(
            {
                "banner_a": LABEL[a], "banner_b": LABEL[b], "match": mtype, "group": group,
                "matched_products": len(prod), "matched_product_days": int(prod["n_days"].sum()),
                "median_regular_premium_a_vs_b": float(np.expm1(prod["lr_reg"].median())),
                "mean_regular_premium_a_vs_b": float(np.expm1(prod["lr_reg"].mean())),
                "median_shelf_premium_a_vs_b": float(np.expm1(prod["lr_shelf"].median())),
                "share_items_a_dearer": float((prod["lr_reg"] > 0.005).mean()),
                "share_items_identical": float((prod["lr_reg"].abs() <= 0.005).mean()),
                "share_items_a_cheaper": float((prod["lr_reg"] < -0.005).mean()),
                "share_days_identical_regular": float(np.average(prod["same_reg_share"], weights=prod["n_days"])),
            }
        )
        if (a, b) in (("Loblaws", "NoFrills"), ("Loblaws", "Walmart"), ("Metro", "Walmart"), ("Loblaws", "Metro")):
            for (cat, pl), g in prod.groupby(["category", "private_label"]):
                if len(g) >= 20:
                    cat_rows.append({"banner_a": LABEL[a], "banner_b": LABEL[b], "category": cat,
                                     "private_label": pl, "products": len(g),
                                     "median_regular_premium": float(np.expm1(g["lr_reg"].median()))})
            m = con.sql(
                f"""
                select date_trunc('month', d) as month, median(ln(reg_a / reg_b)) as lr,
                       avg((abs(reg_a - reg_b) < 0.005)::int) as same_share, count(distinct mkey) as n
                from {v} where reg_a is not null and reg_b is not null group by 1 order by 1
                """
            ).df()
            m["premium"] = np.expm1(m["lr"])
            m["banner_a"], m["banner_b"] = LABEL[a], LABEL[b]
            month_rows.append(m.drop(columns="lr"))
        con.execute(f"drop table {v}")
    save(pd.DataFrame(rows), "02_identical_item_gaps")
    save(pd.DataFrame(cat_rows), "02b_gap_by_category_brand")
    save(pd.concat(month_rows), "02c_gap_by_month")


# ---------------------------------------------------------------------------
def banner_index(con: duckdb.DuckDBPyConnection) -> None:
    """Balanced-basket index: GTINs stocked by all four core banners in a month.
    Index = 100 * exp(mean_i(log p_iv - mean_v log p_iv)), i.e. geometric price
    level relative to the four-banner average for the identical basket."""
    core = ", ".join(f"'{v}'" for v in CORE_TORONTO)
    df = con.sql(
        f"""
        with m as (
            select k.vendor, k.mkey, date_trunc('month', d.d) as month,
                   median(d.regular) as reg, avg(d.cur) as shelf
            from keymap k join daily d using (product_key)
            where k.mtype = 'gtin' and k.vendor in ({core})
            group by 1, 2, 3
        ), full_ as (
            select mkey, month from m where reg is not null group by 1, 2 having count(distinct vendor) = 4
        ), b as (
            select m.* , avg(ln(reg)) over (partition by m.mkey, m.month) as mean_lr,
                   avg(ln(shelf)) over (partition by m.mkey, m.month) as mean_ls
            from m join full_ using (mkey, month)
        )
        select vendor, month, count(*) as basket_items,
               100 * exp(avg(ln(reg) - mean_lr)) as regular_index,
               100 * exp(avg(ln(shelf) - mean_ls)) as shelf_index,
               -- shelf prices measured against the same regular-price base (comparable to regular_index)
               100 * exp(avg(ln(shelf) - mean_lr)) as shelf_vs_regular_base
        from b group by 1, 2 order by 2, 1
        """
    ).df()
    df["banner"] = df["vendor"].map(LABEL)
    save(df, "03_banner_index_monthly")
    overall = con.sql(
        """select banner, avg(regular_index) as regular_index, avg(shelf_index) as shelf_index,
                  avg(shelf_vs_regular_base) as shelf_vs_regular_base,
                  1 - avg(shelf_vs_regular_base) / avg(regular_index) as realised_promo_discount,
                  avg(basket_items) as avg_basket_items
           from df where month >= '2024-08-01' group by 1 order by 2 desc"""
    ).df()
    save(overall, "03b_banner_index_overall")


# ---------------------------------------------------------------------------
def synchronisation(con: duckdb.DuckDBPyConnection) -> None:
    """For each banner pair and direction A->B: of A's regular-price changes on matched
    items (with B observed on that day), what share are matched by a same-direction
    change at B within +/-k days? The placebo shifts A's change dates by +/-2-6 months
    and asks the same question; observed / placebo = synchronisation lift.

    A lift near 1 means changes coincide no more than they would by chance given each
    banner's own change frequency. A high lift means changes cluster in time — which is
    consistent with coordination *or* with common cost shocks (e.g. a supplier list-price
    increase hitting every retailer). The out-of-market control pair helps separate the two.
    """
    shifts = [0, *PLACEBO_SHIFTS]
    kmax = max(SYNC_WINDOWS)
    out, lag_rows, same_price_rows = [], [], []
    con.execute(f"create or replace temp table shifts as select unnest({shifts}) as s")
    for a0, b0, mtype, group in PAIRS:
        for a, b in ((a0, b0), (b0, a0)):
            con.execute(
                f"""
                create or replace temp table ea as
                select k.mkey, e.d, e.dir, e.regular from events e join keymap k using (product_key)
                where k.vendor = '{a}' and k.mtype = '{mtype}' and e.persistent
                """
            )
            con.execute(
                f"""
                create or replace temp table eb as
                select k.mkey, e.d, e.dir, e.regular from events e join keymap k using (product_key)
                where k.vendor = '{b}' and k.mtype = '{mtype}' and e.persistent
                """
            )
            con.execute(
                f"""
                create or replace temp table ob as
                select k.mkey, d.d from daily d join keymap k using (product_key)
                where k.vendor = '{b}' and k.mtype = '{mtype}' and d.regular is not null
                  and k.mkey in (select mkey from ea)
                """
            )
            res = con.sql(
                f"""
                with c as (
                    select x.* from (
                        select ea.mkey, ea.d, ea.dir, s.s, ea.d + s.s::int as d_s from ea cross join shifts s
                    ) x semi join ob on ob.mkey = x.mkey and ob.d = x.d_s
                ), m as (
                    select c.mkey, c.d, c.s, min(abs(date_diff('day', c.d_s, eb.d))) as dist
                    from c left join eb on eb.mkey = c.mkey and eb.dir = c.dir
                         and eb.d between c.d_s - {kmax} and c.d_s + {kmax}
                    group by c.mkey, c.d, c.s
                )
                select s, count(*) as n,
                       {", ".join(f"avg((coalesce(dist, 999) <= {k})::int) as m{k}" for k in SYNC_WINDOWS)}
                from m group by s
                """
            ).df()
            if res.empty or res.loc[res.s == 0, "n"].sum() < 30:
                continue
            obs = res[res.s == 0].iloc[0]
            plc = res[res.s != 0]
            row = {"banner_a": LABEL[a], "banner_b": LABEL[b], "pair_group": group, "match": mtype,
                   "events_a": int(obs["n"])}
            for k in SYNC_WINDOWS:
                p = float(np.average(plc[f"m{k}"], weights=plc["n"]))
                row[f"observed_{k}d"] = float(obs[f"m{k}"])
                row[f"placebo_{k}d"] = p
                row[f"lift_{k}d"] = float(obs[f"m{k}"]) / p if p > 0 else np.nan
            out.append(row)

            # lead/lag + exact-price matching, on unshifted events within +/-14 days
            if a == a0:
                ll = con.sql(
                    """
                    with j as (
                        select ea.mkey, ea.d, ea.regular as ra, eb.regular as rb,
                               date_diff('day', ea.d, eb.d) as lag,
                               row_number() over (partition by ea.mkey, ea.d
                                                  order by abs(date_diff('day', ea.d, eb.d)), eb.d) as rk
                        from ea join eb on eb.mkey = ea.mkey and eb.dir = ea.dir
                             and eb.d between ea.d - 14 and ea.d + 14
                    )
                    select lag, ra, rb from j where rk = 1
                    """
                ).df()
                if len(ll) >= 30:
                    lag_rows.append({
                        "banner_a": LABEL[a], "banner_b": LABEL[b], "pair_group": group, "paired_changes": len(ll),
                        "share_same_day": float((ll.lag == 0).mean()),
                        "share_a_first": float((ll.lag > 0).mean()),
                        "share_b_first": float((ll.lag < 0).mean()),
                        "median_abs_lag_days": float(ll.lag.abs().median()),
                    })
                    same_price_rows.append({
                        "banner_a": LABEL[a], "banner_b": LABEL[b], "pair_group": group, "paired_changes": len(ll),
                        "share_new_price_identical": float((abs(ll.ra - ll.rb) < 0.005).mean()),
                    })
    save(pd.DataFrame(out), "04_synchronisation")
    save(pd.DataFrame(lag_rows), "05_lead_lag")
    save(pd.DataFrame(same_price_rows), "05b_paired_change_same_new_price")


# ---------------------------------------------------------------------------
def in_freeze(d: pd.Series) -> pd.Series:
    md = d.dt.month * 100 + d.dt.day
    start = FREEZE_START[0] * 100 + FREEZE_START[1]
    end = FREEZE_END[0] * 100 + FREEZE_END[1]
    return (md >= start) | (md <= end)


def price_freeze(con: duckdb.DuckDBPyConnection) -> None:
    weekly = con.sql(
        """
        with act as (
            select d.vendor, date_trunc('week', d.d) as week, p.private_label,
                   count(distinct d.product_key) as active
            from daily d join product_dim p using (product_key)
            where d.regular is not null and not p.var_weight
            group by 1, 2, 3
        ), ev as (
            select e.vendor, date_trunc('week', e.d) as week, p.private_label,
                   sum((dir > 0)::int) as ups, sum((dir < 0)::int) as downs
            from events e join product_dim p using (product_key) where e.persistent group by 1, 2, 3
        )
        select act.*, coalesce(ups, 0) as ups, coalesce(downs, 0) as downs
        from act left join ev using (vendor, week, private_label)
        order by vendor, week
        """
    ).df()
    weekly["week"] = pd.to_datetime(weekly["week"])
    # a week is "in window" if its midpoint falls inside Nov 1 - Feb 5
    weekly["freeze_window"] = in_freeze(weekly["week"] + pd.Timedelta(days=3))
    # Drop the first full-extract transition (before 2024-07-15) and partial last week
    weekly = weekly[(weekly["week"] >= "2024-07-15") & (weekly["week"] < weekly["week"].max())]
    weekly["banner"] = weekly["vendor"].map(LABEL)
    tot = weekly.groupby(["banner", "week", "freeze_window"], as_index=False)[["active", "ups", "downs"]].sum()
    tot["ups_per_1000"] = 1000 * tot["ups"] / tot["active"]
    tot["downs_per_1000"] = 1000 * tot["downs"] / tot["active"]
    save(tot, "06_weekly_change_rates")

    rows = []
    for (banner, pl), g in weekly.groupby(["banner", "private_label"]):
        r = {"banner": banner, "brand_type": "Private label" if pl else "National brand"}
        for w, gg in g.groupby("freeze_window"):
            tag = "in_window" if w else "out_window"
            r[f"ups_per_1000_{tag}"] = 1000 * gg["ups"].sum() / gg["active"].sum()
            r[f"downs_per_1000_{tag}"] = 1000 * gg["downs"].sum() / gg["active"].sum()
            r[f"weeks_{tag}"] = gg["week"].nunique()
        r["increase_rate_ratio_in_vs_out"] = r.get("ups_per_1000_in_window", np.nan) / r.get(
            "ups_per_1000_out_window", np.nan)
        rows.append(r)
    save(pd.DataFrame(rows), "06b_freeze_window_test")

    # Week-level rate comparison with a simple two-sample test on weekly rates (all brands)
    from scipy import stats

    t_rows = []
    for banner, g in tot.groupby("banner"):
        inw, outw = g.loc[g.freeze_window, "ups_per_1000"], g.loc[~g.freeze_window, "ups_per_1000"]
        if len(inw) > 5 and len(outw) > 5:
            u = stats.mannwhitneyu(inw, outw, alternative="two-sided")
            t_rows.append({"banner": banner, "weeks_in": len(inw), "weeks_out": len(outw),
                           "median_weekly_ups_per_1000_in": float(inw.median()),
                           "median_weekly_ups_per_1000_out": float(outw.median()),
                           "mann_whitney_p": float(u.pvalue)})
    save(pd.DataFrame(t_rows), "06c_freeze_window_significance")


# ---------------------------------------------------------------------------
def promotions(con: duckdb.DuckDBPyConnection) -> None:
    df = con.sql(
        """
        select p.vendor, p.private_label,
               count(*) as product_days, avg(d.on_promo::int) as promo_share,
               median(d.promo_depth) as median_depth, avg(d.promo_depth) as mean_depth,
               avg(case when d.on_promo then d.multibuy::int end) as multibuy_share_of_promos,
               avg(case when d.promo_depth is not null then (d.promo_depth >= 0.25)::int end) as share_deep_25pct
        from daily d join product_dim p using (product_key)
        where p.category <> 'Other'
        group by 1, 2
        """
    ).df()
    df["banner"] = df["vendor"].map(LABEL)
    df["brand_type"] = np.where(df["private_label"], "Private label", "National brand")
    save(df.drop(columns=["vendor", "private_label"]), "07_promotions_by_banner")

    # Promotion co-timing lift: P(B on promo | A on promo, same item same day) / P(B on promo)
    rows = []
    for a, b, mtype, group in PAIRS:
        v = pair_daily(con, a, b, mtype)
        r = con.sql(
            f"""select avg(promo_b::int) as base_b, avg(case when promo_a then promo_b::int end) as cond_b,
                       avg(promo_a::int) as base_a, count(*) as n from {v}"""
        ).df().iloc[0]
        rows.append({"banner_a": LABEL[a], "banner_b": LABEL[b], "pair_group": group, "matched_days": int(r.n),
                     "p_b_promo": r.base_b, "p_b_promo_given_a_promo": r.cond_b,
                     "promo_cotiming_lift": r.cond_b / r.base_b if r.base_b else np.nan})
        con.execute(f"drop table {v}")
    save(pd.DataFrame(rows), "07b_promo_cotiming")


# ---------------------------------------------------------------------------
def brand_architecture(con: duckdb.DuckDBPyConnection) -> None:
    df = con.sql(
        """
        select p.vendor, p.category,
               count(distinct p.product_key) as products,
               count(distinct case when p.private_label then p.product_key end) as pl_products,
               sum(d.on_promo::int) as promo_days,
               sum(case when p.private_label then d.on_promo::int else 0 end) as pl_promo_days
        from product_dim p join daily d using (product_key)
        where p.category <> 'Other'
        group by 1, 2
        """
    ).df()
    df["banner"] = df["vendor"].map(LABEL)
    save(df.drop(columns="vendor"), "08_brand_architecture_by_category")
    tot = df.groupby("banner", as_index=False)[["products", "pl_products", "promo_days", "pl_promo_days"]].sum()
    tot["pl_share_assortment"] = tot["pl_products"] / tot["products"]
    tot["pl_share_promo_days"] = tot["pl_promo_days"] / tot["promo_days"]
    save(tot, "08b_brand_architecture")


# ---------------------------------------------------------------------------
def inflation(con: duckdb.DuckDBPyConnection) -> None:
    """Chained matched-model (Jevons) monthly index of regular prices per banner,
    compared with StatCan CPI 'Food purchased from stores' (Ontario, Toronto, Canada)."""
    m = con.sql(
        """
        with mp as (
            select d.vendor, d.product_key, date_trunc('month', d.d) as month, median(d.regular) as p
            from daily d join product_dim pr using (product_key)
            where d.regular is not null and not pr.var_weight and pr.category <> 'Other'
            group by 1, 2, 3
        ), ch as (
            select vendor, month, ln(p / lag(p) over (partition by product_key order by month)) as dl,
                   date_diff('month', lag(month) over (partition by product_key order by month), month) as gap
            from mp
        )
        select vendor, month, avg(dl) as mean_dlog, count(*) as n_items
        from ch where gap = 1 and abs(dl) < 0.5 group by 1, 2 order by 1, 2
        """
    ).df()
    m["month"] = pd.to_datetime(m["month"])
    base = pd.Timestamp("2024-08-01")
    m = m[m["month"] > base]
    m["index"] = m.groupby("vendor")["mean_dlog"].transform(lambda s: 100 * np.exp(s.cumsum()))
    m["banner"] = m["vendor"].map(LABEL)
    base_rows = pd.DataFrame({"vendor": m["vendor"].unique(), "month": base, "index": 100.0})
    base_rows["banner"] = base_rows["vendor"].map(LABEL)
    m = pd.concat([base_rows, m]).sort_values(["vendor", "month"])
    m = m[m["month"] < pd.Timestamp("2026-10-01")]  # October 2026 is a partial month

    cpi = statcan_cpi()
    c = cpi[cpi["product"] == "Food purchased from stores"].copy()
    c = c[c["month"] >= base].sort_values(["geo", "month"])
    base_val = c[c["month"] == base].set_index("geo")["value"]
    c["index"] = 100 * c["value"] / c["geo"].map(base_val)
    c = c.rename(columns={"geo": "banner"})[["banner", "month", "index"]]
    c["banner"] = "CPI food from stores — " + c["banner"]
    out = pd.concat([m[["banner", "month", "index", "n_items"]], c])
    save(out, "09_inflation_vs_cpi")
    last = out.sort_values("month").groupby("banner").tail(1)[["banner", "month", "index"]]
    last["change_since_aug2024"] = last["index"] / 100 - 1
    save(last.sort_values("change_since_aug2024", ascending=False), "09d_inflation_summary")


def statcan_cpi() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.sql(
        f"""
        select REF_DATE as month, GEO as geo, "Products and product groups" as product, VALUE as value
        from read_csv_auto('{RAW / '18100004.csv'}')
        where GEO in ('Canada', 'Ontario', 'Toronto, Ontario')
          and "Products and product groups" in ('All-items', 'Food purchased from stores', 'Dairy products and eggs',
              'Bakery and cereal products (excluding baby food)', 'Meat', 'Fresh fruit and vegetables')
          and REF_DATE >= '2017-01'
        """
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def statcan_context() -> None:
    cpi = statcan_cpi()
    cpi = cpi.sort_values("month")
    cpi["yoy"] = cpi.groupby(["geo", "product"])["value"].pct_change(12)
    save(cpi[cpi["month"] >= "2018-01-01"], "09b_statcan_cpi")
    # Cumulative increase since Jan 2020 for the headline
    piv = cpi[(cpi.geo == "Canada")].pivot(index="month", columns="product", values="value")
    last = piv.index.max()
    rows = []
    for col in piv.columns:
        rows.append({"product": col, "jan2020": piv.loc["2020-01-01", col], "latest_month": last.strftime("%Y-%m"),
                     "latest": piv.loc[last, col], "cum_change_since_jan2020": piv.loc[last, col] / piv.loc["2020-01-01", col] - 1})
    save(pd.DataFrame(rows), "09c_statcan_cumulative")


# ---------------------------------------------------------------------------
# KVI definitions: StatCan product, regex on product name, regex on units.
KVIS = [
    ("Milk, 4 litres", r"\bmilk\b", r"^4\s?l$", r"chocolate|almond|oat|soy|lactose|goat|organic|a2|buttermilk|bean|barista|kefir|evaporated|condensed"),
    ("Milk, 2 litres", r"\bmilk\b", r"^2\s?l$", r"chocolate|almond|oat|soy|lactose|goat|organic|a2|buttermilk|coconut|bean|barista|kefir|evaporated|condensed"),
    ("Butter, 454 grams", r"\bbutter\b", r"^454\s?g$", r"peanut|almond|cookie|nut|garlic|herb|spread|ghee|unsalted|whipped|cashew|plant|cauliflower|sunflower|seed|"
     r"alternative|lactose|bites|european|organic|melt"),
    ("Eggs, 1 dozen", r"\beggs?\b", r"^12(\s|ea\b|un\b|pcs\b|$)",
     r"quail|duck|liquid|whites|chocolate|noodle|organic|free.?run|omega|free.?range|roll|salad|dumpling|salt baked|"
     r"haribo|kit|snack|biscuit|tart|pudding"),
    ("White bread, 675 grams", r"white.*bread|bread.*white", r"^675\s?g$", r"gluten|crumb"),
    ("White sugar, 2 kilograms", r"(white|granulated).*sugar|sugar.*(white|granulated)", r"^2\s?kg$", r"brown|icing|organic|cane"),
    ("Peanut butter, 1 kilogram", r"peanut butter", r"^1\s?kg$", r"powder|cookie|cup|bar|organic"),
    ("Ketchup, 1 litre", r"ketchup", r"^1\s?l$", r"chips|organic"),
    ("Canola oil, 3 litres", r"canola", r"^3\s?l$", r"spray|organic"),
    ("Wheat flour, 2.5 kilograms", r"all.?purpose.*flour|flour.*all.?purpose", r"^2\.5\s?kg$", r"gluten|almond|whole wheat|organic"),
]


def kvi_entry_prices(con: duckdb.DuckDBPyConnection) -> None:
    prod = con.sql(
        "select product_key, vendor, product_name, units, brand, private_label from product_dim where not var_weight"
    ).df()
    name = prod["product_name"].fillna("").str.lower()
    units = prod["units"].fillna("").str.lower().str.strip()
    picks = []
    for kvi, pat, upat, excl in KVIS:
        hit = name.str.contains(pat, regex=True) & units.str.contains(upat, regex=True) & ~name.str.contains(excl, regex=True)
        sub = prod[hit].copy()
        sub["kvi"] = kvi
        picks.append(sub)
    picks = pd.concat(picks)
    save(picks[["kvi", "vendor", "product_name", "units", "brand", "private_label"]].sort_values(["kvi", "vendor"]),
         "10a_kvi_item_matches")
    con.register("kvi_picks", picks[["product_key", "kvi"]])
    window_start = "2026-03-01"
    df = con.sql(
        f"""
        with dd as (
            select k.kvi, d.vendor, d.d, min(d.cur) as entry_shelf, min(d.regular) as entry_regular,
                   median(d.regular) as median_regular
            from daily d join kvi_picks k using (product_key)
            where d.d >= '{window_start}' and d.d < '2026-09-01'
            group by 1, 2, 3
        )
        select kvi, vendor, median(entry_shelf) as entry_shelf_price, median(entry_regular) as entry_regular_price,
               median(median_regular) as median_regular_price, count(*) as days
        from dd group by 1, 2
        """
    ).df()
    df["banner"] = df["vendor"].map(LABEL)
    sc = duckdb.sql(
        f"""
        select Products as kvi, GEO as geo, avg(VALUE) as statcan_avg_price
        from read_csv_auto('{RAW / '18100245.csv'}')
        where GEO in ('Ontario', 'Canada') and REF_DATE between '2026-03' and '2026-08'
        group by 1, 2
        """
    ).df()
    sc["kvi"] = sc["kvi"].str.strip()
    sc = sc.pivot(index="kvi", columns="geo", values="statcan_avg_price").reset_index()
    sc.columns = ["kvi", "statcan_canada_avg", "statcan_ontario_avg"]
    df = df.merge(sc, on="kvi", how="left")
    df["entry_vs_ontario_avg"] = df["entry_shelf_price"] / df["statcan_ontario_avg"] - 1
    save(df.drop(columns="vendor"), "10_kvi_entry_prices")


def change_calendar(con: duckdb.DuckDBPyConnection) -> None:
    """Day of week on which persistent regular-price changes are first observed."""
    df = con.sql(
        """
        select vendor, dayname(d) as weekday, count(*) as changes
        from events where persistent group by 1, 2
        """
    ).df()
    df["share"] = df["changes"] / df.groupby("vendor")["changes"].transform("sum")
    df["banner"] = df["vendor"].map(LABEL)
    save(df.drop(columns="vendor").sort_values(["banner", "share"], ascending=[True, False]), "11_change_weekday")


def sibling_ratio_distribution(con: duckdb.DuckDBPyConnection) -> None:
    """Distribution across SKUs of the Loblaws / No Frills regular-price ratio (last full quarter)."""
    v = pair_daily(con, "Loblaws", "NoFrills", "sku")
    df = con.sql(
        f"""
        with pr as (
            select mkey, median(reg_a / reg_b) as r from {v}
            where reg_a is not null and reg_b is not null and d >= '2026-07-01' and d < '2026-10-01'
            group by mkey having count(*) >= 14
        )
        select count(*) as skus, quantile_cont(r, 0.10) as p10, quantile_cont(r, 0.25) as p25, median(r) as p50,
               quantile_cont(r, 0.75) as p75, quantile_cont(r, 0.90) as p90,
               avg((r between 0.995 and 1.005)::int) as share_identical, avg((r > 1.20)::int) as share_over_20pct,
               avg((r < 0.995)::int) as share_cheaper_at_loblaws
        from pr
        """
    ).df()
    con.execute(f"drop table {v}")
    save(df, "02d_loblaws_nofrills_ratio_distribution")


def main() -> None:
    con = con_()
    print("keys"); build_keys(con)
    print("01"); coverage(con)
    print("02"); identical_item_gaps(con)
    print("03"); banner_index(con)
    print("04/05"); synchronisation(con)
    print("06"); price_freeze(con)
    print("07"); promotions(con)
    print("08"); brand_architecture(con)
    print("09"); inflation(con); statcan_context()
    print("10"); kvi_entry_prices(con)
    print("11"); change_calendar(con); sibling_ratio_distribution(con)
    con.close()


if __name__ == "__main__":
    main()
