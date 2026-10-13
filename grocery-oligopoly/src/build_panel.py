"""Clean Project Hammer scrape data into an analysis-ready daily panel.

Input  : data/raw/hammer-product.csv, data/raw/hammer-raw.csv (Project Hammer)
Output : DuckDB tables in data/processed/grocery.duckdb
           product_dim   one row per vendor x product, with parent, private-label
                         flag, category, normalised GTIN, variable-weight flag
           daily         one row per vendor x product x day, with shelf price,
                         regular price, promotion flags
           events        regular-price change events (date known to +/- 3 days)
         outputs/tables/00_cleaning_log.csv  row counts at every cleaning step

Cleaning rules (all logged):
  * Multi-buy prices ("2/$7.00") are converted to a unit price and flagged promo.
  * "old_price" is the struck-through regular price. When it is present and above
    the current price the item is on promotion and regular = old_price.
  * Per-unit old prices ("1.99/100g") cannot be compared with a pack price, so the
    regular price for that day is unknown (NULL) but the promo flag is set.
  * Prices outside $0.10-$1,000 and old prices more than 5x current are dropped.
  * Multiple scrapes on the same day are collapsed (min current, max old).
  * Sobeys (4 days of data) is excluded.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from config import (
    CATEGORY_RULES,
    DB_PATH,
    EXCLUDED_VENDORS,
    MAX_GAP_DAYS,
    MIN_CHANGE_PCT,
    PERSIST_DAYS,
    PRIVATE_LABEL_PATTERNS,
    RAW,
    TABLES,
    VENDORS,
)

CSV_OPTS = "header=true, all_varchar=true, delim=',', quote='\"', escape='\"', strict_mode=false"
MULTIBUY_RE = r"^\d+\s*/\s*\$?\d+(\.\d+)?$"


def log_step(log: list[dict], step: str, con: duckdb.DuckDBPyConnection, table: str) -> None:
    n = con.sql(f"select count(*) from {table}").fetchone()[0]
    log.append({"step": step, "table": table, "rows": int(n)})
    print(f"{step:<55s} {table:<14s} {n:>12,d}")


def classify(series: pd.Series, rules: list[tuple[str, str]]) -> pd.Series:
    out = pd.Series("Other", index=series.index, dtype="object")
    unassigned = pd.Series(True, index=series.index)
    for label, pattern in rules:
        hit = unassigned & series.str.contains(pattern, regex=True, na=False)
        out[hit] = label
        unassigned &= ~hit
    return out


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    log: list[dict] = []

    # 1. Load raw CSVs (all text; parsing is explicit below).
    con.execute(f"create or replace table raw_product as select * from read_csv('{RAW / 'hammer-product.csv'}', {CSV_OPTS})")
    con.execute(
        f"create or replace table raw_price as select * from read_csv('{RAW / 'hammer-raw.csv'}', {CSV_OPTS}, ignore_errors=true)"
    )
    log_step(log, "1. raw products loaded", con, "raw_product")
    log_step(log, "1. raw price observations loaded", con, "raw_price")

    # 2. Product dimension.
    prod = con.sql(
        """
        select id, vendor, coalesce(nullif(sku, ''), id) as sku, product_name, units, brand, upc
        from raw_product
        """
    ).df()
    prod = prod[prod["vendor"].isin(VENDORS) & ~prod["vendor"].isin(EXCLUDED_VENDORS)].copy()
    prod["product_key"] = prod["vendor"] + "|" + prod["sku"].astype(str)
    text = (prod["brand"].fillna("") + " " + prod["product_name"].fillna("")).str.lower().str.strip()
    name = prod["product_name"].fillna("").str.lower()
    prod["private_label"] = False
    for vendor, pattern in PRIVATE_LABEL_PATTERNS.items():
        m = prod["vendor"] == vendor
        prod.loc[m, "private_label"] = text[m].str.contains(pattern, regex=True) | name[m].str.contains(pattern, regex=True)
    prod["category"] = classify(name, CATEGORY_RULES)
    digits = prod["upc"].fillna("").map(lambda s: re.sub(r"\D", "", s).lstrip("0"))
    prod["gtin"] = digits.where(digits.str.len() >= 7)
    prod["var_weight"] = (
        prod["sku"].astype(str).str.upper().str.endswith("_KG")
        | name.str.contains(r"per kg|/kg|per lb|/lb", regex=True)
    )
    prod["parent"] = prod["vendor"].map(lambda v: VENDORS[v]["parent"])
    prod["role"] = prod["vendor"].map(lambda v: VENDORS[v]["role"])
    prod["banner"] = prod["vendor"].map(lambda v: VENDORS[v]["label"])
    con.register("prod_df", prod)
    con.execute("create or replace table product_ids as select id, product_key from prod_df")
    con.execute(
        """
        create or replace table product_dim as
        select product_key, any_value(vendor) as vendor, any_value(banner) as banner, any_value(parent) as parent,
               any_value(role) as role, any_value(sku) as sku, any_value(product_name) as product_name,
               any_value(units) as units, any_value(brand) as brand, max(gtin) as gtin,
               bool_or(private_label) as private_label, any_value(category) as category,
               bool_or(var_weight) as var_weight
        from prod_df group by product_key
        """
    )
    log_step(log, "2. product dimension (excl. Sobeys)", con, "product_dim")

    # 3. Parse observations.
    con.execute(
        f"""
        create or replace table obs as
        with r as (
            select i.product_key, try_cast(p.nowtime as date) as d,
                   trim(p.current_price) as cp, trim(p.old_price) as op, coalesce(p.other, '') as other
            from raw_price p join product_ids i on p.product_id = i.id
        )
        select product_key, d,
               regexp_matches(cp, '{MULTIBUY_RE}') as multibuy,
               case when regexp_matches(cp, '{MULTIBUY_RE}')
                    then try_cast(regexp_extract(cp, '\\$?(\\d+(\\.\\d+)?)$', 1) as double)
                         / nullif(try_cast(regexp_extract(cp, '^(\\d+)', 1) as double), 0)
                    else try_cast(replace(replace(cp, '$', ''), ',', '') as double) end as cur,
               try_cast(replace(replace(op, '$', ''), ',', '') as double) as old,
               (op is not null and op <> '' and try_cast(replace(replace(op, '$', ''), ',', '') as double) is null)
                   as old_unparsed,
               other as other_raw,
               lower(other) like '%out of stock%' as oos,
               regexp_matches(lower(other), 'sale|rollback|\\d+ for \\$|min \\d') as promo_text
        from r
        """
    )
    log_step(log, "3. observations joined to products", con, "obs")
    con.execute("delete from obs where d is null or cur is null")
    log_step(log, "3a. drop unparseable date/price", con, "obs")
    con.execute("delete from obs where cur < 0.10 or cur > 1000")
    log_step(log, "3b. drop prices outside $0.10-$1,000", con, "obs")
    # Some scrapes record prices in cents ("377" for $3.77). Repair a value when it is
    # >30x the product's median price and dividing by 100 brings it back in range.
    con.execute(
        """
        create or replace temp table pmed as
        select product_key, median(cur) as med from obs group by product_key
        """
    )
    n_fix = con.sql(
        """select count(*) from obs join pmed using (product_key)
           where cur > 30 * med and cur / 100 between med / 3 and med * 3"""
    ).fetchone()[0]
    con.execute(
        """
        update obs set cur = cur / 100 from pmed
        where obs.product_key = pmed.product_key and cur > 30 * med and cur / 100 between med / 3 and med * 3
        """
    )
    log.append({"step": "3c. cents-scaled prices repaired", "table": "obs", "rows": int(n_fix)})
    print(f"{'3c. cents-scaled prices repaired':<55s} {'obs':<14s} {n_fix:>12,d}")
    con.execute("update obs set old = null where old is not null and (old > 5 * cur or old <= 0)")
    # Conditional deals carried only in the 'other' text — "sale $3.50 MIN 2" (Loblaw banners) or
    # "2 for $7" — leave current_price at the single-unit regular price. Convert them to an
    # effective promotional unit price and keep the single-unit price as the regular price.
    con.execute(
        """
        create or replace temp table deals as
        select * from (
            select rowid as rid,
                   coalesce(
                       try_cast(regexp_extract(other_l, 'sale\\s*:?\\s*\\$(\\d+(\\.\\d+)?)\\s*min', 1) as double),
                       try_cast(regexp_extract(other_l, '(\\d+)\\s*for\\s*\\$(\\d+(\\.\\d+)?)', 2) as double)
                         / nullif(try_cast(regexp_extract(other_l, '(\\d+)\\s*for\\s*\\$', 1) as double), 0)
                   ) as deal
            from (select rowid, lower(other_raw) as other_l from obs where other_raw <> '')
        ) x where deal is not null
        """
    )
    con.execute(
        """
        update obs set old = cur, cur = deals.deal, promo_text = true
        from deals
        where obs.rowid = deals.rid and obs.old is null and deals.deal < obs.cur * 0.995 and deals.deal > obs.cur * 0.3
        """
    )
    n_deal = con.sql("select count(*) from obs where old is not null and abs(old - cur) > 0 and rowid in (select rid from deals)").fetchone()[0]
    print(f"{'3d. text-only deals converted to effective unit price':<55s} {'obs':<14s} {n_deal:>12,d}")
    log.append({"step": "3d. text-only deals converted to effective unit price", "table": "obs", "rows": int(n_deal)})

    # 4. Collapse to one row per product-day.
    con.execute(
        """
        create or replace table daily as
        with g as (
            select product_key, d, min(cur) as cur, max(old) as old, bool_or(multibuy) as multibuy,
                   bool_or(old_unparsed) as old_unparsed, bool_or(oos) as oos, bool_or(promo_text) as promo_text
            from obs group by product_key, d
        )
        select g.*, p.vendor,
               (coalesce(old > cur * 1.005, false) or multibuy or old_unparsed or promo_text) as on_promo,
               case when old > cur * 1.005 then old
                    when multibuy or old_unparsed or promo_text then null
                    else cur end as regular,
               case when old > cur * 1.005 then 1 - cur / old end as promo_depth
        from g join product_dim p using (product_key)
        """
    )
    log_step(log, "4. collapsed to product-day panel", con, "daily")
    con.execute("drop table obs")

    # 5. Regular-price change events. A change is "persistent" unless the very next
    #    change, within 14 days, returns the price to where it started — the signature of
    #    an unflagged temporary price reduction rather than a new regular price
    #    (in the spirit of the sales filters in Nakamura & Steinsson, 2008).
    con.execute(
        f"""
        create or replace table events as
        with k as (
            select product_key, vendor, d, regular,
                   lag(regular) over w as prev_regular, lag(d) over w as prev_d,
                   lead(regular) over w as next_regular
            from daily where regular is not null
            window w as (partition by product_key order by d)
        ), e as (
            select k.product_key, k.vendor, k.d, k.prev_d, k.prev_regular, k.regular,
                   k.regular / k.prev_regular - 1 as pct, sign(k.regular - k.prev_regular)::int as dir
            from k join product_dim p using (product_key)
            where prev_regular is not null
              and abs(regular / prev_regular - 1) >= {MIN_CHANGE_PCT}
              and abs(regular - prev_regular) >= 0.05
              and date_diff('day', prev_d, d) <= {MAX_GAP_DAYS}
              and regular / prev_regular between 1/3.0 and 3.0
              -- one-day spikes that immediately revert are treated as scrape errors
              and not (next_regular is not null and abs(next_regular - prev_regular) < 0.005
                       and abs(regular / prev_regular - 1) > 0.5)
              and not p.var_weight
        )
        select e.*,
               not coalesce(date_diff('day', d, lead(d) over w) <= {PERSIST_DAYS}
                            and abs(lead(regular) over w - prev_regular) < 0.005, false) as persistent
        from e
        window w as (partition by product_key order by d)
        """
    )
    log_step(log, "5. regular-price change events (all)", con, "events")
    n_p = con.sql("select count(*) from events where persistent").fetchone()[0]
    log.append({"step": "5a. persistent regular-price changes", "table": "events", "rows": int(n_p)})
    print(f"{'5a. persistent regular-price changes':<55s} {'events':<14s} {n_p:>12,d}")

    pd.DataFrame(log).to_csv(TABLES / "00_cleaning_log.csv", index=False)
    con.close()


if __name__ == "__main__":
    main()
