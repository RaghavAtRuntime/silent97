"""CFPB Consumer Complaint Database: the semi-synthetic real-text benchmark.

Source: the public export of the CFPB Consumer Complaint Database
(``https://files.consumerfinance.gov/ccdb/complaints.csv.zip``), US government data.

Kept: complaints received from 2015-01-01 to 2017-04-30 (inclusive) with a non-empty
consumer narrative and a "Consumer disputed?" value of Yes or No. The CFPB stopped
collecting that field in April 2017. The outcome is ``disputed``, a binary proxy for
dissatisfaction, not NPS.

Labelers may see only ``narrative``, ``product``, ``issue`` and ``state``
(:data:`LABELER_FEATURES`); :func:`assert_labeler_columns` enforces this. Everything
else in the export (company responses and so on) is dropped on load.

To run it through the NPS estimators, disputed is coded as score 0 (detractor) and
not disputed as 7 (passive) (:func:`dispute_scores`). The promoter indicator is then
identically 0, so NPS = -100 x dispute rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

__all__ = [
    "COLUMNS",
    "EXPORT_URL",
    "LABELER_FEATURES",
    "OUTCOME",
    "assert_labeler_columns",
    "build",
    "dispute_scores",
    "download",
    "filter_complaints",
    "load_export",
    "main",
    "tabular_design",
]

EXPORT_URL: Final = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
WINDOW: Final = (date(2015, 1, 1), date(2017, 4, 30))

COLUMNS: Final = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Issue": "issue",
    "State": "state",
    "Consumer complaint narrative": "narrative",
    "Consumer disputed?": "disputed",
}
"""Export column -> our column. Nothing else is read."""

LABELER_FEATURES: Final = ("narrative", "product", "issue", "state")
OUTCOME: Final = ("disputed",)


def assert_labeler_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Raise ``ValueError`` if anything outside :data:`LABELER_FEATURES` is requested."""
    extra = sorted(set(columns) - set(LABELER_FEATURES))
    if extra:
        raise ValueError(f"CFPB labelers may only see {LABELER_FEATURES}; got also {extra}")


def download(url: str, dest: str | Path, *, fetch: Callable[[str], bytes] | None = None) -> Path:
    """Download ``url`` to ``dest`` once (skipped if ``dest`` exists), atomically."""
    out = Path(dest)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    if fetch is not None:
        part.write_bytes(fetch(url))
    else:  # pragma: no cover - network
        with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    os.replace(part, out)
    return out


def _extract(path: Path) -> Path:
    """Extract the single CSV inside ``path`` next to it (once), atomically."""
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one CSV inside the zip, found {members}")
        target = path.parent / Path(members[0]).name
        if target.exists() and target.stat().st_size == zf.getinfo(members[0]).file_size:
            return target
        part = target.with_name(target.name + ".part")
        with zf.open(members[0]) as src, open(part, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    os.replace(part, target)
    return target


def load_export(path: str | Path) -> pl.LazyFrame:
    """Lazily scan the export (``.csv``, or ``.csv.zip`` extracted once), keeping :data:`COLUMNS`.

    The full export is several GB; scanning keeps memory to the kept columns and rows.
    """
    src = Path(path)
    csv_path = _extract(src) if src.suffix == ".zip" else src
    lf = pl.scan_csv(csv_path, infer_schema=False)  # every column as text; parsed below
    return lf.select(list(COLUMNS)).rename(COLUMNS)


def filter_complaints(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Apply the benchmark's inclusion rules and type the columns.

    Dates are accepted as ``YYYY-MM-DD`` or ``MM/DD/YYYY`` (both appear in exports).
    """
    parsed = df.lazy().with_columns(
        pl.coalesce(
            pl.col("date_received").str.to_date("%Y-%m-%d", strict=False),
            pl.col("date_received").str.to_date("%m/%d/%Y", strict=False),
        ).alias("date_received"),
        pl.col("narrative").fill_null("").str.strip_chars(),
        pl.col("disputed").fill_null("").str.strip_chars(),
        *(pl.col(c).fill_null("").str.strip_chars() for c in ("product", "issue", "state")),
    )
    out = (
        parsed.filter(pl.col("date_received").is_between(*WINDOW, closed="both"))
        .filter(pl.col("narrative") != "")
        .filter(pl.col("disputed").is_in(["Yes", "No"]))
        .with_columns((pl.col("disputed") == "Yes").alias("disputed"))
    )
    order = pl.col("complaint_id").cast(pl.Int64, strict=False)  # numeric ids, sorted as numbers
    return out.sort(order, "complaint_id").select(list(COLUMNS.values())).collect()


def _drop_counts(lf: pl.LazyFrame) -> dict[str, int]:
    dates = pl.coalesce(
        pl.col("date_received").str.to_date("%Y-%m-%d", strict=False),
        pl.col("date_received").str.to_date("%m/%d/%Y", strict=False),
    )
    in_window = dates.is_between(*WINDOW, closed="both").fill_null(False)
    has_text = pl.col("narrative").fill_null("").str.strip_chars() != ""
    has_dispute = pl.col("disputed").fill_null("").str.strip_chars().is_in(["Yes", "No"])
    c = (
        lf.select(
            pl.len().alias("total"),
            in_window.sum().alias("window"),
            (in_window & has_text).sum().alias("text"),
            (in_window & has_text & has_dispute).sum().alias("kept"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "outside_window": int(c["total"] - c["window"]),
        "no_narrative": int(c["window"] - c["text"]),
        "no_dispute_value": int(c["text"] - c["kept"]),
    }


def build(source: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Filter the export at ``source`` into ``cfpb.parquet`` plus ``cfpb_manifest.json``."""
    src = Path(source)
    raw = load_export(src)
    df = filter_complaints(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "cfpb.parquet"
    df.write_parquet(parquet, compression="zstd", statistics=False)
    manifest = {
        "source": src.name,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "window": [WINDOW[0].isoformat(), WINDOW[1].isoformat()],
        "n_export_rows": int(raw.select(pl.len()).collect().item()),
        "n_rows": df.height,
        "dispute_rate": float(df["disputed"].to_numpy().mean()) if df.height else None,
        "dropped": _drop_counts(raw),
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
    }
    path = out / "cfpb_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return parquet, path


def dispute_scores(disputed: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Code disputed as score 0 (detractor) and not disputed as 7 (passive)."""
    return np.where(np.asarray(disputed, dtype=bool), 0, 7).astype(np.int64)


def tabular_design(df: pl.DataFrame) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """One-hot encode product, issue and state (the tabular labeler's only inputs)."""
    assert_labeler_columns(["product", "issue", "state"])
    cols: list[NDArray[np.float64]] = []
    names: list[str] = []
    for field in ("product", "issue", "state"):
        values = df[field].to_numpy().astype(str)
        for level in sorted(set(values.tolist())):
            cols.append((values == level).astype(np.float64))
            names.append(f"{field}={level}")
    return np.column_stack(cols), tuple(names)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs the network
    """``python -m silent97.data.cfpb``: download the export once and build the Parquet."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and filter the CFPB export")
    parser.add_argument("--url", default=EXPORT_URL)
    parser.add_argument("--raw", default="data/raw/cfpb/complaints.csv.zip")
    parser.add_argument("--out", default="data/processed/cfpb")
    args = parser.parse_args(argv)
    src = download(args.url, args.raw)
    parquet, manifest = build(src, args.out)
    info = json.loads(manifest.read_text())
    print(f"wrote {parquet}: {info['n_rows']} complaints, dispute rate {info['dispute_rate']:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
