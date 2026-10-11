"""CFPB Consumer Complaint Database: the semi-synthetic real-text benchmark.

Source: the public export of the CFPB Consumer Complaint Database, US government data.
The current official export (``https://files.consumerfinance.gov/ccdb/complaints.csv.zip``)
no longer carries the "Consumer complaint narrative" or "Consumer disputed?" columns
(checked October 2026), so the default source is a public CC0 mirror of the older
export on Hugging Face (:data:`MIRROR_FILES`, the ``has-text`` subset of
``BEE-spoke-data/consumer-finance-complaints``), pinned by SHA-256. :func:`crosscheck`
compares the kept complaints' date, product, issue and state against the current
official export, and the result is recorded in the manifest.

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
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

__all__ = [
    "COLUMNS",
    "EXPORT_URL",
    "LABELER_FEATURES",
    "MIRROR_FILES",
    "OUTCOME",
    "assert_labeler_columns",
    "build",
    "crosscheck",
    "dispute_scores",
    "download",
    "filter_complaints",
    "load_export",
    "main",
    "tabular_design",
]

EXPORT_URL: Final = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
WINDOW: Final = (date(2015, 1, 1), date(2017, 4, 30))

_MIRROR: Final = (
    "https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints/resolve/"
    "main/has-text/train-{:05d}-of-00003.parquet"
)
MIRROR_FILES: Final = (
    (_MIRROR.format(0), "60902c32f0a6b3db25a378570d421765d0a15ced396c0c471a9d60e49b38d833"),
    (_MIRROR.format(1), "367c06699e95a391916b4e73685112b3fe2d1242f9160654dcb995a73908a4e4"),
    (_MIRROR.format(2), "c9ad758d64795e780ef6b1f6a1367b7e18877db0a6be530a50777c2d205d8127"),
)
"""(URL, SHA-256) of the CC0 mirror's shards with a narrative; same columns as the export."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: str | Path,
    *,
    fetch: Callable[[str], bytes] | None = None,
    sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``dest`` once (skipped if ``dest`` exists), atomically.

    With ``sha256``, the file (new or existing) must have that digest, or ``ValueError``
    is raised and a new download is discarded.
    """
    out = Path(dest)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        if fetch is not None:
            part.write_bytes(fetch(url))
        else:  # pragma: no cover - network
            with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        if sha256 is not None and (got := _sha256(part)) != sha256:
            part.unlink()
            raise ValueError(f"{url}: SHA-256 {got} does not match the pinned {sha256}")
        os.replace(part, out)
    elif sha256 is not None and (got := _sha256(out)) != sha256:
        raise ValueError(f"{out}: SHA-256 {got} does not match the pinned {sha256}")
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


def _parquet_files(src: Path) -> list[Path]:
    return sorted(src.glob("*.parquet")) if src.is_dir() else [src]


def _scan(path: str | Path, columns: list[str]) -> pl.LazyFrame:
    """Scan ``columns`` of an export (``.csv``, ``.csv.zip``, ``.parquet`` or a directory
    of ``.parquet`` shards), every column as text."""
    src = Path(path)
    if src.is_dir() or src.suffix == ".parquet":
        files = _parquet_files(src)
        if not files:
            raise ValueError(f"{src}: no .parquet files")
        lf = pl.scan_parquet(files)
    else:
        csv_path = _extract(src) if src.suffix == ".zip" else src
        lf = pl.scan_csv(csv_path, infer_schema=False)  # every column as text; parsed below
    return lf.select(pl.col(c).cast(pl.String) for c in columns)


def load_export(path: str | Path) -> pl.LazyFrame:
    """Lazily scan the export, keeping :data:`COLUMNS` as text under our names.

    ``path`` is the CSV (``.csv``, or ``.csv.zip`` extracted once) or the mirror's
    Parquet (one ``.parquet`` file or a directory of shards). The full export is several
    GB; scanning keeps memory to the kept columns and rows.
    """
    return _scan(path, list(COLUMNS)).rename(COLUMNS)


def _parse_dates(col: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(col).str.to_date("%Y-%m-%d", strict=False),
        pl.col(col).str.to_date("%m/%d/%Y", strict=False),
    )


_CHECKED: Final = ("date_received", "product", "issue", "state")


def crosscheck(kept: pl.DataFrame, official: str | Path) -> dict[str, Any]:
    """Compare kept complaints with the official export on the fields both still carry.

    Matches on complaint id; counts how many kept complaints are found in ``official``
    and, among those, how many differ in each of date, product, issue and state (text
    stripped, dates parsed). Neither narrative nor dispute can be checked: the current
    official export no longer has them.
    """
    names = {k: v for k, v in COLUMNS.items() if v in ("complaint_id", *_CHECKED)}
    ref = (
        _scan(official, list(names))
        .rename(names)
        .with_columns(
            _parse_dates("date_received"),
            *(pl.col(c).fill_null("").str.strip_chars() for c in _CHECKED[1:]),
        )
        .join(kept.lazy().select("complaint_id"), on="complaint_id", how="semi")
        .unique("complaint_id", keep="first")
        .collect()
    )
    both = kept.select("complaint_id", *_CHECKED).join(
        ref, on="complaint_id", how="inner", suffix="_official"
    )
    diff = {c: both[c].ne_missing(both[f"{c}_official"]) for c in _CHECKED}
    any_diff = pl.Series([False] * both.height)
    for d in diff.values():
        any_diff = any_diff | d
    return {
        "n_checked": kept.height,
        "n_found": both.height,
        "n_identical": int((~any_diff).sum()),
        "mismatches": {c: int(d.sum()) for c, d in diff.items()},
        "official_source": Path(official).name,
    }


def filter_complaints(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Apply the benchmark's inclusion rules and type the columns.

    Dates are accepted as ``YYYY-MM-DD`` or ``MM/DD/YYYY`` (both appear in exports).
    """
    parsed = df.lazy().with_columns(
        _parse_dates("date_received"),
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
    dates = _parse_dates("date_received")
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


def build(
    source: str | Path, out_dir: str | Path, *, crosscheck_export: str | Path | None = None
) -> tuple[Path, Path]:
    """Filter the export at ``source`` into ``cfpb.parquet`` plus ``cfpb_manifest.json``.

    ``source`` is anything :func:`load_export` reads. With ``crosscheck_export`` (the
    current official export), the manifest also records :func:`crosscheck`.
    """
    src = Path(source)
    raw = load_export(src)
    df = filter_complaints(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "cfpb.parquet"
    df.write_parquet(parquet, compression="zstd", statistics=False)
    files = _parquet_files(src) if src.is_dir() else [src]
    file_sha = {f.name: _sha256(f) for f in files}
    manifest: dict[str, Any] = {
        "source": src.name,
        "source_files": file_sha,
        "source_sha256": (
            file_sha[src.name]
            if not src.is_dir()
            else hashlib.sha256(json.dumps(file_sha, sort_keys=True).encode()).hexdigest()
        ),
        "window": [WINDOW[0].isoformat(), WINDOW[1].isoformat()],
        "n_export_rows": int(raw.select(pl.len()).collect().item()),
        "n_rows": df.height,
        "dispute_rate": float(df["disputed"].to_numpy().mean()) if df.height else None,
        "dropped": _drop_counts(raw),
        "parquet_sha256": _sha256(parquet),
    }
    if crosscheck_export is not None:
        manifest["crosscheck"] = crosscheck(df, crosscheck_export)
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
    """``python -m silent97.data.cfpb``: download the data once and build the Parquet.

    ``--source mirror`` (default) downloads the pinned mirror shards, plus the current
    official export for :func:`crosscheck`. ``--source export`` reads the official
    export alone, which works only with a copy that still has the narrative and
    dispute columns.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download and filter the CFPB complaints")
    parser.add_argument("--source", choices=("mirror", "export"), default="mirror")
    parser.add_argument("--url", default=EXPORT_URL, help="official export URL")
    parser.add_argument("--raw", default="data/raw/cfpb/complaints.csv.zip")
    parser.add_argument("--mirror-dir", default="data/raw/cfpb_mirror")
    parser.add_argument("--out", default="data/processed/cfpb")
    args = parser.parse_args(argv)
    official = download(args.url, args.raw)
    if args.source == "mirror":
        for url, sha in MIRROR_FILES:
            download(url, Path(args.mirror_dir) / url.rsplit("/", 1)[-1], sha256=sha)
        parquet, manifest = build(args.mirror_dir, args.out, crosscheck_export=official)
    else:
        parquet, manifest = build(official, args.out)
    info = json.loads(manifest.read_text())
    print(f"wrote {parquet}: {info['n_rows']} complaints, dispute rate {info['dispute_rate']:.4f}")
    if "crosscheck" in info:
        print(f"crosscheck against the official export: {info['crosscheck']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
