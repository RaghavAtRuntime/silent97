"""Amazon Reviews 2023, Gift Cards: a real-text benchmark with self-reported ratings.

Source: McAuley Lab, *Amazon Reviews 2023* (Hou et al., "Bridging Language and Items
for Retrieval and Recommendation", arXiv 2403.03952), category ``Gift_Cards``,
distributed on Hugging Face (``McAuley-Lab/Amazon-Reviews-2023``) for research use.
The file is pinned by SHA-256.

Why this dataset. The CFPB benchmark's outcome (disputing) is close to unpredictable
from public data (``docs/progress/phase-3.md``), so it cannot show efficiency gains.
Here the outcome is the customer's own 1-5 star rating, the closest public analogue to
an NPS answer, and gift cards are a stored-value consumer-finance product.

Star ratings map to the 0-10 NPS scale as ``score = 2 x stars``: 5 stars -> 10
(promoter), 4 -> 8 (passive), 1-3 -> 2, 4, 6 (detractor). This keeps the usual
"top box = promoter, next box = passive" reading of a 5-point scale.

Kept: every review with a non-empty title or text; exact duplicates of (user, product,
timestamp) are dropped. Labelers see only the title and text (:data:`LABELER_FEATURES`)
or metadata derived without the rating (:func:`tabular_design`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

__all__ = [
    "LABELER_FEATURES",
    "OUTCOME",
    "SOURCE_SHA256",
    "SOURCE_URL",
    "assert_labeler_columns",
    "build",
    "load",
    "main",
    "tabular_design",
]

SOURCE_URL: Final = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Gift_Cards.jsonl"
)
SOURCE_SHA256: Final = "b7ceead0dd6378eebfc6f4736402f85cc08e8dcb328f813c95ca68fc62d9544e"

LABELER_FEATURES: Final = ("title", "text")
OUTCOME: Final = ("rating", "score")
_LENGTH_EDGES: Final = (30, 60, 120, 250)


def assert_labeler_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Raise ``ValueError`` if a labeler asks for the outcome or anything unexpected."""
    extra = sorted(set(columns) - set(LABELER_FEATURES))
    if extra:
        raise ValueError(f"review labelers may only see {LABELER_FEATURES}; got also {extra}")


def _read(path: str | Path) -> pl.DataFrame:
    raw = pl.read_ndjson(
        path,
        schema={
            "rating": pl.Float64,
            "title": pl.String,
            "text": pl.String,
            "images": pl.List(pl.String),
            "asin": pl.String,
            "user_id": pl.String,
            "timestamp": pl.Int64,
            "helpful_vote": pl.Int64,
            "verified_purchase": pl.Boolean,
        },
    )
    return raw.with_columns(
        pl.col("title").fill_null("").str.strip_chars(),
        pl.col("text").fill_null("").str.strip_chars(),
        (pl.col("images").list.len().fill_null(0) > 0).alias("has_images"),
    ).drop("images")


def _filter(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    has_text = raw.filter((pl.col("title") != "") | (pl.col("text") != ""))
    unique = has_text.unique(["user_id", "asin", "timestamp"], keep="first", maintain_order=True)
    dropped = {
        "no_text": raw.height - has_text.height,
        "duplicate": has_text.height - unique.height,
    }
    out = (
        unique.with_columns(
            (pl.col("rating").round(0).cast(pl.Int64) * 2).alias("score"),
            pl.concat_str(
                ["user_id", "asin", pl.col("timestamp").cast(pl.String)], separator="|"
            ).alias("review_id"),
        )
        .sort("timestamp", "user_id", "asin")
        .select(
            "review_id",
            "timestamp",
            "user_id",
            "asin",
            "title",
            "text",
            "helpful_vote",
            "verified_purchase",
            "has_images",
            "rating",
            "score",
        )
    )
    if not out["rating"].is_between(1, 5).all():
        raise ValueError("ratings outside 1..5")
    return out, dropped


def load(path: str | Path) -> pl.DataFrame:
    """Read the category file and apply the inclusion rules (sorted by time)."""
    return _filter(_read(path))[0]


def build(source: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``amazon.parquet`` and ``amazon_manifest.json`` from the raw file."""
    src = Path(source)
    raw = _read(src)
    df, dropped = _filter(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "amazon.parquet"
    df.write_parquet(parquet, compression="zstd", statistics=False)
    s = df["score"].to_numpy()
    manifest: dict[str, Any] = {
        "source": src.name,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "n_raw_rows": raw.height,
        "n_rows": df.height,
        "dropped": dropped,
        "true_nps": float(100 * ((s >= 9).mean() - (s <= 6).mean())) if df.height else None,
        "rating_counts": {
            str(k): int(v) for k, v in df.group_by("rating").len().sort("rating").iter_rows()
        },
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
    }
    path = out / "amazon_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return parquet, path


def tabular_design(df: pl.DataFrame) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Metadata features (no rating): verified, images, any helpful vote, year, length band."""
    cols: list[NDArray[np.float64]] = []
    names: list[str] = []
    for name in ("verified_purchase", "has_images"):
        cols.append(df[name].to_numpy().astype(np.float64))
        names.append(name)
    cols.append((df["helpful_vote"].to_numpy() > 0).astype(np.float64))
    names.append("helpful_any")
    years = df.select(pl.from_epoch("timestamp", time_unit="ms").dt.year()).to_series().to_numpy()
    for year in sorted(set(years.tolist())):
        cols.append((years == year).astype(np.float64))
        names.append(f"year={year}")
    length = (df["title"].str.len_chars() + df["text"].str.len_chars()).to_numpy()
    band = np.searchsorted(np.asarray(_LENGTH_EDGES), length, side="right")
    for b in range(len(_LENGTH_EDGES) + 1):
        cols.append((band == b).astype(np.float64))
        names.append(f"length_band={b}")
    return np.column_stack(cols), tuple(names)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs the network
    """``python -m silent97.data.amazon``: download the pinned file once and build."""
    import argparse

    from silent97.data.cfpb import download

    parser = argparse.ArgumentParser(description="Download and build Amazon Reviews 2023")
    parser.add_argument("--raw", default="data/raw/amazon/Gift_Cards.jsonl")
    parser.add_argument("--out", default="data/processed/amazon")
    args = parser.parse_args(argv)
    src = download(SOURCE_URL, args.raw, sha256=SOURCE_SHA256)
    parquet, manifest = build(src, args.out)
    info = json.loads(manifest.read_text())
    print(f"wrote {parquet}: {info['n_rows']} reviews, true NPS {info['true_nps']:+.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
