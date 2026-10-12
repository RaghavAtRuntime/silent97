"""Tests for the Amazon Reviews 2023 loader on a small file in the dataset's format."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.data import amazon

ROWS = [
    # rating, title, text, user, asin, timestamp (ms), helpful, verified, images
    (5.0, "Great", "Worked perfectly.", "u1", "a1", 1_600_000_000_000, 0, True, []),
    (1.0, "Scam", "Balance was zero when it arrived.", "u2", "a1", 1_500_000_000_000, 3, True, []),
    (4.0, "", "Fine card, slow delivery.", "u3", "a2", 1_550_000_000_000, 0, False, ["x"]),
    (3.0, "Meh", "", "u4", "a2", 1_560_000_000_000, 1, True, []),
    (2.0, "", "   ", "u5", "a3", 1_570_000_000_000, 0, True, []),  # dropped: no text
    (5.0, "Great", "Worked perfectly.", "u1", "a1", 1_600_000_000_000, 0, True, []),  # duplicate
]


def _write(path: Path) -> Path:
    with open(path, "w") as fh:
        for r, title, text, user, asin, ts, helpful, verified, images in ROWS:
            fh.write(
                json.dumps(
                    {
                        "rating": r,
                        "title": title,
                        "text": text,
                        "images": images,
                        "asin": asin,
                        "parent_asin": asin,
                        "user_id": user,
                        "timestamp": ts,
                        "helpful_vote": helpful,
                        "verified_purchase": verified,
                    }
                )
                + "\n"
            )
    return path


def test_load_filters_dedupes_and_sorts(tmp_path: Path) -> None:
    df = amazon.load(_write(tmp_path / "Gift_Cards.jsonl"))
    assert df.height == 4
    # Sorted by time: u2 (2017), u3, u4, u1.
    assert df["user_id"].to_list() == ["u2", "u3", "u4", "u1"]
    assert df["score"].to_list() == [2, 8, 6, 10]  # 2 x stars
    assert df["review_id"].n_unique() == 4
    assert df.filter(pl.col("user_id") == "u3")["has_images"].to_list() == [True]


def test_scores_map_stars_to_nps_groups(tmp_path: Path) -> None:
    df = amazon.load(_write(tmp_path / "g.jsonl"))
    s = df["score"].to_numpy()
    promoter, detractor = s >= 9, s <= 6
    stars = df["rating"].to_numpy()
    np.testing.assert_array_equal(promoter, stars == 5)
    np.testing.assert_array_equal(detractor, stars <= 3)


def test_build_is_deterministic_with_manifest(tmp_path: Path) -> None:
    src = _write(tmp_path / "Gift_Cards.jsonl")
    pa, ma = amazon.build(src, tmp_path / "a")
    pb, _ = amazon.build(src, tmp_path / "b")
    assert pa.read_bytes() == pb.read_bytes()
    manifest = json.loads(ma.read_text())
    assert manifest["n_rows"] == 4
    assert manifest["dropped"] == {"no_text": 1, "duplicate": 1}
    assert manifest["true_nps"] == pytest.approx(100 * (1 / 4 - 2 / 4))
    assert len(manifest["source_sha256"]) == 64


def test_labeler_columns_exclude_rating() -> None:
    amazon.assert_labeler_columns(amazon.LABELER_FEATURES)
    for leak in ("rating", "score"):
        with pytest.raises(ValueError, match=leak):
            amazon.assert_labeler_columns([*amazon.LABELER_FEATURES, leak])


def test_tabular_design_uses_metadata_only(tmp_path: Path) -> None:
    df = amazon.load(_write(tmp_path / "g.jsonl"))
    x, names = amazon.tabular_design(df)
    assert x.shape == (4, len(names))
    assert not any("rating" in n or "score" in n for n in names)
    assert {"verified_purchase", "has_images", "helpful_any"} <= set(names)


def test_source_is_pinned() -> None:
    assert amazon.SOURCE_URL.startswith(
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    )
    assert len(amazon.SOURCE_SHA256) == 64
