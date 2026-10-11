"""Tests for the CFPB loader on a small synthetic file in the export's format."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from silent97.data import cfpb

HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Consumer consent provided?", "Submitted via",
    "Date sent to company", "Company response to consumer", "Timely response?",
    "Consumer disputed?", "Complaint ID",
]  # fmt: skip

ROWS = [
    # date, product, issue, narrative, state, disputed, id  -> kept?
    (
        "2015-03-19",
        "Credit card",
        "Billing disputes",
        "They charged me twice.",
        "CA",
        "Yes",
        "1",
    ),  # yes
    (
        "03/20/2015",
        "Mortgage",
        "Loan servicing",
        "Escrow was wrong.",
        "TX",
        "No",
        "2",
    ),  # yes (old date format)
    ("2014-12-31", "Mortgage", "Loan servicing", "Too early.", "TX", "No", "3"),  # no: before 2015
    (
        "2017-04-30",
        "Bank account or service",
        "Account opening",
        "Last day.",
        "NY",
        "Yes",
        "4",
    ),  # yes
    (
        "2017-05-01",
        "Bank account or service",
        "Account opening",
        "Too late.",
        "NY",
        "No",
        "5",
    ),  # no: after Apr 2017
    ("2016-06-01", "Credit card", "APR", "", "FL", "No", "6"),  # no: empty narrative
    ("2016-06-02", "Credit card", "APR", "   ", "FL", "No", "7"),  # no: blank narrative
    ("2016-06-03", "Credit card", "APR", "Rate went up.", "FL", "N/A", "8"),  # no: dispute N/A
    ("2016-06-04", "Credit card", "APR", "Rate went up again.", "", "", "9"),  # no: dispute missing
    (
        "2016-06-05",
        "Student loan",
        "Repaying",
        "Payments lost.",
        "",
        "No",
        "10",
    ),  # yes: missing state ok
]


def _write_export(path: Path, zipped: bool) -> Path:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    for date, product, issue, narrative, state, disputed, cid in ROWS:
        row = dict.fromkeys(HEADER, "x")
        row.update({
            "Date received": date, "Product": product, "Issue": issue,
            "Consumer complaint narrative": narrative, "State": state,
            "Consumer disputed?": disputed, "Complaint ID": cid,
            "Company response to consumer": "Closed with explanation",
        })  # fmt: skip
        w.writerow([row[h] for h in HEADER])
    if zipped:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("complaints.csv", buf.getvalue())
    else:
        path.write_text(buf.getvalue())
    return path


@pytest.mark.parametrize("zipped", [True, False])
def test_load_and_filter(tmp_path: Path, zipped: bool) -> None:
    src = _write_export(tmp_path / ("complaints.csv.zip" if zipped else "complaints.csv"), zipped)
    df = cfpb.filter_complaints(cfpb.load_export(src))
    assert df["complaint_id"].to_list() == ["1", "2", "4", "10"]
    assert df.columns == list(cfpb.COLUMNS.values())
    assert df["disputed"].to_list() == [True, False, True, False]
    assert str(df["date_received"].dtype) == "Date"
    assert df.filter(pl.col("complaint_id") == "10")["state"].to_list() == [""]


def test_build_writes_deterministic_parquet_and_manifest(tmp_path: Path) -> None:
    src = _write_export(tmp_path / "complaints.csv.zip", True)
    out_a = cfpb.build(src, tmp_path / "a")
    out_b = cfpb.build(src, tmp_path / "b")
    assert out_a[0].read_bytes() == out_b[0].read_bytes()
    manifest = json.loads(out_a[1].read_text())
    assert manifest["n_rows"] == 4
    assert manifest["dispute_rate"] == pytest.approx(0.5)
    assert manifest["window"] == ["2015-01-01", "2017-04-30"]
    assert len(manifest["source_sha256"]) == 64
    assert manifest["dropped"] == {"outside_window": 2, "no_narrative": 2, "no_dispute_value": 2}


def test_labeler_columns_exclude_outcome() -> None:
    cfpb.assert_labeler_columns(cfpb.LABELER_FEATURES)
    with pytest.raises(ValueError, match="disputed"):
        cfpb.assert_labeler_columns([*cfpb.LABELER_FEATURES, "disputed"])
    assert "Company response to consumer" not in cfpb.COLUMNS


def test_scores_encoding() -> None:
    np.testing.assert_array_equal(cfpb.dispute_scores(np.array([True, False, True])), [0, 7, 0])


def test_tabular_design(tmp_path: Path) -> None:
    src = _write_export(tmp_path / "c.csv", False)
    df = cfpb.filter_complaints(cfpb.load_export(src))
    x, names = cfpb.tabular_design(df)
    assert x.shape == (4, len(names))
    assert {"product=Credit card", "issue=Loan servicing", "state=CA", "state="} <= set(names)
    np.testing.assert_array_equal(x.sum(axis=1), [3, 3, 3, 3])  # one level per field


def test_download_skips_existing(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "complaints.csv.zip"
    dest.parent.mkdir()
    dest.write_bytes(b"already here")

    def boom(url: str) -> bytes:  # pragma: no cover - must not be called
        raise AssertionError("downloaded again")

    assert cfpb.download("https://example.invalid/x.zip", dest, fetch=boom) == dest


def test_download_writes_atomically(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "complaints.csv.zip"
    cfpb.download("https://example.invalid/x.zip", dest, fetch=lambda url: b"payload")
    assert dest.read_bytes() == b"payload"
    assert not list(dest.parent.glob("*.part"))


# --- Hugging Face mirror of the older export (the current official export no longer
# carries the narrative or "Consumer disputed?" columns) ---------------------------

CURRENT_HEADER = [h for h in HEADER if h not in (
    "Consumer complaint narrative", "Consumer consent provided?", "Consumer disputed?",
)]  # fmt: skip


def _write_mirror(directory: Path) -> Path:
    """The fixture rows as two Parquet shards, with integer complaint ids as in the mirror."""
    directory.mkdir(parents=True)
    rows = []
    for date, product, issue, narrative, state, disputed, cid in ROWS:
        row: dict[str, object] = dict.fromkeys(HEADER, "x")
        row.update({
            "Date received": date, "Product": product, "Issue": issue,
            "Consumer complaint narrative": narrative or None, "State": state or None,
            "Consumer disputed?": disputed or None, "Complaint ID": int(cid),
        })  # fmt: skip
        rows.append(row)
    frame = pl.DataFrame(
        rows, schema={h: (pl.Int64 if h == "Complaint ID" else pl.String) for h in HEADER}
    )
    frame[:5].write_parquet(directory / "train-00000-of-00002.parquet")
    frame[5:].write_parquet(directory / "train-00001-of-00002.parquet")
    return directory


def _write_current_export(path: Path, edits: dict[str, dict[str, str]], drop: set[str]) -> Path:
    """The fixture rows in today's export format (no narrative or dispute columns)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CURRENT_HEADER)
    for date, product, issue, _narrative, state, _disputed, cid in ROWS:
        if cid in drop:
            continue
        iso = date if "-" in date else f"{date[6:]}-{date[:2]}-{date[3:5]}"
        row = dict.fromkeys(CURRENT_HEADER, "x")
        row.update({"Date received": iso, "Product": product, "Issue": issue, "State": state,
                    "Complaint ID": cid, **edits.get(cid, {})})  # fmt: skip
        w.writerow([row[h] for h in CURRENT_HEADER])
    path.write_text(buf.getvalue())
    return path


def test_load_mirror_parquet_matches_csv(tmp_path: Path) -> None:
    csv_df = cfpb.filter_complaints(cfpb.load_export(_write_export(tmp_path / "c.csv", False)))
    mirror = _write_mirror(tmp_path / "mirror")
    from_dir = cfpb.filter_complaints(cfpb.load_export(mirror))
    from_file = cfpb.filter_complaints(cfpb.load_export(mirror / "train-00000-of-00002.parquet"))
    assert from_dir.equals(csv_df)
    assert from_file["complaint_id"].to_list() == ["1", "2", "4"]


def test_download_verifies_sha256(tmp_path: Path) -> None:
    dest = tmp_path / "raw" / "shard.parquet"
    good = hashlib.sha256(b"payload").hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        cfpb.download("https://example.invalid/x", dest, fetch=lambda url: b"tampered", sha256=good)
    assert not dest.exists() and not list(dest.parent.glob("*.part"))
    cfpb.download("https://example.invalid/x", dest, fetch=lambda url: b"payload", sha256=good)
    assert dest.read_bytes() == b"payload"
    dest.write_bytes(b"corrupted later")
    with pytest.raises(ValueError, match="SHA-256"):
        cfpb.download("https://example.invalid/x", dest, fetch=lambda url: b"payload", sha256=good)


def test_mirror_files_are_pinned() -> None:
    assert len(cfpb.MIRROR_FILES) == 3
    for url, sha in cfpb.MIRROR_FILES:
        assert url.startswith("https://huggingface.co/datasets/BEE-spoke-data/")
        assert len(sha) == 64 and int(sha, 16) >= 0


def test_crosscheck_against_current_export(tmp_path: Path) -> None:
    kept = cfpb.filter_complaints(cfpb.load_export(_write_mirror(tmp_path / "mirror")))
    official = _write_current_export(
        tmp_path / "official.csv", edits={"2": {"Product": "Credit card"}}, drop={"4"}
    )
    check = cfpb.crosscheck(kept, official)
    assert check == {
        "n_checked": 4,
        "n_found": 3,
        "n_identical": 2,
        "mismatches": {"date_received": 0, "product": 1, "issue": 0, "state": 0},
        "official_source": "official.csv",
    }


def test_build_records_mirror_sources_and_crosscheck(tmp_path: Path) -> None:
    mirror = _write_mirror(tmp_path / "mirror")
    official = _write_current_export(tmp_path / "official.csv", edits={}, drop=set())
    parquet, manifest_path = cfpb.build(mirror, tmp_path / "out", crosscheck_export=official)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_rows"] == 4
    assert sorted(manifest["source_files"]) == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    assert manifest["crosscheck"]["n_identical"] == 4
    assert cfpb.build(mirror, tmp_path / "again")[0].read_bytes() == parquet.read_bytes()
