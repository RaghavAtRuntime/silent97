"""Download the raw data (Project Hammer + Statistics Canada) into data/raw/.

Writes data/raw/MANIFEST.csv with URL, download time, size and SHA-256 for each
file so a later run can confirm it used the same vintage. Project Hammer is
updated daily, so re-running this will extend the panel to the latest date.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from config import RAW, SOURCES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "grocery-oligopoly-research/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def main() -> None:
    rows = []
    for key, src in SOURCES.items():
        print(f"downloading {key}: {src['url']}")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "download.zip"
            fetch(src["url"], zpath)
            with zipfile.ZipFile(zpath) as z:
                for member, target in src["files"].items():
                    out = RAW / target
                    if out.is_symlink():
                        out.unlink()
                    with z.open(member) as fin, out.open("wb") as fout:
                        shutil.copyfileobj(fin, fout, length=1 << 20)
                    rows.append({
                        "source": key, "url": src["url"], "file": target,
                        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "bytes": out.stat().st_size, "sha256": sha256(out),
                    })
                    print(f"  {target}: {out.stat().st_size / 1e6:,.1f} MB")
    with (RAW / "MANIFEST.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
