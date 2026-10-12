"""The 'trap' figure: naive model-only intervals against the truth, beside valid ones."""

from pathlib import Path

import polars as pl
import pytest

from silent97.viz import figures


def _summary() -> pl.DataFrame:
    rows = []
    for scenario in ("embedding", "llm"):
        for est, bias, width, cov in (
            ("naive_model", 60.0, 0.7, 0.0),
            ("classical", 0.1, 9.0, 0.9),
            ("ppi_pp", 0.0, 6.0, 0.9),
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "estimator": est,
                    "n_labeled": 200,
                    "bias": bias,
                    "mean_width": width,
                    "coverage": cov,
                }
            )
    return pl.DataFrame(rows)


def test_trap_figure_writes_png(tmp_path: Path) -> None:
    out = figures.trap_figure(
        _summary(), tmp_path / "trap.png", n=200, truth=21.3, value_label="dispute rate (%)"
    )
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_trap_figure_is_reproducible(tmp_path: Path) -> None:
    a = figures.trap_figure(_summary(), tmp_path / "a.png", n=200, truth=21.3, value_label="x")
    b = figures.trap_figure(_summary(), tmp_path / "b.png", n=200, truth=21.3, value_label="x")
    assert a.read_bytes() == b.read_bytes()


def test_trap_figure_needs_the_requested_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n = 50"):
        figures.trap_figure(_summary(), tmp_path / "t.png", n=50, truth=0.0, value_label="x")
