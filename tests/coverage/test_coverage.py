"""Simulation-based coverage tests (slow; run with ``make test-slow``).

CLAUDE.md section 7: under MCAR, 1,000 replicates, nominal 90%, each valid
estimator's empirical coverage must have a Wilson 95% interval overlapping
[0.88, 0.92]. Naive baselines must be shown to fail under MAR and MNAR.

These use a different seed from E1, so the gate experiment is not the only
evidence.
"""

import polars as pl
import pytest

from silent97.estimators import VALID_UNDER_MCAR
from silent97.eval import harness, metrics

pytestmark = pytest.mark.slow

BASE = {
    "seed": 20261012,
    "population": {"n_customers": 200_000},
    "labeler": {"n_train": 5000},
    "alpha": 0.1,
    "crossfit_folds": 10,
    "n_jobs": 4,
}


@pytest.fixture(scope="module")
def mcar_summary() -> pl.DataFrame:
    cfg = harness.GridConfig.from_dict(
        {
            **BASE,
            "name": "coverage_mcar",
            "scenarios": [{"name": "mcar", "mechanism": "mcar", "signal_strength": 0.6}],
            "estimators": sorted(VALID_UNDER_MCAR | {"naive_model"}),
            "n_labeled": [50, 200, 1000],
            "replicates": 1000,
        }
    )
    return metrics.summarize(harness.run_grid(cfg))


@pytest.mark.parametrize("estimator", sorted(VALID_UNDER_MCAR))
@pytest.mark.parametrize("n", [50, 200, 1000])
def test_valid_estimators_cover_under_mcar(
    mcar_summary: pl.DataFrame, estimator: str, n: int
) -> None:
    row = mcar_summary.filter((pl.col("estimator") == estimator) & (pl.col("n_labeled") == n)).row(
        0, named=True
    )
    assert row["replicates"] == 1000
    assert row["coverage_ok"], (
        f"{estimator} at n={n}: coverage {row['coverage']:.3f} "
        f"[{row['coverage_lo']:.3f}, {row['coverage_hi']:.3f}] misses [0.88, 0.92]"
    )


def test_naive_model_fails_under_mcar(mcar_summary: pl.DataFrame) -> None:
    rows = mcar_summary.filter(pl.col("estimator") == "naive_model")
    assert (rows["coverage_hi"] < 0.88).all(), rows


@pytest.fixture(scope="module")
def selected_summary() -> pl.DataFrame:
    """Self-selected respondents on the default population (configs/simulate.yaml seed)."""
    cfg = harness.GridConfig.from_dict(
        {
            **BASE,
            "seed": 20261010,
            "name": "coverage_selected",
            "scenarios": [
                {"name": "mar", "mechanism": "mar", "signal_strength": 0.6},
                {"name": "mnar", "mechanism": "mnar", "signal_strength": 0.6},
            ],
            "estimators": ["naive_respondent", "naive_model"],
            "n_labeled": [1000],
            "replicates": 1000,
        }
    )
    return metrics.summarize(harness.run_grid(cfg))


@pytest.mark.parametrize("scenario", ["mar", "mnar"])
@pytest.mark.parametrize("estimator", ["naive_respondent", "naive_model"])
def test_naive_baselines_fail_under_self_selection(
    selected_summary: pl.DataFrame, scenario: str, estimator: str
) -> None:
    row = selected_summary.filter(
        (pl.col("scenario") == scenario) & (pl.col("estimator") == estimator)
    ).row(0, named=True)
    assert row["coverage_hi"] < 0.88, row
