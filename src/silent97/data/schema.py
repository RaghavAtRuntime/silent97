"""Column schema of the simulated population, grouped by role.

The groups exist so later phases can enforce, not just document, that ground-truth
columns never reach a labeler and that the response columns are only used by
sampling designs and the evaluation harness.
"""

from typing import Final

import pyarrow as pa

ID_COLUMNS: Final = ("customer_id",)

COVARIATES: Final = (
    "age_band",
    "province",
    "tenure_years",
    "tenure_band",
    "has_credit_card",
    "has_chequing",
    "has_wealth",
    "has_insurance",
    "digital_engagement",
    "complaints_past_year",
    "monthly_txn_count",
)
"""Observed customer traits. Usable by labelers, designs and MRP."""

SEGMENT_COLUMNS: Final = ("segment", "segment_id")
"""Segment = province x age band (60 cells). A function of the covariates."""

SIGNAL_COLUMNS: Final = ("signal",)
"""Noisy measurement of the latent response; the labeler-quality knob."""

LABELER_FEATURES: Final = COVARIATES + SEGMENT_COLUMNS + SIGNAL_COLUMNS
"""Everything a labeler may see."""

GROUND_TRUTH: Final = ("latent", "latent_response", "score", "promoter", "detractor")
"""Outcome columns. Never shown to a labeler; ``score`` is revealed only for surveyed
customers by a sampling design."""

MECHANISMS: Final = ("mcar", "mar", "mnar")

RESPONSE_COLUMNS: Final = tuple(f"propensity_{m}" for m in MECHANISMS) + tuple(
    f"responded_{m}" for m in MECHANISMS
)
"""Per-mechanism response propensities and one realized response flag each."""

ALL_COLUMNS: Final = (
    ID_COLUMNS + COVARIATES + SEGMENT_COLUMNS + SIGNAL_COLUMNS + GROUND_TRUTH + RESPONSE_COLUMNS
)

_CATEGORY = pa.dictionary(pa.int8(), pa.string())

SCHEMA: Final = pa.schema(
    [
        ("customer_id", pa.int64()),
        ("age_band", _CATEGORY),
        ("province", _CATEGORY),
        ("tenure_years", pa.float64()),
        ("tenure_band", _CATEGORY),
        ("has_credit_card", pa.bool_()),
        ("has_chequing", pa.bool_()),
        ("has_wealth", pa.bool_()),
        ("has_insurance", pa.bool_()),
        ("digital_engagement", pa.float64()),
        ("complaints_past_year", pa.int32()),
        ("monthly_txn_count", pa.int32()),
        ("segment", _CATEGORY),
        ("segment_id", pa.int16()),
        ("signal", pa.float64()),
        ("latent", pa.float64()),
        ("latent_response", pa.float64()),
        ("score", pa.int8()),
        ("promoter", pa.bool_()),
        ("detractor", pa.bool_()),
        ("propensity_mcar", pa.float64()),
        ("propensity_mar", pa.float64()),
        ("propensity_mnar", pa.float64()),
        ("responded_mcar", pa.bool_()),
        ("responded_mar", pa.bool_()),
        ("responded_mnar", pa.bool_()),
    ]
)

if tuple(SCHEMA.names) != ALL_COLUMNS:  # pragma: no cover - guards edits to this module
    raise RuntimeError("SCHEMA and ALL_COLUMNS are out of sync")


def assert_no_ground_truth(columns: list[str] | tuple[str, ...]) -> None:
    """Raise ``ValueError`` if any ground-truth or response column is in ``columns``.

    Call this at every boundary where a feature matrix is handed to a labeler.
    """
    forbidden = sorted(set(columns) & (set(GROUND_TRUTH) | set(RESPONSE_COLUMNS)))
    if forbidden:
        raise ValueError(f"labeler features must not include outcome columns: {forbidden}")
