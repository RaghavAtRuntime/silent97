"""Tests for the population column schema and the ground-truth guard."""

import pytest

from silent97.data import schema


def test_groups_are_disjoint_and_complete() -> None:
    groups = [
        schema.ID_COLUMNS,
        schema.COVARIATES,
        schema.SEGMENT_COLUMNS,
        schema.SIGNAL_COLUMNS,
        schema.GROUND_TRUTH,
        schema.RESPONSE_COLUMNS,
    ]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat))
    assert tuple(flat) == schema.ALL_COLUMNS == tuple(schema.SCHEMA.names)


def test_labeler_features_exclude_outcomes() -> None:
    schema.assert_no_ground_truth(schema.LABELER_FEATURES)
    assert not set(schema.LABELER_FEATURES) & set(schema.GROUND_TRUTH)


@pytest.mark.parametrize("leak", ["score", "promoter", "latent", "responded_mar"])
def test_guard_rejects_leaks(leak: str) -> None:
    with pytest.raises(ValueError, match=leak):
        schema.assert_no_ground_truth([*schema.COVARIATES, leak])
