"""Tests for the core Estimate and Design types (interfaces fixed by CLAUDE.md section 5)."""

import numpy as np

from silent97.types import Design, Estimate


def test_estimate_fields() -> None:
    assert Estimate._fields == (
        "point",
        "lower",
        "upper",
        "alpha",
        "se",
        "n_labeled",
        "n_unlabeled",
        "method",
        "diagnostics",
    )
    est = Estimate(20.0, 15.0, 25.0, 0.1, 3.0, 100, 1000, "classical", {"effective_n": 100})
    assert est.point == 20.0
    assert est.diagnostics["effective_n"] == 100
    assert est._replace(point=21.0).point == 21.0


def test_design_fields() -> None:
    assert Design._fields == ("idx", "pi", "name")
    d = Design(np.array([0, 2]), np.full(4, 0.5), "uniform")
    assert d.name == "uniform"
    assert d.pi.shape == (4,)
