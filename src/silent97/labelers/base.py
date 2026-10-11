"""Labeler interface shared by every model that predicts NPS classes.

A labeler maps whatever it consumes (a feature matrix, a list of texts, a list of
LLM items) to calibrated probabilities ``[P(promoter), P(detractor)]`` with shape
``(n, 2)``, the input every estimator expects. Classes are
0 = detractor (0..6), 1 = passive (7..8), 2 = promoter (9..10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.estimators.nps import (
    DETRACTOR_MAX,
    PROMOTER_MIN,
    validate_predictions,
    validate_scores,
)

__all__ = [
    "DETRACTOR",
    "PASSIVE",
    "PROMOTER",
    "Labeler",
    "as_fit_fn",
    "class_probs_to_pair",
    "score_classes",
]

DETRACTOR: Final = 0
PASSIVE: Final = 1
PROMOTER: Final = 2


class Labeler(Protocol):
    """A model that can be fit on scored examples and predict the class pair."""

    name: str

    def fit(self, x: Any, y: ArrayLike) -> Labeler:
        """Fit on inputs ``x`` and 0..10 scores ``y``; returns self."""
        ...

    def predict(self, x: Any) -> NDArray[np.float64]:
        """Calibrated ``[P(promoter), P(detractor)]``, shape ``(len(x), 2)``."""
        ...


def score_classes(y: ArrayLike) -> NDArray[np.int64]:
    """Map 0..10 scores to classes 0 = detractor, 1 = passive, 2 = promoter."""
    scores = validate_scores(y)
    return np.where(
        scores >= PROMOTER_MIN, PROMOTER, np.where(scores <= DETRACTOR_MAX, DETRACTOR, PASSIVE)
    ).astype(np.int64)


def class_probs_to_pair(proba: ArrayLike, classes: Sequence[int]) -> NDArray[np.float64]:
    """Turn class-probability columns into ``[P(promoter), P(detractor)]``.

    ``classes[j]`` names the class of column j. A class that is absent from
    ``classes`` (never seen in training) gets probability 0.
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(classes):
        raise ValueError(
            f"proba has {p.shape[-1] if p.ndim else 0} columns but {len(classes)} classes given"
        )
    position = {int(c): j for j, c in enumerate(classes)}
    out = np.zeros((p.shape[0], 2))
    for col, cls in enumerate((PROMOTER, DETRACTOR)):
        if cls in position:
            out[:, col] = p[:, position[cls]]
    return validate_predictions(out, name="labeler predictions")


def as_fit_fn(
    factory: Callable[[], Labeler],
) -> Callable[[Any, NDArray[np.int64]], Callable[[Any], NDArray[np.float64]]]:
    """Adapt a labeler factory to the ``fit(x, y) -> predictor`` form used by cross-fitting."""

    def fit(x: Any, y: NDArray[np.int64]) -> Callable[[Any], NDArray[np.float64]]:
        return factory().fit(x, y).predict

    return fit
