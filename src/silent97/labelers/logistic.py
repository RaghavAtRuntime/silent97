"""Minimal labeler: multinomial logistic regression on the simulator's labeler features.

Phase 2 needs predictions before the Phase 3 labelers exist. This model predicts the
three NPS classes (detractor 0..6, passive 7..8, promoter 9..10) from observed
covariates and the ``signal`` feature, and returns ``[P(promoter), P(detractor)]``.

Multinomial logistic regression fitted by maximum likelihood (with an intercept) is
calibrated in the large on its training data: mean predicted class probabilities
equal the training class shares. PPI does not need calibrated predictions to be
valid, only to be narrow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silent97.data import schema
from silent97.data.simulate import AGE_BANDS, PROVINCES
from silent97.estimators.nps import DETRACTOR_MAX, PROMOTER_MIN, validate_scores

__all__ = ["Predictor", "design_matrix", "fit_logistic", "score_classes"]

DETRACTOR: Final = 0
PASSIVE: Final = 1
PROMOTER: Final = 2

Predictor = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Maps a feature matrix (m, p) to predictions (m, 2) = [P(promoter), P(detractor)]."""

_FEATURE_COLUMNS: Final = (
    "age_band",
    "province",
    "tenure_years",
    "has_credit_card",
    "has_chequing",
    "has_wealth",
    "has_insurance",
    "digital_engagement",
    "complaints_past_year",
    "monthly_txn_count",
    "signal",
)


def score_classes(y: ArrayLike) -> NDArray[np.int64]:
    """Map 0..10 scores to classes 0 = detractor, 1 = passive, 2 = promoter."""
    scores = validate_scores(y)
    return np.where(
        scores >= PROMOTER_MIN, PROMOTER, np.where(scores <= DETRACTOR_MAX, DETRACTOR, PASSIVE)
    ).astype(np.int64)


def design_matrix(table: pa.Table) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Feature matrix from a population table, using labeler features only.

    Age band and province are one-hot encoded (all levels; the L2 penalty handles the
    collinearity). Tenure and transaction counts enter as ``log1p``.

    Returns
    -------
    x : ndarray, shape (N, p)
    names : tuple of str, length p
    """
    schema.assert_no_ground_truth(_FEATURE_COLUMNS)
    cols: list[NDArray[np.float64]] = []
    names: list[str] = []

    def column(name: str) -> NDArray[Any]:
        return np.asarray(table.column(name).to_numpy(zero_copy_only=False))

    for name, levels in (("age_band", AGE_BANDS), ("province", PROVINCES)):
        values = column(name).astype(str)
        for level in levels:
            cols.append((values == level).astype(np.float64))
            names.append(f"{name}={level}")
    cols.append(np.log1p(column("tenure_years").astype(np.float64)))
    names.append("log1p_tenure_years")
    for name in ("has_credit_card", "has_chequing", "has_wealth", "has_insurance"):
        cols.append(column(name).astype(np.float64))
        names.append(name)
    cols.append(column("digital_engagement").astype(np.float64))
    names.append("digital_engagement")
    cols.append(column("complaints_past_year").astype(np.float64))
    names.append("complaints_past_year")
    cols.append(np.log1p(column("monthly_txn_count").astype(np.float64)))
    names.append("log1p_monthly_txn_count")
    cols.append(column("signal").astype(np.float64))
    names.append("signal")
    return np.column_stack(cols), tuple(names)


def fit_logistic(x: ArrayLike, y: ArrayLike, *, c: float = 1.0) -> Predictor:
    """Fit a standardized multinomial logistic regression and return its predictor.

    Parameters
    ----------
    x : array_like, shape (n, p)
    y : array_like, shape (n,)
        0..10 scores.
    c : float, default 1.0
        Inverse L2 penalty (scikit-learn ``C``). Fixed, never tuned on evaluation data.

    Returns
    -------
    Predictor
        Classes absent from the training data get probability 0. With a single class
        present, the prediction is that class with probability 1 for every row.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    classes = score_classes(y)
    if x_arr.ndim != 2 or x_arr.shape[0] != classes.shape[0]:
        raise ValueError(
            f"x must be 2-D with as many rows as y; got x {x_arr.shape}, y {classes.shape}"
        )
    present = np.unique(classes)
    if present.size == 1:
        constant = np.zeros(3)
        constant[present[0]] = 1.0
        out = constant[[PROMOTER, DETRACTOR]]

        def predict_constant(x_new: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.tile(out, (np.asarray(x_new).shape[0], 1))

        return predict_constant

    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    model.fit(x_arr, classes)
    fitted_classes = np.asarray(model.classes_, dtype=np.int64)

    def predict(x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        proba = model.predict_proba(np.asarray(x_new, dtype=np.float64))
        full = np.zeros((proba.shape[0], 3))
        full[:, fitted_classes] = proba
        return np.ascontiguousarray(full[:, [PROMOTER, DETRACTOR]])

    return predict
