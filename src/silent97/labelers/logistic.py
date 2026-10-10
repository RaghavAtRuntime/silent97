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

from collections.abc import Callable, Sequence
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

__all__ = [
    "LinearSoftmaxPredictor",
    "Predictor",
    "design_matrix",
    "fit_logistic",
    "predict_folds",
    "score_classes",
]

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
    scaler, lr = model[0], model[-1]
    fitted_classes = np.asarray(lr.classes_, dtype=np.int64)
    # Fold the standardization into the linear predictor: (x - mu)/s @ B + b0
    # = x @ (B / s) + (b0 - (mu / s) @ B). Same model as ``model.predict_proba``,
    # without scikit-learn's per-call validation overhead on large unlabeled sets.
    coef = np.asarray(lr.coef_, dtype=np.float64).T  # (p, k), or (p, 1) for 2 classes
    intercept = np.asarray(lr.intercept_, dtype=np.float64)
    weights = coef / scaler.scale_[:, None]
    offset = intercept - (scaler.mean_ / scaler.scale_) @ coef
    if fitted_classes.size == 2:  # binary: a single logit for the second class
        weights = np.column_stack([-weights[:, 0] / 2, weights[:, 0] / 2])
        offset = np.array([-offset[0] / 2, offset[0] / 2])
    # Expand to all three classes; an absent class gets logit -inf (probability 0).
    weights3 = np.zeros((x_arr.shape[1], 3))
    offset3 = np.full(3, -np.inf)
    weights3[:, fitted_classes] = weights
    offset3[fitted_classes] = offset
    return LinearSoftmaxPredictor(weights3, offset3, model)


class LinearSoftmaxPredictor:
    """Multinomial-logit predictor ``softmax(x @ weights + offset)`` over 3 classes.

    Calling it returns ``[P(promoter), P(detractor)]``. ``model`` keeps the fitted
    scikit-learn pipeline for inspection.
    """

    def __init__(
        self, weights: NDArray[np.float64], offset: NDArray[np.float64], model: Any
    ) -> None:
        self.weights = weights
        self.offset = offset
        self.model = model

    def __call__(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        return predict_folds([self], x_new)[:, 0, :]


def predict_folds(predictors: Sequence[Predictor], x: ArrayLike) -> NDArray[np.float64]:
    """Predictions of several predictors on the same rows, shape (m, K, 2).

    When every predictor is a :class:`LinearSoftmaxPredictor`, all K models are
    evaluated with one stacked matrix product, which is much faster for large m.
    Otherwise each predictor is called in turn.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if not all(isinstance(p, LinearSoftmaxPredictor) for p in predictors):
        return np.stack([p(x_arr) for p in predictors], axis=1)
    k = len(predictors)
    weights = np.concatenate([p.weights for p in predictors], axis=1)  # type: ignore[attr-defined]
    offset = np.concatenate([p.offset for p in predictors])  # type: ignore[attr-defined]
    logits = (x_arr @ weights + offset).reshape(x_arr.shape[0], k, 3)
    # Softmax over the 3 classes with elementwise ops on the class slices; numpy
    # reductions along a length-3 axis are several times slower on large arrays.
    det, pas, pro = logits[..., DETRACTOR], logits[..., PASSIVE], logits[..., PROMOTER]
    top = np.maximum(np.maximum(det, pas), pro)
    e_det, e_pas, e_pro = np.exp(det - top), np.exp(pas - top), np.exp(pro - top)
    total = e_det + e_pas + e_pro
    out = np.empty((x_arr.shape[0], k, 2))
    np.divide(e_pro, total, out=out[..., 0])
    np.divide(e_det, total, out=out[..., 1])
    return out
