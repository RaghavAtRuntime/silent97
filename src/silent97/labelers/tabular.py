"""Tabular labeler: LightGBM multiclass with isotonic calibration.

The model predicts the three NPS classes (detractor / passive / promoter) from a
numeric feature matrix. Raw boosted-tree probabilities are often miscalibrated, so
the two probabilities the estimators use are calibrated with isotonic regression:

1. Out-of-fold multiclass probabilities are computed for every training row with
   K-fold stratified cross-validation (each row is predicted by a model that did not
   see it).
2. One isotonic map is fitted per target, on the out-of-fold P(promoter) against
   1[promoter] and on P(detractor) against 1[detractor].
3. The final LightGBM model is trained on all training rows. At prediction time
   its P(promoter) and P(detractor) pass through the two maps, and P(passive) is the
   remainder. In the rare row where the calibrated pair sums to more than 1, both are
   scaled down proportionally.

Calibrating the two used columns directly avoids one-vs-rest calibration over all
three classes followed by row renormalization, which leaves the individual columns
miscalibrated (by up to about 5 points per decile in tests). Calibration helps
width and makes ``naive_model`` less wrong; PPI's validity does not depend on it.

Hyperparameters are fixed defaults, never tuned on evaluation replicates;
``random_state`` comes from the experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from silent97.labelers.base import DETRACTOR, PROMOTER, class_probs_to_pair, score_classes

__all__ = ["TabularLabeler"]


@dataclass
class TabularLabeler:
    """LightGBM multiclass + isotonic calibration (see module docstring)."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 40
    calibration_folds: int = 5
    random_state: int = 0
    name: str = "tabular"
    calibrated_: bool = field(default=False, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _classes: list[int] = field(default_factory=list, init=False, repr=False)
    _maps: list[IsotonicRegression] = field(default_factory=list, init=False, repr=False)
    _constant: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def _new_model(self) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )

    def fit(self, x: ArrayLike, y: ArrayLike) -> TabularLabeler:
        """Fit on a feature matrix (n, p) and 0..10 scores (n,)."""
        x_arr = np.asarray(x, dtype=np.float64)
        classes = score_classes(y)
        if x_arr.ndim != 2 or x_arr.shape[0] != classes.shape[0]:
            raise ValueError(f"x must be (n, p) with n = {classes.shape[0]}, got {x_arr.shape}")
        present, counts = np.unique(classes, return_counts=True)
        self._maps = []
        if present.size == 1:
            self._constant = class_probs_to_pair(np.ones((1, 1)), [int(present[0])])[0]
            self._model, self.calibrated_ = None, False
            return self
        self._constant = None
        self._classes = [int(c) for c in present]
        folds = int(min(self.calibration_folds, counts.min()))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            oof = class_probs_to_pair(
                cross_val_predict(self._new_model(), x_arr, classes, cv=cv, method="predict_proba"),
                self._classes,
            )
            for col, cls in enumerate((PROMOTER, DETRACTOR)):
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(oof[:, col], (classes == cls).astype(np.float64))
                self._maps.append(iso)
            self.calibrated_ = True
        else:  # a class with a single member: cross-validated calibration is impossible
            self.calibrated_ = False
        self._model = self._new_model().fit(x_arr, classes)
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Calibrated ``[P(promoter), P(detractor)]`` for each row of ``x``."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self._constant is not None:
            return np.tile(self._constant, (x_arr.shape[0], 1))
        if self._model is None:
            raise RuntimeError("TabularLabeler.predict called before fit")
        pair = class_probs_to_pair(self._model.predict_proba(x_arr), self._classes)
        if not self.calibrated_:
            return pair
        out = np.column_stack([m.predict(pair[:, j]) for j, m in enumerate(self._maps)])
        total = out.sum(axis=1)
        over = total > 1.0
        out[over] /= total[over, None]
        return out
