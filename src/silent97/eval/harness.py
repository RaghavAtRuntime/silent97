"""Experiment harness: run (estimator x scenario x n x replicate) grids.

Per scenario (built once):

1. Simulate the population from ``population`` overrides and the scenario's signal
   strength, seeded with ``default_rng(seed)``. Populations therefore match
   ``python -m silent97.data.simulate --seed <seed>`` with the same parameters.
2. Draw the labeler's training set: ``labeler.n_train`` customers with probability
   proportional to the ``labeler.train_mechanism`` propensity (uniform by default;
   ``"scenario"`` uses the scenario's own mechanism, as if the model were trained on
   an earlier self-selected survey wave),
   using ``default_rng([seed, 1])``. Fit the logistic labeler and predict everyone.
3. The inference pool is everyone *not* in the training set; the truth is the
   pool's NPS. (Other datasets, such as CFPB, can build a :class:`PreparedScenario`
   whose pool is any subset; the unlabeled set is always the pool minus the draw.)

Per replicate, with ``default_rng([seed, 2, scenario_index, n, replicate])``:

4. Draw n surveyed customers from the pool with probability proportional to the
   scenario's response propensity (uniform under MCAR).
5. Assert the draw is disjoint from the training set (CLAUDE.md section 6, rule 4).
6. The unlabeled set is the rest of the pool. Every estimator sees the same draw.
   ``ppi_crossfit`` ignores the trained labeler and cross-fits a fresh model of the
   same family on the surveyed customers only.

Seeds depend only on (scenario, n, replicate), so results do not depend on
``n_jobs``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from silent97 import estimators
from silent97.data import schema
from silent97.data.simulate import SimConfig, simulate_population
from silent97.estimators.ppi import cross_fit, ppi_crossfit
from silent97.labelers.logistic import design_matrix, fit_logistic, predict_folds
from silent97.labelers.tabular import TabularLabeler

__all__ = [
    "RESULT_COLUMNS",
    "GridConfig",
    "LabelerConfig",
    "PreparedScenario",
    "Scenario",
    "check_disjoint",
    "draw_respondents",
    "load_grid_config",
    "model_correlation",
    "prepare_scenario",
    "run_grid",
    "run_replicate",
    "write_results",
]

RESULT_COLUMNS = (
    "experiment",
    "scenario",
    "mechanism",
    "signal_strength",
    "estimator",
    "n_labeled",
    "n_unlabeled",
    "replicate",
    "alpha",
    "truth",
    "point",
    "lower",
    "upper",
    "se",
    "width",
    "covered",
    "lambda_hat",
    "effective_n",
    "model_corr",
)


def draw_respondents(
    pool: ArrayLike, weights: ArrayLike, n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Draw ``n`` distinct customers from ``pool`` with probability proportional to ``weights``.

    Uses Efraimidis-Spirakis weighted sampling without replacement (keys
    ``log(U) / w``; keep the ``n`` largest). This is successive sampling: each draw
    picks a remaining unit with probability proportional to its weight. With equal
    weights it is simple random sampling without replacement. With n much smaller
    than the pool, unit i's inclusion probability is close to ``n * w_i / sum(w)``.

    Parameters
    ----------
    pool : array_like of int, shape (M,)
        Candidate customer indices.
    weights : array_like of float, shape (M,)
        Strictly positive, finite weights (e.g. response propensities).
    n : int
        Number to draw, 1 <= n <= M.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int64, shape (n,), sorted ascending.
    """
    pool_arr = np.asarray(pool, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != pool_arr.shape:
        raise ValueError(f"weights length {w.shape} must match pool length {pool_arr.shape}")
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if n > pool_arr.size:
        raise ValueError(f"n = {n} exceeds the pool size {pool_arr.size}")
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    u = 1.0 - rng.random(pool_arr.size)  # in (0, 1], so log(u) is finite
    keys = np.log(u) / w
    chosen = np.argpartition(-keys, n - 1)[:n]
    return np.sort(pool_arr[chosen])


def check_disjoint(train_idx: ArrayLike, labeled_idx: ArrayLike) -> None:
    """Raise ``ValueError`` if any labeled inference index was used to train the labeler."""
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(labeled_idx))
    if overlap.size:
        raise ValueError(
            f"training leak: {overlap.size} labeled customer(s) were in the labeler's "
            f"training set (e.g. {overlap[:5].tolist()}); only ppi_crossfit may reuse "
            "labeled data, and it does so by cross-fitting"
        )


# --------------------------------------------------------------------------- config


def _reject_unknown(values: Mapping[str, Any], known: set[str], where: str) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {unknown}; expected {sorted(known)}")


@dataclass(frozen=True)
class Scenario:
    """A response mechanism plus an optional labeler signal strength override."""

    name: str
    mechanism: str
    signal_strength: float | None = None
    labeler_features: str = "all"
    """``"all"`` labeler features, or ``"signal_only"`` (no covariates; lets E2 reach rho = 0)."""

    def __post_init__(self) -> None:
        if self.labeler_features not in ("all", "signal_only"):
            raise ValueError(
                f"scenario {self.name!r}: labeler_features must be 'all' or 'signal_only', "
                f"got {self.labeler_features!r}"
            )
        if self.mechanism not in schema.MECHANISMS:
            raise ValueError(
                f"scenario {self.name!r}: mechanism must be one of {schema.MECHANISMS}, "
                f"got {self.mechanism!r}"
            )


@dataclass(frozen=True)
class LabelerConfig:
    """Training split and model of the labeler.

    ``kind`` is ``"logistic"`` (multinomial logistic regression with penalty ``c``)
    or ``"tabular"`` (LightGBM + isotonic calibration). Cross-fitting
    (``ppi_crossfit``) always refits the logistic model, which is fast enough to fit
    K times per replicate.
    """

    n_train: int = 5_000
    c: float = 1.0
    train_mechanism: str = "mcar"
    """``"mcar"``, ``"mar"``, ``"mnar"``, or ``"scenario"`` for the scenario's mechanism."""
    kind: str = "logistic"

    def __post_init__(self) -> None:
        if self.kind not in ("logistic", "tabular"):
            raise ValueError(f"labeler.kind must be 'logistic' or 'tabular', got {self.kind!r}")
        if self.n_train < 50:
            raise ValueError(f"labeler.n_train must be >= 50, got {self.n_train}")
        allowed = (*schema.MECHANISMS, "scenario")
        if self.train_mechanism not in allowed:
            raise ValueError(
                f"labeler.train_mechanism must be one of {allowed}, got {self.train_mechanism!r}"
            )


@dataclass(frozen=True)
class GridConfig:
    """An experiment grid. See ``configs/e01_coverage.yaml`` for an example."""

    name: str
    seed: int
    scenarios: tuple[Scenario, ...]
    estimators: tuple[str, ...]
    n_labeled: tuple[int, ...]
    replicates: int
    alpha: float = 0.1
    population: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    labeler: LabelerConfig = LabelerConfig()
    crossfit_folds: int = 10
    n_jobs: int = 1

    def __post_init__(self) -> None:
        unknown = sorted(set(self.estimators) - set(estimators.REGISTRY))
        if unknown:
            raise ValueError(
                f"unknown estimator(s) {unknown}; known: {sorted(estimators.REGISTRY)}"
            )
        names = [s.name for s in self.scenarios]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"scenario names must be non-empty and unique, got {names}")
        if not self.n_labeled or min(self.n_labeled) < 2:
            raise ValueError(f"n_labeled values must be >= 2, got {self.n_labeled}")
        if self.replicates < 1:
            raise ValueError(f"replicates must be >= 1, got {self.replicates}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if self.crossfit_folds < 2:
            raise ValueError(f"crossfit_folds must be >= 2, got {self.crossfit_folds}")
        SimConfig.from_dict(dict(self.population))  # validate overrides early

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GridConfig:
        """Build from a plain mapping (e.g. parsed YAML); unknown keys raise."""
        known = {f.name for f in dataclasses.fields(cls)}
        _reject_unknown(values, known, "grid config")
        kw = dict(values)
        scenarios = []
        for raw in kw.get("scenarios", []):
            _reject_unknown(raw, {f.name for f in dataclasses.fields(Scenario)}, "scenario")
            scenarios.append(Scenario(**raw))
        kw["scenarios"] = tuple(scenarios)
        labeler = kw.get("labeler", {}) or {}
        _reject_unknown(labeler, {f.name for f in dataclasses.fields(LabelerConfig)}, "labeler")
        kw["labeler"] = LabelerConfig(**labeler)
        kw["estimators"] = tuple(kw.get("estimators", ()))
        kw["n_labeled"] = tuple(int(n) for n in kw.get("n_labeled", ()))
        kw["population"] = dict(kw.get("population", {}) or {})
        return cls(**kw)


def load_grid_config(path: str | Path) -> GridConfig:
    """Read a grid config from YAML."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return GridConfig.from_dict(raw)


# --------------------------------------------------------------------------- running


@dataclass(frozen=True)
class PreparedScenario:
    """Everything a replicate needs, computed once per scenario."""

    scenario: Scenario
    signal_strength: float
    x: NDArray[np.float64]
    scores: NDArray[np.int64]
    p_all: NDArray[np.float64]
    propensity: NDArray[np.float64]
    train_idx: NDArray[np.int64]
    pool: NDArray[np.int64]
    truth: float
    model_corr: float = float("nan")
    """Correlation over the pool between predicted and true per-customer NPS values."""


def prepare_scenario(cfg: GridConfig, scenario_index: int) -> PreparedScenario:
    """Simulate the population, train the labeler and fix the inference pool."""
    scenario = cfg.scenarios[scenario_index]
    overrides = dict(cfg.population)
    if scenario.signal_strength is not None:
        overrides["signal_strength"] = scenario.signal_strength
    sim_cfg = SimConfig.from_dict(overrides)
    table = simulate_population(sim_cfg, np.random.default_rng(cfg.seed)).table
    n_total = table.num_rows

    x, names = design_matrix(table)
    if scenario.labeler_features == "signal_only":
        x = x[:, [names.index("signal")]]
    scores = np.asarray(table.column("score").to_numpy(), dtype=np.int64)
    train_mechanism = cfg.labeler.train_mechanism
    if train_mechanism == "scenario":
        train_mechanism = scenario.mechanism
    train_weights = table.column(f"propensity_{train_mechanism}").to_numpy()
    everyone = np.arange(n_total, dtype=np.int64)
    train_idx = draw_respondents(
        everyone, train_weights, cfg.labeler.n_train, np.random.default_rng([cfg.seed, 1])
    )
    if cfg.labeler.kind == "tabular":
        tabular = TabularLabeler(random_state=cfg.seed % (2**31 - 1))
        p_all = tabular.fit(x[train_idx], scores[train_idx]).predict(x)
    else:
        p_all = fit_logistic(x[train_idx], scores[train_idx], c=cfg.labeler.c)(x)

    pool = np.setdiff1d(everyone, train_idx)
    value = 100.0 * ((scores >= 9).astype(np.float64) - (scores <= 6).astype(np.float64))
    return PreparedScenario(
        scenario=scenario,
        signal_strength=sim_cfg.signal_strength,
        x=x,
        scores=scores,
        p_all=p_all,
        propensity=np.asarray(table.column(f"propensity_{scenario.mechanism}").to_numpy()),
        train_idx=train_idx,
        pool=pool,
        truth=float(value[pool].mean()),
        model_corr=model_correlation(p_all[pool], scores[pool]),
    )


def model_correlation(p: NDArray[np.float64], scores: NDArray[np.int64]) -> float:
    """corr(h, v) with h = 100 (P(promoter) - P(detractor)) and v the true NPS value."""
    h = p @ np.array([100.0, -100.0])
    v = 100.0 * ((scores >= 9).astype(np.float64) - (scores <= 6).astype(np.float64))
    if h.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(h, v)[0, 1])


def run_replicate(
    prep: PreparedScenario, cfg: GridConfig, scenario_index: int, n: int, replicate: int
) -> list[dict[str, Any]]:
    """Run every configured estimator on one survey draw; one result row each."""
    rng = np.random.default_rng([cfg.seed, 2, scenario_index, n, replicate])
    lab = draw_respondents(prep.pool, prep.propensity[prep.pool], n, rng)
    check_disjoint(prep.train_idx, lab)
    unlab_mask = np.zeros(prep.scores.size, dtype=bool)  # unlabeled = pool minus this draw
    unlab_mask[prep.pool] = True
    unlab_mask[lab] = False

    y_lab = prep.scores[lab]
    p_lab, p_unlab = prep.p_all[lab], prep.p_all[unlab_mask]
    rows = []
    for name in cfg.estimators:
        if name == "ppi_crossfit":
            cf = cross_fit(
                prep.x[lab],
                y_lab,
                None,
                fit=functools.partial(fit_logistic, c=cfg.labeler.c),
                n_folds=min(cfg.crossfit_folds, n),
                rng=rng,
            )
            folds_unlab = predict_folds(cf.predictors, prep.x[unlab_mask])
            est = ppi_crossfit(y_lab, cf.p_lab, folds_unlab, alpha=cfg.alpha)
        else:
            est = estimators.REGISTRY[name](y_lab, p_lab, p_unlab, alpha=cfg.alpha)
        rows.append(
            {
                "experiment": cfg.name,
                "scenario": prep.scenario.name,
                "mechanism": prep.scenario.mechanism,
                "signal_strength": prep.signal_strength,
                "estimator": name,
                "n_labeled": est.n_labeled,
                "n_unlabeled": est.n_unlabeled,
                "replicate": replicate,
                "alpha": est.alpha,
                "truth": prep.truth,
                "point": est.point,
                "lower": est.lower,
                "upper": est.upper,
                "se": est.se,
                "width": est.upper - est.lower,
                "covered": bool(est.lower <= prep.truth <= est.upper),
                "lambda_hat": float(est.diagnostics.get("lambda_hat", math.nan)),
                "effective_n": float(est.diagnostics.get("effective_n", math.nan)),
                "model_corr": prep.model_corr,
            }
        )
    return rows


def _run_chunk(
    prep: PreparedScenario, cfg: GridConfig, scenario_index: int, n: int, reps: Sequence[int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in reps:
        out.extend(run_replicate(prep, cfg, scenario_index, n, r))
    return out


def run_grid(
    cfg: GridConfig,
    *,
    chunk_size: int = 25,
    prepare: Callable[[GridConfig, int], PreparedScenario] | None = None,
) -> pl.DataFrame:
    """Run the full grid and return one row per (scenario, n, replicate, estimator).

    ``prepare`` builds each scenario (default :func:`prepare_scenario`, the simulator).
    Other datasets pass their own, e.g. E6 for CFPB.
    """
    build = prepare_scenario if prepare is None else prepare
    rows: list[dict[str, Any]] = []
    for s_idx in range(len(cfg.scenarios)):
        prep = build(cfg, s_idx)
        tasks = [
            (n, range(start, min(start + chunk_size, cfg.replicates)))
            for n in cfg.n_labeled
            for start in range(0, cfg.replicates, chunk_size)
        ]
        chunks = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_run_chunk)(prep, cfg, s_idx, n, reps) for n, reps in tasks
        )
        for chunk in chunks:
            rows.extend(chunk)
    df = pl.DataFrame(rows).select(RESULT_COLUMNS)
    return df.sort(["scenario", "n_labeled", "replicate", "estimator"])


def write_results(df: pl.DataFrame, path: str | Path) -> Path:
    """Write results to Parquet, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out
