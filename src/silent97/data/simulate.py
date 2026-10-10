r"""Synthetic customer population with known ground-truth NPS.

The simulator is the primary benchmark of this project: only here is the true NPS
known exactly, so only here can interval coverage be checked.

Generative model
----------------
For each of :math:`N` customers (default 200,000):

1. **Covariates.** Age band (6 levels) and province (10 provinces, roughly
   population-weighted) are drawn independently. Exact age is uniform within band and
   is used only to generate the other covariates; it is not written out. Tenure is
   ``(age - 18) * Beta(1.1, 2.5)`` years, so it never exceeds adult years. Chequing,
   credit card, wealth and insurance holdings are Bernoulli with logit-linear
   probabilities in age (and tenure / other holdings). Digital engagement is
   ``Beta(6m, 6(1 - m))`` with mean ``m`` decreasing in age. Monthly transactions are
   Poisson-lognormal, increasing in chequing, credit card and engagement. Complaints in
   the past year are Poisson with a rate increasing in credit card and transactions.
   The exact coefficients are in :func:`_draw_covariates`.

2. **Segments.** segment = province x age band (60 cells), with random intercepts
   :math:`u_p \sim N(0, \sigma_p^2)`, :math:`u_a \sim N(0, \sigma_a^2)`,
   :math:`u_s \sim N(0, \sigma_s^2)` for province, age band and segment.

3. **Latent satisfaction.**

   .. math::

       \eta_i = x_i^\top \beta + u_{p(i)} + u_{a(i)} + u_{s(i)} + \sigma_e e_i,
       \qquad e_i \sim N(0, 1),

   with features ``log1p(tenure)``, the four product flags, digital engagement,
   complaints and ``log1p(transactions)``. The :math:`\sigma_e e_i` term is
   satisfaction no covariate explains; it is what makes MNAR non-ignorable.

4. **Scores (cumulative logit).** With thresholds :math:`t_k = c_k + s`,
   :math:`k = 0, \dots, 9`,

   .. math::

       P(Y_i \le k \mid \eta_i) = \operatorname{logit}^{-1}(t_k - \eta_i),

   implemented as :math:`Y_i^* = \eta_i + \varepsilon_i`,
   :math:`\varepsilon_i \sim \mathrm{Logistic}(0, 1)`, and
   :math:`Y_i = \#\{k : t_k < Y_i^*\}`. The cutpoint shape :math:`c` is the logit of
   the cumulative ``base_score_probs`` (the score distribution at :math:`\eta = 0`).
   The scalar shift :math:`s` is solved by root-finding so the population's
   *expected* NPS given the drawn :math:`\eta`,

   .. math::

       \frac{100}{N} \sum_i \left[ P(Y_i \ge 9 \mid \eta_i) - P(Y_i \le 6 \mid \eta_i)
       \right],

   equals ``target_nps``. Tuning never looks at the drawn scores; the realized
   finite-population NPS then differs from the target only by Monte Carlo noise
   (about 0.2 points at N = 200,000).

5. **Response propensities.** With a covariate index

   .. math::

       m_i = \gamma_{age} \mathrm{age}_c + \gamma_{dig} (\mathrm{digital} - 0.5)
       + \gamma_{comp} \mathrm{complaints} + \gamma_{ten} \log(1 + \mathrm{tenure})
       + \gamma_{obs}\, z^{obs}_i,

   where :math:`z^{obs}` standardizes the covariate-explained part of latent
   satisfaction, :math:`\eta_i - \sigma_e e_i` (fixed effects plus segment
   intercepts). That term makes the MAR bias in the raw respondent NPS stable across
   seeds while keeping MAR a function of observed traits and segment only. With the
   standardized latent :math:`z_i = (\eta_i - \bar\eta) / \mathrm{sd}(\eta)`:

   - ``mcar``: :math:`\rho_i = r`;
   - ``mar``: :math:`\rho_i = \operatorname{logit}^{-1}(\alpha + m_i)`;
   - ``mnar``: :math:`\rho_i = \operatorname{logit}^{-1}(\alpha + m_i + \kappa z_i^2)`,
     U-shaped in latent satisfaction, so both extremes respond more;

   where :math:`\alpha` is solved per mechanism so that the mean propensity equals the
   response rate :math:`r`. One realized response flag per mechanism is drawn with
   common random numbers :math:`R_i = 1[U_i < \rho_i]`.

6. **Labeler signal knob.** ``signal = a * z(Y*) + sqrt(1 - a^2) * xi`` with
   :math:`\xi \sim N(0, 1)` and ``a = signal_strength``. ``a = 0`` is pure noise and
   ``a = 1`` reveals :math:`Y^*` (hence the score) exactly. Realized correlations are
   recorded in the manifest.

Random streams
--------------
The caller's ``Generator`` seeds five independent child streams (covariates, effects,
outcome, signal, response). Changing ``signal_strength`` changes only ``signal``;
changing a response parameter changes only the response columns. This lets
experiments compare settings on the same population.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit, logit

import silent97
from silent97.data import schema

__all__ = [
    "AGE_BANDS",
    "PROVINCES",
    "TENURE_BANDS",
    "TENURE_EDGES",
    "Population",
    "SimConfig",
    "base_cutpoints",
    "expected_nps",
    "load_config",
    "main",
    "response_propensity",
    "run",
    "simulate_population",
    "solve_cutpoint_shift",
    "write_population",
]

SIMULATOR_VERSION: Final = "1"

AGE_BANDS: Final = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
AGE_BAND_PROBS: Final = (0.10, 0.17, 0.18, 0.17, 0.17, 0.21)
AGE_BAND_BOUNDS: Final = ((18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 90))

PROVINCES: Final = ("ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE")
PROVINCE_PROBS: Final = (0.397, 0.220, 0.135, 0.117, 0.036, 0.030, 0.026, 0.021, 0.014, 0.004)

TENURE_BANDS: Final = ("<2", "2-5", "5-10", "10+")
TENURE_EDGES: Final = (2.0, 5.0, 10.0)

N_SEGMENTS: Final = len(PROVINCES) * len(AGE_BANDS)

BETA_KEYS: Final = (
    "log_tenure",
    "credit_card",
    "chequing",
    "wealth",
    "insurance",
    "digital",
    "complaints",
    "log_txn",
)
DEFAULT_BETA: Final[Mapping[str, float]] = {
    "log_tenure": 0.15,
    "credit_card": -0.05,
    "chequing": 0.10,
    "wealth": 0.30,
    "insurance": 0.10,
    "digital": 0.60,
    "complaints": -0.80,
    "log_txn": 0.10,
}

MAR_KEYS: Final = ("age", "digital", "complaints", "log_tenure", "observed_latent")
DEFAULT_MAR_COEFS: Final[Mapping[str, float]] = {
    "age": 0.40,
    "digital": 0.50,
    "complaints": 0.60,
    "log_tenure": 0.10,
    "observed_latent": 0.30,
}

# Score distribution at eta = 0 (before the shift); 27% detractors, 39% promoters.
DEFAULT_BASE_SCORE_PROBS: Final = (
    0.02, 0.01, 0.02, 0.03, 0.04, 0.07, 0.08, 0.14, 0.20, 0.17, 0.22,
)  # fmt: skip

_MIN_CUSTOMERS: Final = 1_000
_STREAMS: Final = ("covariates", "effects", "outcome", "signal", "response")

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the simulated population. See the module docstring for the model.

    Attributes
    ----------
    n_customers : int
        Population size N (at least 1,000).
    target_nps : float
        Expected population NPS the cutpoints are tuned to, in (-100, 100).
    base_score_probs : tuple of 11 floats
        Score distribution at latent satisfaction 0; sets the cutpoint shape.
    beta : mapping
        Fixed effects on latent satisfaction; keys ``BETA_KEYS``.
    sd_province, sd_age, sd_segment : float
        Random-intercept standard deviations.
    sigma_idiosyncratic : float
        SD of the customer-level latent term no covariate explains.
    signal_strength : float
        Labeler signal knob ``a`` in [0, 1].
    response_rate : float
        Mean response propensity under every mechanism, in (0, 1).
    mar_coefs : mapping
        Covariate coefficients of the response index; keys ``MAR_KEYS``.
    mnar_strength : float
        Coefficient ``kappa >= 0`` on the squared standardized latent under ``mnar``.
    """

    n_customers: int = 200_000
    target_nps: float = 20.0
    base_score_probs: tuple[float, ...] = DEFAULT_BASE_SCORE_PROBS
    beta: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_BETA))
    sd_province: float = 0.15
    sd_age: float = 0.20
    sd_segment: float = 0.25
    sigma_idiosyncratic: float = 1.0
    signal_strength: float = 0.6
    response_rate: float = 0.03
    mar_coefs: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MAR_COEFS))
    mnar_strength: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_customers, bool)
            or not isinstance(self.n_customers, int)
            or self.n_customers < _MIN_CUSTOMERS
        ):
            raise ValueError(
                f"n_customers must be an integer >= {_MIN_CUSTOMERS}, got {self.n_customers!r}"
            )
        if not -100.0 < self.target_nps < 100.0:
            raise ValueError(f"target_nps must lie in (-100, 100), got {self.target_nps}")
        probs = np.asarray(self.base_score_probs, dtype=np.float64)
        if probs.shape != (11,) or np.any(probs <= 0) or not np.isclose(probs.sum(), 1.0):
            raise ValueError(
                "base_score_probs must be 11 strictly positive probabilities summing to 1, "
                f"got {self.base_score_probs}"
            )
        object.__setattr__(self, "base_score_probs", tuple(float(p) for p in probs))
        for name in ("sd_province", "sd_age", "sd_segment", "sigma_idiosyncratic"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError(f"signal_strength must lie in [0, 1], got {self.signal_strength}")
        if not 0.0 < self.response_rate < 1.0:
            raise ValueError(f"response_rate must lie in (0, 1), got {self.response_rate}")
        if not self.mnar_strength >= 0:
            raise ValueError(f"mnar_strength must be >= 0, got {self.mnar_strength}")
        for name, keys in (("beta", BETA_KEYS), ("mar_coefs", MAR_KEYS)):
            given = dict(getattr(self, name))
            if set(given) != set(keys):
                raise ValueError(f"{name} must have exactly the keys {keys}, got {sorted(given)}")
            object.__setattr__(self, name, {k: float(given[k]) for k in keys})

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SimConfig:
        """Build a config from a plain mapping (e.g. parsed YAML); unknown keys raise."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"unknown simulation parameter(s): {unknown}; expected {sorted(known)}"
            )
        kwargs = dict(values)
        if "base_score_probs" in kwargs:
            kwargs["base_score_probs"] = tuple(kwargs["base_score_probs"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the config."""
        out = dataclasses.asdict(self)
        out["base_score_probs"] = list(self.base_score_probs)
        out["beta"] = dict(self.beta)
        out["mar_coefs"] = dict(self.mar_coefs)
        return out


def load_config(path: str | Path) -> tuple[SimConfig, int, Path]:
    """Read a simulation YAML file.

    Expected layout::

        seed: 20261010              # required integer
        out_dir: data/processed/sim # optional
        simulation:                 # optional; any SimConfig field
          n_customers: 200000

    Returns
    -------
    (config, seed, out_dir)
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    unknown = sorted(set(raw) - {"seed", "out_dir", "simulation"})
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {unknown}")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{path}: 'seed' must be a non-negative integer, got {seed!r}")
    out_dir = Path(raw.get("out_dir", "data/processed/sim"))
    config = SimConfig.from_dict(raw.get("simulation") or {})
    return config, seed, out_dir


# --------------------------------------------------------------------------- model pieces


def base_cutpoints(base_score_probs: Sequence[float]) -> FloatArray:
    """Cutpoint shape ``c_k = logit(P(Y <= k))``, k = 0..9, from an 11-point distribution."""
    probs = np.asarray(base_score_probs, dtype=np.float64)
    return np.asarray(logit(np.cumsum(probs)[:-1]), dtype=np.float64)


def expected_nps(eta: FloatArray, thresholds: FloatArray) -> float:
    """Expected population NPS given latent satisfaction and the 10 thresholds.

    ``100 * mean_i [1 - expit(t_8 - eta_i) - expit(t_6 - eta_i)]``.
    """
    p_promoter = 1.0 - expit(thresholds[8] - eta)
    p_detractor = expit(thresholds[6] - eta)
    return float(100.0 * np.mean(p_promoter - p_detractor))


def solve_cutpoint_shift(eta: FloatArray, base: FloatArray, target_nps: float) -> float:
    """Shift ``s`` such that ``expected_nps(eta, base + s) == target_nps``.

    Expected NPS is strictly decreasing in ``s``, so the root is unique.
    """

    def gap(shift: float) -> float:
        return expected_nps(eta, base + shift) - target_nps

    lo, hi = -50.0, 50.0
    if not gap(lo) > 0 > gap(hi):  # pragma: no cover - unreachable for valid targets
        raise ValueError(f"cannot reach target NPS {target_nps} by shifting cutpoints")
    return float(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500))


def _solve_intercept(index: FloatArray, rate: float) -> float:
    """Intercept ``alpha`` with ``mean(expit(alpha + index)) == rate``."""

    def gap(alpha: float) -> float:
        return float(np.mean(expit(alpha + index))) - rate

    return float(brentq(gap, -60.0, 60.0, xtol=1e-14, rtol=1e-14, maxiter=500))


def response_propensity(
    mar_index: FloatArray,
    latent_z: FloatArray,
    *,
    mechanism: str,
    response_rate: float,
    mnar_strength: float,
) -> FloatArray:
    """Response propensities under one mechanism, with mean exactly ``response_rate``.

    Parameters
    ----------
    mar_index : ndarray, shape (N,)
        Covariate part of the response logit.
    latent_z : ndarray, shape (N,)
        Standardized latent satisfaction; used only by ``mnar``.
    mechanism : {"mcar", "mar", "mnar"}
    response_rate : float
        Target mean propensity in (0, 1).
    mnar_strength : float
        Coefficient on ``latent_z ** 2`` under ``mnar``.
    """
    if mechanism == "mcar":
        return np.full(mar_index.shape, float(response_rate))
    if mechanism == "mar":
        index = mar_index
    elif mechanism == "mnar":
        index = mar_index + mnar_strength * latent_z**2
    else:
        raise ValueError(f"mechanism must be one of {schema.MECHANISMS}, got {mechanism!r}")
    alpha = _solve_intercept(index, response_rate)
    return np.asarray(expit(alpha + index), dtype=np.float64)


# --------------------------------------------------------------------------- simulation


@dataclass(frozen=True)
class Population:
    """A simulated population: the table (see ``schema.SCHEMA``) and its manifest."""

    table: pa.Table
    manifest: dict[str, Any]


def _spawn_streams(rng: np.random.Generator) -> dict[str, np.random.Generator]:
    entropy = rng.integers(0, 2**63, size=4, dtype=np.int64)
    children = np.random.SeedSequence([int(e) for e in entropy]).spawn(len(_STREAMS))
    return {name: np.random.default_rng(ss) for name, ss in zip(_STREAMS, children, strict=True)}


def _draw_covariates(g: np.random.Generator, n: int) -> dict[str, NDArray[Any]]:
    age_idx = g.choice(len(AGE_BANDS), size=n, p=AGE_BAND_PROBS)
    bounds = np.array(AGE_BAND_BOUNDS, dtype=np.float64)
    age_years = bounds[age_idx, 0] + (bounds[age_idx, 1] - bounds[age_idx, 0]) * g.random(n)
    province_idx = g.choice(len(PROVINCES), size=n, p=PROVINCE_PROBS)
    tenure = (age_years - 18.0) * g.beta(1.1, 2.5, size=n)
    age_c = (age_years - 45.0) / 15.0

    chequing = g.random(n) < expit(2.0 + 0.2 * age_c)
    credit_card = g.random(n) < expit(0.6 + 0.3 * age_c - 0.4 * age_c**2 + 0.5 * chequing)
    wealth = g.random(n) < expit(-1.6 + 0.8 * age_c + 0.05 * tenure)
    insurance = g.random(n) < expit(-1.5 + 0.4 * age_c + 0.3 * wealth)
    m = expit(0.6 - 0.7 * age_c)
    digital = g.beta(6.0 * m, 6.0 * (1.0 - m))
    log_rate_txn = (
        2.6 + 0.9 * chequing + 0.3 * credit_card + 0.8 * (digital - 0.5)
        + 0.5 * g.standard_normal(n)
    )  # fmt: skip
    txn = g.poisson(np.exp(log_rate_txn))
    complaints = g.poisson(np.exp(-3.0 + 0.3 * credit_card + 0.2 * np.log1p(txn)))

    return {
        "age_idx": age_idx,
        "age_c": age_c,
        "province_idx": province_idx,
        "tenure_years": tenure,
        "has_credit_card": credit_card,
        "has_chequing": chequing,
        "has_wealth": wealth,
        "has_insurance": insurance,
        "digital_engagement": digital,
        "complaints_past_year": complaints,
        "monthly_txn_count": txn,
    }


def _category(idx: NDArray[Any], labels: Sequence[str]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(np.asarray(idx, dtype=np.int8), type=pa.int8()),
        pa.array(list(labels), type=pa.string()),
    )


def _corr(a: FloatArray, b: FloatArray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def simulate_population(config: SimConfig, rng: np.random.Generator) -> Population:
    """Simulate a customer population with known ground truth.

    Parameters
    ----------
    config : SimConfig
    rng : numpy.random.Generator
        Source of all randomness. Five independent child streams are derived from it.

    Returns
    -------
    Population
        ``table`` follows ``schema.SCHEMA``; ``manifest`` records the config, derived
        parameters (cutpoints, random effects, intercepts) and ground-truth summaries.
    """
    n = config.n_customers
    streams = _spawn_streams(rng)

    cov = _draw_covariates(streams["covariates"], n)
    age_idx, province_idx = cov["age_idx"], cov["province_idx"]
    segment_idx = province_idx * len(AGE_BANDS) + age_idx
    tenure_idx = np.digitize(cov["tenure_years"], TENURE_EDGES)

    g_eff = streams["effects"]
    u_province = config.sd_province * g_eff.standard_normal(len(PROVINCES))
    u_age = config.sd_age * g_eff.standard_normal(len(AGE_BANDS))
    u_segment = config.sd_segment * g_eff.standard_normal(N_SEGMENTS)

    beta = config.beta
    features = {
        "log_tenure": np.log1p(cov["tenure_years"]),
        "credit_card": cov["has_credit_card"].astype(np.float64),
        "chequing": cov["has_chequing"].astype(np.float64),
        "wealth": cov["has_wealth"].astype(np.float64),
        "insurance": cov["has_insurance"].astype(np.float64),
        "digital": cov["digital_engagement"],
        "complaints": cov["complaints_past_year"].astype(np.float64),
        "log_txn": np.log1p(cov["monthly_txn_count"]),
    }
    g_out = streams["outcome"]
    idiosyncratic = g_out.standard_normal(n)
    eps = g_out.logistic(0.0, 1.0, size=n)
    latent = (
        sum(beta[k] * features[k] for k in BETA_KEYS)
        + u_province[province_idx]
        + u_age[age_idx]
        + u_segment[segment_idx]
        + config.sigma_idiosyncratic * idiosyncratic
    )
    latent = np.asarray(latent, dtype=np.float64)

    base = base_cutpoints(config.base_score_probs)
    shift = solve_cutpoint_shift(latent, base, config.target_nps)
    thresholds = base + shift
    latent_response = latent + eps
    score = np.searchsorted(thresholds, latent_response, side="left")
    promoter = score >= 9
    detractor = score <= 6

    a = config.signal_strength
    z_response = (latent_response - latent_response.mean()) / latent_response.std()
    signal = a * z_response + np.sqrt(1.0 - a * a) * streams["signal"].standard_normal(n)

    mc = config.mar_coefs
    observed_latent = latent - config.sigma_idiosyncratic * idiosyncratic
    observed_z = (observed_latent - observed_latent.mean()) / observed_latent.std()
    mar_index = (
        mc["age"] * cov["age_c"]
        + mc["digital"] * (cov["digital_engagement"] - 0.5)
        + mc["complaints"] * features["complaints"]
        + mc["log_tenure"] * features["log_tenure"]
        + mc["observed_latent"] * observed_z
    )
    latent_z = (latent - latent.mean()) / latent.std()
    u_response = streams["response"].random(n)
    propensity = {
        m: response_propensity(
            mar_index,
            latent_z,
            mechanism=m,
            response_rate=config.response_rate,
            mnar_strength=config.mnar_strength,
        )
        for m in schema.MECHANISMS
    }
    responded = {m: u_response < propensity[m] for m in schema.MECHANISMS}

    segment_labels = [f"{p}|{b}" for p in PROVINCES for b in AGE_BANDS]
    columns: dict[str, pa.Array] = {
        "customer_id": pa.array(np.arange(n, dtype=np.int64)),
        "age_band": _category(age_idx, AGE_BANDS),
        "province": _category(province_idx, PROVINCES),
        "tenure_years": pa.array(cov["tenure_years"], type=pa.float64()),
        "tenure_band": _category(tenure_idx, TENURE_BANDS),
        "has_credit_card": pa.array(cov["has_credit_card"]),
        "has_chequing": pa.array(cov["has_chequing"]),
        "has_wealth": pa.array(cov["has_wealth"]),
        "has_insurance": pa.array(cov["has_insurance"]),
        "digital_engagement": pa.array(cov["digital_engagement"], type=pa.float64()),
        "complaints_past_year": pa.array(cov["complaints_past_year"].astype(np.int32)),
        "monthly_txn_count": pa.array(cov["monthly_txn_count"].astype(np.int32)),
        "segment": _category(segment_idx, segment_labels),
        "segment_id": pa.array(segment_idx.astype(np.int16)),
        "signal": pa.array(signal, type=pa.float64()),
        "latent": pa.array(latent, type=pa.float64()),
        "latent_response": pa.array(latent_response, type=pa.float64()),
        "score": pa.array(score.astype(np.int8)),
        "promoter": pa.array(promoter),
        "detractor": pa.array(detractor),
    }
    for m in schema.MECHANISMS:
        columns[f"propensity_{m}"] = pa.array(propensity[m], type=pa.float64())
    for m in schema.MECHANISMS:
        columns[f"responded_{m}"] = pa.array(responded[m])
    table = pa.Table.from_arrays(
        [columns[name] for name in schema.ALL_COLUMNS], schema=schema.SCHEMA
    )

    value = 100.0 * (promoter.astype(np.float64) - detractor.astype(np.float64))
    seg_n = np.bincount(segment_idx, minlength=N_SEGMENTS)
    seg_prom = np.bincount(segment_idx, weights=promoter, minlength=N_SEGMENTS)
    seg_det = np.bincount(segment_idx, weights=detractor, minlength=N_SEGMENTS)
    segment_nps = {
        label: {
            "n": int(seg_n[j]),
            "promoter_share": float(seg_prom[j] / seg_n[j]) if seg_n[j] else None,
            "detractor_share": float(seg_det[j] / seg_n[j]) if seg_n[j] else None,
            "nps": float(100.0 * (seg_prom[j] - seg_det[j]) / seg_n[j]) if seg_n[j] else None,
        }
        for j, label in enumerate(segment_labels)
    }

    response_summary: dict[str, Any] = {}
    for m in schema.MECHANISMS:
        rho, flag = propensity[m], responded[m]
        response_summary[m] = {
            "mean_propensity": float(rho.mean()),
            "min_propensity": float(rho.min()),
            "max_propensity": float(rho.max()),
            "n_responded": int(flag.sum()),
            # Large-sample limit of the raw respondent NPS under this mechanism.
            "expected_respondent_nps": float(np.sum(rho * value) / np.sum(rho)),
            "realized_respondent_nps": float(value[flag].mean()) if flag.any() else None,
        }

    manifest: dict[str, Any] = {
        "simulator_version": SIMULATOR_VERSION,
        "versions": {
            "silent97": silent97.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "config": config.to_dict(),
        "derived": {
            "base_cutpoints": base.tolist(),
            "cutpoint_shift": shift,
            "thresholds": thresholds.tolist(),
            "random_effects": {
                "province": dict(zip(PROVINCES, u_province.tolist(), strict=True)),
                "age_band": dict(zip(AGE_BANDS, u_age.tolist(), strict=True)),
                "segment": dict(zip(segment_labels, u_segment.tolist(), strict=True)),
            },
            "latent_mean": float(latent.mean()),
            "latent_sd": float(latent.std()),
        },
        "truth": {
            "n_customers": n,
            "nps": float(value.mean()),
            "expected_nps": expected_nps(latent, thresholds),
            "promoter_share": float(promoter.mean()),
            "detractor_share": float(detractor.mean()),
            "passive_share": float(1.0 - promoter.mean() - detractor.mean()),
            "score_counts": np.bincount(score, minlength=11).tolist(),
            "segment_nps": segment_nps,
        },
        "response": response_summary,
        "signal": {
            "strength": a,
            "corr_score": _corr(signal, score.astype(np.float64)),
            "corr_nps_value": _corr(signal, value),
            "corr_latent": _corr(signal, latent),
        },
    }
    return Population(table=table, manifest=manifest)


# --------------------------------------------------------------------------- I/O


def write_population(
    population: Population, out_dir: str | Path, *, seed: int | None = None
) -> tuple[Path, Path]:
    """Write ``population.parquet`` and ``manifest.json`` to ``out_dir``.

    Output is deterministic: no timestamps, sorted JSON keys, fixed compression. The
    manifest gains ``seed`` and a ``files`` entry with the Parquet SHA-256.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "population.parquet"
    manifest_path = out / "manifest.json"
    pq.write_table(population.table, parquet_path, compression="zstd", compression_level=3)
    manifest = dict(population.manifest)
    manifest["seed"] = seed
    manifest["files"] = {
        "parquet": {
            "name": parquet_path.name,
            "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "n_rows": population.table.num_rows,
        }
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return parquet_path, manifest_path


def run(config: SimConfig, *, seed: int, out_dir: str | Path) -> tuple[Path, Path]:
    """Simulate with ``numpy.random.default_rng(seed)`` and write the outputs."""
    population = simulate_population(config, np.random.default_rng(seed))
    return write_population(population, out_dir, seed=seed)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: ``python -m silent97.data.simulate``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", default="configs/simulate.yaml", help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--n-customers", type=int, default=None, help="override N")
    parser.add_argument("--out", default=None, help="override the output directory")
    args = parser.parse_args(argv)

    config, seed, out_dir = load_config(args.config)
    if args.seed is not None:
        seed = args.seed
    if args.n_customers is not None:
        config = dataclasses.replace(config, n_customers=args.n_customers)
    if args.out is not None:
        out_dir = Path(args.out)
    parquet_path, manifest_path = run(config, seed=seed, out_dir=out_dir)
    truth = json.loads(manifest_path.read_text())["truth"]
    print(f"wrote {parquet_path} ({config.n_customers} customers, true NPS {truth['nps']:+.2f})")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
