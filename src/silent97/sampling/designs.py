r"""Sampling designs for choosing whom to survey.

Every design returns :class:`silent97.types.Design` with ``pi``, the inclusion
probability of *every* customer in the pool (not only those drawn), and every ``pi``
is strictly positive (CLAUDE.md section 6, rule 5). Design-based estimators reweight
by :math:`1/\pi`; a zero probability would make that impossible.

- :func:`uniform`: simple random sampling without replacement, :math:`\pi_i = n/N`.
- :func:`stratified`: a fixed :math:`n_h` per stratum, simple random sampling within each,
  :math:`\pi_i = n_h/N_h`. Proportional allocation uses :math:`n_h \propto N_h`; Neyman
  allocation uses :math:`n_h \propto N_h \sigma_h` (Neyman 1934), with :math:`\sigma_h`
  from the model's predictions (:func:`neyman_sigma`), so it uses no survey data.
  Every stratum gets at least 2 responses, so its variance can be estimated.
- :func:`active`: active statistical inference (Zrnic & Candès, ICML 2024). Each
  customer is surveyed *independently* with probability :math:`\pi_i` (Poisson
  sampling, so the realized n is random with mean n), where

  .. math::

      \pi_i = (1 - \tau)\, n \frac{u_i}{\sum_j u_j} + \tau \frac{n}{N},

  capped at 1 with the budget redistributed. Here :math:`u_i` is the model's
  uncertainty and :math:`\tau` the uniform floor (default 0.1), so
  :math:`\pi_i \ge 0.1\, n/N`. For PPI with :math:`\lambda = 1`, the variance-minimizing
  :math:`\pi` is proportional to :math:`\sqrt{E[(v - h)^2 \mid x]}`; with a calibrated
  model that is the predicted standard deviation of the NPS value, :func:`nps_uncertainty`.

All functions take a ``numpy.random.Generator`` (rule 9) and are pure.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.types import Design

__all__ = [
    "MIN_PER_STRATUM",
    "active",
    "active_probabilities",
    "check_pi",
    "neyman_sigma",
    "nps_uncertainty",
    "stratified",
    "uniform",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MIN_PER_STRATUM: Final = 2


def check_pi(pi: ArrayLike) -> FloatArray:
    """Assert inclusion probabilities are finite, strictly positive and at most 1."""
    p = np.asarray(pi, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("pi must be a finite 1-D array")
    if not (p > 0).all():
        raise ValueError("every inclusion probability must be strictly positive")
    if not (p <= 1 + 1e-12).all():
        raise ValueError("every inclusion probability must be at most 1")
    return p


def _check_n(n: int, size: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or not 1 <= n <= size:
        raise ValueError(f"n must be an integer in [1, {size}], got {n!r}")


def uniform(n: int, size: int, rng: np.random.Generator) -> Design:
    """Simple random sample of ``n`` from a pool of ``size``; ``pi = n / size``."""
    _check_n(n, size)
    idx = np.sort(rng.choice(size, size=n, replace=False)).astype(np.int64)
    pi = check_pi(np.full(size, n / size))
    return Design(idx=idx, pi=pi, name="uniform")


def nps_uncertainty(p: ArrayLike) -> FloatArray:
    r"""Predicted sd of the NPS value / 100: :math:`\sqrt{p_P + p_D - (p_P - p_D)^2}`."""
    q = np.asarray(p, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
        raise ValueError(f"p must be a finite (N, 2) array, got shape {q.shape}")
    var = q[:, 0] + q[:, 1] - (q[:, 0] - q[:, 1]) ** 2
    out: FloatArray = np.sqrt(np.clip(var, 0.0, None))
    return out


def neyman_sigma(strata: ArrayLike, p: ArrayLike) -> FloatArray:
    r"""Per-stratum sd of the NPS value / 100 implied by the predictions.

    By the law of total variance, :math:`\sigma_h^2 = \mathrm{mean}_h \mathrm{Var}(v \mid x)
    + \mathrm{var}_h E[v \mid x]`, both computed from the predictions alone.
    """
    s = np.asarray(strata)
    q = np.asarray(p, dtype=np.float64)
    labels, inv = np.unique(s, return_inverse=True)
    within = nps_uncertainty(q) ** 2
    mean = q[:, 0] - q[:, 1]
    out = np.empty(labels.size)
    for h in range(labels.size):
        m = inv == h
        out[h] = np.sqrt(within[m].mean() + mean[m].var())
    return out


def _allocate(n: int, weights: FloatArray, caps: IntArray) -> IntArray:
    """Integer n_h with sum n, MIN_PER_STRATUM <= n_h <= N_h, close to n * w_h / sum(w)."""
    lo = np.minimum(MIN_PER_STRATUM, caps).astype(np.float64)
    hi = caps.astype(np.float64)
    target = np.zeros_like(weights)
    free = np.ones(weights.size, dtype=bool)
    remaining = float(n)
    for _ in range(weights.size + 1):  # clamp to bounds, redistribute the rest
        w = np.where(free, weights, 0.0)
        share = (
            remaining * w / w.sum() if w.sum() > 0 else np.where(free, remaining / free.sum(), 0)
        )
        target = np.where(free, share, target)
        low, high = free & (target < lo), free & (target > hi)
        if not (low.any() or high.any()):
            break
        target[low], target[high] = lo[low], hi[high]
        free &= ~(low | high)
        remaining = n - target[~free].sum()
    alloc = np.floor(target + 1e-9).astype(np.int64)
    order = np.argsort(-(target - alloc), kind="stable")
    for h in order:  # largest remainder, within caps
        if alloc.sum() >= n:
            break
        if alloc[h] < caps[h]:
            alloc[h] += 1
    for h in np.argsort(-caps, kind="stable"):  # any shortfall from caps
        while alloc.sum() < n and alloc[h] < caps[h]:
            alloc[h] += 1
    return alloc


def stratified(
    n: int,
    strata: ArrayLike,
    rng: np.random.Generator,
    *,
    allocation: str = "proportional",
    sigma: ArrayLike | None = None,
) -> Design:
    """Stratified simple random sampling (see the module docstring).

    ``strata`` gives each pool customer's stratum label. For Neyman allocation,
    ``sigma`` holds one value per stratum in sorted label order (:func:`neyman_sigma`).
    """
    s = np.asarray(strata)
    if s.ndim != 1:
        raise ValueError("strata must be 1-D, one label per pool customer")
    _check_n(n, s.size)
    labels, inv = np.unique(s, return_inverse=True)
    sizes = np.bincount(inv).astype(np.int64)
    if n < MIN_PER_STRATUM * labels.size:
        raise ValueError(f"need n >= {MIN_PER_STRATUM} per stratum ({labels.size} strata), got {n}")
    if allocation == "proportional":
        weights = sizes.astype(np.float64)
    elif allocation == "neyman":
        if sigma is None:
            raise ValueError("Neyman allocation needs sigma, one value per stratum")
        sig = np.asarray(sigma, dtype=np.float64)
        if sig.shape != (labels.size,) or not np.isfinite(sig).all() or (sig < 0).any():
            raise ValueError(f"sigma must be {labels.size} finite non-negative values")
        weights = sizes * sig
    else:
        raise ValueError(f"allocation must be 'proportional' or 'neyman', got {allocation!r}")
    alloc = _allocate(n, weights, sizes)
    picks = []
    for h in range(labels.size):
        members = np.flatnonzero(inv == h)
        picks.append(rng.choice(members, size=int(alloc[h]), replace=False))
    idx = np.sort(np.concatenate(picks)).astype(np.int64)
    pi = check_pi((alloc / sizes)[inv])
    return Design(idx=idx, pi=pi, name=f"stratified_{allocation}")


def active_probabilities(n: int, uncertainty: ArrayLike, *, floor: float = 0.1) -> FloatArray:
    """Inclusion probabilities for :func:`active` (sum n, at least ``floor * n / N``, at most 1)."""
    u = np.asarray(uncertainty, dtype=np.float64)
    if u.ndim != 1 or not np.isfinite(u).all() or (u < 0).any():
        raise ValueError("uncertainty must be a finite, non-negative 1-D array")
    size = u.size
    _check_n(n, size)
    if not 0 < floor <= 1:
        raise ValueError(f"floor must lie in (0, 1], got {floor}")
    rate = n / size
    shape = u / u.sum() if u.sum() > 0 else np.full(size, 1.0 / size)
    pi = (1 - floor) * n * shape + floor * rate
    capped = np.zeros(size, dtype=bool)
    for _ in range(size):  # cap at 1, give the excess to the uncapped, proportionally
        over = ~capped & (pi > 1)
        if not over.any():
            break
        capped |= over
        pi[capped] = 1.0
        rest = n - capped.sum()
        free = ~capped
        pi[free] = pi[free] * rest / pi[free].sum()
    return check_pi(pi)


def active(
    n: int, uncertainty: ArrayLike, rng: np.random.Generator, *, floor: float = 0.1
) -> Design:
    """Poisson sampling with :func:`active_probabilities`; the expected sample size is n."""
    pi = active_probabilities(n, uncertainty, floor=floor)
    idx = np.flatnonzero(rng.random(pi.size) < pi).astype(np.int64)
    return Design(idx=idx, pi=pi, name="active")
