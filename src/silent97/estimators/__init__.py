"""Estimators of NPS. Every module here is pure: no I/O, no global state.

All estimators share the signature ``f(y_lab, p_lab, p_unlab, *, alpha=0.1, **kw)``
and return a :class:`silent97.types.Estimate`. ``ppi_crossfit`` expects
``p_unlab`` of shape (N, K, 2) from :func:`silent97.estimators.ppi.cross_fit`.
``ppi_pp_cf`` also needs ``rng`` (or ``folds``) for its fold split.
"""

from collections.abc import Callable
from typing import Final

from silent97.estimators import active as _active
from silent97.estimators import classical as _classical
from silent97.estimators import ppi as _ppi
from silent97.estimators import stratified as _stratified
from silent97.types import Estimate

__all__ = ["DESIGN_BASED", "REGISTRY", "VALID_UNDER_MCAR"]

REGISTRY: Final[dict[str, Callable[..., Estimate]]] = {
    "classical": _classical.classical,
    "naive_model": _classical.naive_model,
    "ppi": _ppi.ppi,
    "ppi_pp": _ppi.ppi_pp,
    "ppi_pp_cf": _ppi.ppi_pp_cf,
    "ppi_crossfit": _ppi.ppi_crossfit,
    "ppi_active": _active.ppi_active,
    "ppi_stratified": _stratified.ppi_stratified,
}

DESIGN_BASED: Final = frozenset({"ppi_active", "ppi_stratified"})
"""Estimators valid under their own sampling design (they need pi or strata)."""

VALID_UNDER_MCAR: Final = frozenset({"classical", "ppi", "ppi_pp", "ppi_pp_cf", "ppi_crossfit"})
"""Estimators whose intervals are valid when the labeled set is a random sample."""
