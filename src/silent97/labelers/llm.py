"""LLM labeler: zero-shot probabilities for detractor / passive / promoter.

Each item (a public CFPB complaint or a synthetic text) is sent once to the Claude
API with fixed instructions. The response is constrained to a JSON schema
``{"detractor", "passive", "promoter"}`` with structured outputs. Probabilities are
validated and renormalized, and the predictor returns ``[P(promoter), P(detractor)]``.

Rules (CLAUDE.md, Phase 3):

- **Dry run by default.** :func:`label_items` makes no call unless
  ``dry_run=False``. Every run first prints the item count and the projected cost.
- **Hard caps.** ``max_items`` and ``max_spend_usd`` are cumulative over all runs
  that share a cache, using the cache's ``ledger.jsonl``. ``max_spend_usd: null``
  refuses any call, and the committed config ships that way so a human must set
  the cap. Before every request (sync) or batch (batch mode), the run reserves the
  *worst case* (estimated input plus ``max_tokens`` of output). Actual spend can
  therefore not pass the cap, as long as the input estimate (characters / 3.5,
  which overestimates English token counts) holds.
- **Never re-call a cached key.** Keys hash (model, prompt version, input), where
  the prompt version covers the instructions, schema and effort. Failed or
  refused items are cached too.
- **Only public CFPB or synthetic text** may be sent; any other ``source`` raises.
- **Credentials** come from ``.env`` (``ANTHROPIC_API_KEY``) or the environment
  and are never printed. The client always targets ``base_url`` from config
  (default ``https://api.anthropic.com``), never an inherited ``ANTHROPIC_BASE_URL``.

Model and effort come from config (default ``claude-opus-5-5`` at ``low`` effort,
for a classification task). Sync mode opts into server-side refusal fallbacks
(``fallbacks="default"``). Batch mode cannot, because the Batches API rejects the
parameter; there a refusal is a failed item.

Statistical note: a cached response is a fixed function of the item, so PPI's
validity argument applies. A failed item gets the constant prediction
(1/3, 1/3, 1/3), also a fixed function of the item. Failures are counted and reported.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from silent97.labelers.base import DETRACTOR, PROMOTER
from silent97.labelers.cache import DiskCache, make_key

__all__ = [
    "ALLOWED_SOURCES",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "CapError",
    "CredentialsError",
    "LLMConfig",
    "LLMItem",
    "LabelResult",
    "RunPlan",
    "SpendCapReached",
    "label_items",
    "load_api_key",
    "load_config",
    "make_client",
    "parse_probabilities",
    "plan_run",
    "render_cfpb_item",
]

PROMPT_VERSION: Final = "nps-zero-shot-v1"
ALLOWED_SOURCES: Final = frozenset({"cfpb", "synthetic"})
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
FAILED_PROBS: Final = (1 / 3, 1 / 3, 1 / 3)
_SUM_TOLERANCE: Final = 0.1

SYSTEM_PROMPT: Final = (
    "You read a consumer's complaint about a financial company and estimate how the "
    "consumer feels about the company, in the three Net Promoter Score groups:\n"
    "- detractor: clearly dissatisfied; likely to keep fighting the company's response "
    "or to warn others away;\n"
    "- passive: neither strongly negative nor positive; likely to accept a reasonable "
    "resolution;\n"
    "- promoter: satisfied overall and likely to recommend the company.\n"
    "Judge only from the complaint text, the product, the issue and the state given. "
    "Give calibrated probabilities that sum to 1: if the text gives little signal, "
    "stay close to what is typical for complaints. Respond with the JSON object only."
)

RESPONSE_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "detractor": {"type": "number"},
        "passive": {"type": "number"},
        "promoter": {"type": "number"},
    },
    "required": ["detractor", "passive", "promoter"],
    "additionalProperties": False,
}


class CapError(RuntimeError):
    """A configured cap (items or spend) forbids starting this run."""


class SpendCapReached(RuntimeError):
    """The run stopped because the next request could exceed the spend cap."""


class CredentialsError(RuntimeError):
    """No API key was found (the message never contains a secret)."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class LLMConfig:
    """LLM labeling settings; see ``configs/llm.yaml``."""

    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 2048
    mode: Literal["batch", "sync"] = "batch"
    max_items: int = 0
    max_spend_usd: float | None = None
    price_input_per_mtok: float = 4.0
    price_output_per_mtok: float = 20.0
    batch_discount: float = 0.5
    expected_output_tokens: int = 150
    chars_per_token: float = 3.5
    base_url: str = "https://api.anthropic.com"
    batch_size: int = 2000
    poll_seconds: float = 60.0
    sdk_max_retries: int = 4

    def __post_init__(self) -> None:
        if self.mode not in ("batch", "sync"):
            raise ValueError(f"mode must be 'batch' or 'sync', got {self.mode!r}")
        if self.max_items < 0 or self.batch_size < 1 or self.max_tokens < 16:
            raise ValueError("max_items >= 0, batch_size >= 1 and max_tokens >= 16 are required")
        if self.max_spend_usd is not None and not self.max_spend_usd >= 0:
            raise ValueError(f"max_spend_usd must be null or >= 0, got {self.max_spend_usd}")
        if not 0 < self.batch_discount <= 1:
            raise ValueError(f"batch_discount must lie in (0, 1], got {self.batch_discount}")

    @property
    def prompt_version(self) -> str:
        """Version tag hashed into every cache key (instructions, schema, effort)."""
        spec = json.dumps([SYSTEM_PROMPT, RESPONSE_SCHEMA, self.effort], sort_keys=True)
        return f"{PROMPT_VERSION}:{hashlib.sha256(spec.encode()).hexdigest()[:12]}"

    def price_factor(self) -> float:
        return self.batch_discount if self.mode == "batch" else 1.0

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        """USD for the given token counts under this config's mode."""
        raw = input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok
        return raw / 1e6 * self.price_factor()


def load_config(path: str | Path) -> LLMConfig:
    """Read ``LLMConfig`` fields from YAML; unknown keys raise."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(LLMConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown LLM config key(s) {unknown}")
    return LLMConfig(**raw)


# --------------------------------------------------------------------------- items


@dataclass(frozen=True)
class LLMItem:
    """One text to label. ``source`` must be in :data:`ALLOWED_SOURCES`."""

    item_id: str
    source: str
    text: str


def render_cfpb_item(
    *, complaint_id: str, narrative: str, product: str, issue: str, state: str
) -> LLMItem:
    """The user message for a CFPB complaint: product, issue, state and narrative only."""
    text = (
        f"Product: {product}\nIssue: {issue}\nState: {state or 'unknown'}\n\n"
        f"Complaint:\n{narrative.strip()}"
    )
    return LLMItem(item_id=str(complaint_id), source="cfpb", text=text)


# --------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, before any call."""

    n_items: int
    n_cached: int
    n_to_call: int
    projected_input_tokens: float
    projected_output_tokens: float
    projected_cost_usd: float
    worst_case_cost_usd: float
    spent_before_usd: float
    called_before: int

    def describe(self, cfg: LLMConfig) -> str:
        cap = (
            "NOT SET (no call allowed)"
            if cfg.max_spend_usd is None
            else f"${cfg.max_spend_usd:.2f}"
        )
        return (
            f"LLM labeling plan: model {cfg.model}, effort {cfg.effort}, mode {cfg.mode}\n"
            f"  items: {self.n_items} total, {self.n_cached} cached, {self.n_to_call} to call\n"
            f"  projected: {self.projected_input_tokens:,.0f} input + "
            f"{self.projected_output_tokens:,.0f} output tokens = ${self.projected_cost_usd:.2f} "
            f"(worst case ${self.worst_case_cost_usd:.2f} if every response used max_tokens)\n"
            f"  already spent from this cache: ${self.spent_before_usd:.2f} on "
            f"{self.called_before} items; spend cap {cap}; item cap {cfg.max_items}"
        )


def _keys(items: Sequence[LLMItem], cfg: LLMConfig) -> list[str]:
    return [
        make_key(model=cfg.model, prompt_version=cfg.prompt_version, payload=it.text)
        for it in items
    ]


def _input_tokens(item: LLMItem, cfg: LLMConfig) -> float:
    return (len(SYSTEM_PROMPT) + len(item.text)) / cfg.chars_per_token


def _ledger(cache: DiskCache) -> list[dict[str, Any]]:
    path = cache.root / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_ledger(cache: DiskCache, entry: dict[str, Any]) -> None:
    cache.root.mkdir(parents=True, exist_ok=True)
    with open(cache.root / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def plan_run(items: Sequence[LLMItem], cfg: LLMConfig, cache: DiskCache) -> RunPlan:
    """Count cached items and project the cost of the rest (no API call)."""
    keys = _keys(items, cfg)
    todo = [it for it, k in zip(items, keys, strict=True) if k not in cache]
    tin = sum(_input_tokens(it, cfg) for it in todo)
    tout = float(cfg.expected_output_tokens * len(todo))
    ledger = _ledger(cache)
    return RunPlan(
        n_items=len(items),
        n_cached=len(items) - len(todo),
        n_to_call=len(todo),
        projected_input_tokens=tin,
        projected_output_tokens=tout,
        projected_cost_usd=cfg.cost(tin, tout),
        worst_case_cost_usd=cfg.cost(tin, float(cfg.max_tokens * len(todo))),
        spent_before_usd=float(sum(e["cost_usd"] for e in ledger)),
        called_before=len(ledger),
    )


# --------------------------------------------------------------------------- responses


def parse_probabilities(text: str) -> tuple[NDArray[np.float64], bool]:
    """Parse ``{"detractor", "passive", "promoter"}`` into class probabilities.

    Returns ``(probs, ok)`` with probs ordered (detractor, passive, promoter). Values
    must be finite and in [0, 1] and sum to 1 within 0.1; they are then renormalized.
    Anything else returns the constant (1/3, 1/3, 1/3) with ``ok=False``.
    """
    try:
        data = json.loads(text)
        raw = np.array([float(data[k]) for k in ("detractor", "passive", "promoter")])
    except (ValueError, TypeError, KeyError):
        return np.array(FAILED_PROBS), False
    total = raw.sum()
    if (
        not np.isfinite(raw).all()
        or raw.min() < 0
        or raw.max() > 1
        or abs(total - 1) > _SUM_TOLERANCE
    ):
        return np.array(FAILED_PROBS), False
    return raw / total, True


def _record(message: Any, cfg: LLMConfig) -> dict[str, Any]:
    text = next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")
    if message.stop_reason in ("refusal", "max_tokens"):
        probs, ok = np.array(FAILED_PROBS), False
    else:
        probs, ok = parse_probabilities(text)
    tin, tout = int(message.usage.input_tokens), int(message.usage.output_tokens)
    return {
        "probs": probs.tolist(),
        "ok": bool(ok),
        "stop_reason": str(message.stop_reason),
        "served_model": str(getattr(message, "model", cfg.model)),
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": cfg.cost(tin, tout),
        "prompt_version": cfg.prompt_version,
    }


def _params(item: LLMItem, cfg: LLMConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": item.text}],
        "output_config": {
            "effort": cfg.effort,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
    }


# --------------------------------------------------------------------------- running


@dataclass(frozen=True)
class LabelResult:
    """Outcome of :func:`label_items`. ``predictions`` is None for a dry run."""

    plan: RunPlan
    predictions: NDArray[np.float64] | None
    class_probs: NDArray[np.float64] | None
    n_called: int
    n_failed: int
    spent_usd: float


def _check_sources(items: Iterable[LLMItem]) -> None:
    bad = sorted({it.source for it in items} - ALLOWED_SOURCES)
    if bad:
        raise ValueError(
            f"item source(s) {bad} not allowed; only {sorted(ALLOWED_SOURCES)} may be sent"
        )


def _check_caps(plan: RunPlan, cfg: LLMConfig) -> None:
    if plan.n_to_call == 0:
        return
    if cfg.max_spend_usd is None:
        raise CapError("max_spend_usd is not set; a human must set a spend cap before any call")
    if plan.called_before + plan.n_to_call > cfg.max_items:
        raise CapError(
            f"{plan.n_to_call} new items (+{plan.called_before} already called) exceeds "
            f"max_items = {cfg.max_items}"
        )
    if plan.spent_before_usd + plan.projected_cost_usd > cfg.max_spend_usd:
        raise CapError(
            f"projected spend ${plan.projected_cost_usd:.2f} (+${plan.spent_before_usd:.2f} "
            f"already spent) exceeds max_spend_usd = ${cfg.max_spend_usd:.2f}"
        )


def label_items(
    items: Sequence[LLMItem],
    cfg: LLMConfig,
    *,
    cache: DiskCache,
    client: Any = None,
    dry_run: bool = True,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> LabelResult:
    """Label ``items`` (cache first), respecting every rule in the module docstring.

    Returns predictions ``(len(items), 2)`` = [P(promoter), P(detractor)] and class
    probabilities ``(len(items), 3)`` in (detractor, passive, promoter) order.
    """
    _check_sources(items)
    plan = plan_run(items, cfg, cache)
    log(("[dry run] " if dry_run else "") + plan.describe(cfg))
    if dry_run:
        return LabelResult(plan, None, None, 0, 0, 0.0)
    _check_caps(plan, cfg)
    cap = (
        math.inf if cfg.max_spend_usd is None else cfg.max_spend_usd
    )  # None only if nothing to call

    keys = _keys(items, cfg)
    todo = [(k, it) for k, it in zip(keys, items, strict=True) if k not in cache]
    unique_todo = list(dict(todo).items())
    spent = plan.spent_before_usd
    new_spend = 0.0
    n_called = 0
    if unique_todo and client is None:
        raise ValueError("a client is required when dry_run=False and items are uncached")

    def worst(group: Sequence[tuple[str, LLMItem]]) -> float:
        tin = sum(_input_tokens(it, cfg) for _, it in group)
        return cfg.cost(tin, float(cfg.max_tokens * len(group)))

    def store(key: str, item: LLMItem, rec: dict[str, Any]) -> None:
        nonlocal spent, new_spend, n_called
        cache.put(key, {**rec, "item_id": item.item_id, "source": item.source})
        _append_ledger(
            cache,
            {"key": key, "item_id": item.item_id, "cost_usd": rec["cost_usd"], "ok": rec["ok"],
             "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"]},
        )  # fmt: skip
        spent += rec["cost_usd"]
        new_spend += rec["cost_usd"]
        n_called += 1

    if cfg.mode == "sync":
        for key, item in unique_todo:
            if spent + worst([(key, item)]) > cap:
                raise SpendCapReached(
                    f"stopped after {n_called} calls: the next could exceed ${cap:.2f} "
                    f"(spent ${spent:.2f})"
                )
            message = client.beta.messages.create(
                **_params(item, cfg), betas=[FALLBACK_BETA], fallbacks="default"
            )
            store(key, item, _record(message, cfg))
    else:
        for start in range(0, len(unique_todo), cfg.batch_size):
            chunk = unique_todo[start : start + cfg.batch_size]
            if spent + worst(chunk) > cap:
                raise SpendCapReached(
                    f"stopped after {n_called} items: the next batch of {len(chunk)} could "
                    f"exceed ${cap:.2f} (spent ${spent:.2f})"
                )
            by_id = dict(chunk)
            batch = client.messages.batches.create(
                requests=[{"custom_id": k, "params": _params(it, cfg)} for k, it in chunk]
            )
            log(f"submitted batch {batch.id} ({len(chunk)} items)")
            while batch.processing_status != "ended":
                sleep(cfg.poll_seconds)
                batch = client.messages.batches.retrieve(batch.id)
            for res in client.messages.batches.results(batch.id):
                item = by_id[res.custom_id]
                if res.result.type == "succeeded":
                    store(res.custom_id, item, _record(res.result.message, cfg))
                elif (
                    res.result.type == "errored"
                    and getattr(getattr(res.result, "error", None), "type", "") == "invalid_request"
                ):
                    store(res.custom_id, item, {
                        "probs": list(FAILED_PROBS), "ok": False, "stop_reason": "invalid_request",
                        "served_model": cfg.model, "input_tokens": 0, "output_tokens": 0,
                        "cost_usd": 0.0, "prompt_version": cfg.prompt_version,
                    })  # fmt: skip
                # server errors, cancellations and expiries stay uncached for a later run

    class_probs = np.empty((len(items), 3))
    n_failed = 0
    for i, key in enumerate(keys):
        entry = cache.get(key)
        if entry is None:  # pragma: no cover - only after a transient batch failure
            raise RuntimeError(f"item {items[i].item_id} is still unlabeled; rerun to retry it")
        class_probs[i] = entry["probs"]
        n_failed += not entry["ok"]
    predictions = np.column_stack([class_probs[:, PROMOTER], class_probs[:, DETRACTOR]])
    log(f"labeled {len(items)} items: {n_called} new calls, {n_failed} failed")
    log(f"spent ${new_spend:.4f} this run")
    return LabelResult(plan, predictions, class_probs, n_called, n_failed, new_spend)


# --------------------------------------------------------------------------- credentials


def load_api_key(env_file: str | Path = ".env") -> str:
    """``ANTHROPIC_API_KEY`` from ``env_file`` (git-ignored), else from the environment."""
    path = Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                return value.strip().strip("'\"")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    raise CredentialsError(
        f"no ANTHROPIC_API_KEY in {path} or in the environment; add it to .env (git-ignored)"
    )


def make_client(cfg: LLMConfig, *, env_file: str | Path = ".env") -> Any:
    """An ``anthropic.Anthropic`` client pinned to ``cfg.base_url`` (needs the ``llm`` extra)."""
    import anthropic

    return anthropic.Anthropic(
        api_key=load_api_key(env_file), base_url=cfg.base_url, max_retries=cfg.sdk_max_retries
    )
