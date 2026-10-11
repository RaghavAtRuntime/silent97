"""Local LLM labeler: zero-shot probabilities for detractor / passive / promoter.

An open-weight instruction model (set in ``configs/llm.yaml``; Apache-2.0 models
only) runs locally through Hugging Face ``transformers``, on CPU or Apple-silicon
``mps``. Nothing is sent to any external API, and there is no spend.

How probabilities are produced. Each item gets one forward pass, with no
generation. The chat-formatted prompt asks for a single letter, A (detractor),
B (passive) or C (promoter). The model's next-token probabilities for the three
letter tokens are renormalized to sum to 1:

.. math::

    \\hat p_k = \\frac{\\exp(\\ell_k)}{\\sum_{j \\in \\{A,B,C\\}} \\exp(\\ell_j)}.

This is the standard zero-shot classification readout, and it is deterministic.
Very small models are unreliable at writing calibrated numbers as JSON, so this
replaces the "JSON probabilities" readout planned for an API model. The labeler
returns ``[P(promoter), P(detractor)]``.

Rules (CLAUDE.md, Phase 3, adapted to a local model):

- **Dry run by default:** :func:`label_items` scores nothing unless
  ``dry_run=False``. Every run first prints the item count and the projected
  compute (prompt tokens and seconds).
- **Hard item cap:** ``max_items``, cumulative over all runs sharing a cache (from
  the cache's ``ledger.jsonl``). Spend is zero by construction.
- **Never recompute a cached key:** keys hash (model@revision, prompt version,
  input text). The prompt version covers the instructions, the option letters and
  the truncation length. Invalid outputs are cached too.
- **Only public CFPB or synthetic text** is accepted (``source``), even though
  nothing leaves the machine.
- **Bounded input:** prompts longer than ``max_input_tokens`` tokens have the item
  text truncated from the end. Truncations are counted.

Statistical note: a cached score is a fixed function of the item, so PPI's validity
argument applies. An invalid output (non-finite, or not summing to 1) gets the
constant (1/3, 1/3, 1/3), which is also a fixed function of the item. PPI does not
need the model to be calibrated; miscalibration costs width only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from silent97.labelers.base import DETRACTOR, PROMOTER
from silent97.labelers.cache import DiskCache, make_key

__all__ = [
    "ALLOWED_SOURCES",
    "OPTION_LETTERS",
    "SYSTEM_PROMPT",
    "CapError",
    "LLMConfig",
    "LLMItem",
    "LabelResult",
    "RunPlan",
    "Scorer",
    "check_probabilities",
    "label_items",
    "load_config",
    "plan_run",
    "render_cfpb_item",
    "render_prompt",
    "transformers_scorer",
    "upcast_linears",
]

_DTYPES: Final = frozenset({"float32", "bfloat16", "float16"})

PROMPT_VERSION: Final = "nps-local-choice-v1"
ALLOWED_SOURCES: Final = frozenset({"cfpb", "synthetic"})
OPTION_LETTERS: Final = ("A", "B", "C")
"""Answer letters for (detractor, passive, promoter), in class order."""
FAILED_PROBS: Final = (1 / 3, 1 / 3, 1 / 3)
_SUM_TOLERANCE: Final = 0.1

SYSTEM_PROMPT: Final = (
    "You read a consumer's complaint about a financial company and judge how the "
    "consumer feels about the company. Choose one group:\n"
    "A) detractor: clearly dissatisfied; likely to keep fighting the company's "
    "response or to warn others away.\n"
    "B) passive: neither strongly negative nor positive; likely to accept a "
    "reasonable resolution.\n"
    "C) promoter: satisfied overall and likely to recommend the company.\n"
    "Judge only from the complaint text, the product, the issue and the state given. "
    "Answer with the single letter A, B or C."
)

Scorer = Callable[[list[str]], Any]
"""Maps item texts to class probabilities, shape (n, 3) in (detractor, passive, promoter)."""


class CapError(RuntimeError):
    """The configured item cap forbids starting this run."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class LLMConfig:
    """Local LLM labeling settings; see ``configs/llm.yaml``."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    max_items: int = 10_000
    max_input_tokens: int = 512
    batch_size: int = 8
    device: str = "cpu"
    dtype: str = "float32"
    weights_dtype: str | None = None
    threads: int | None = None
    tokens_per_second: float = 450.0
    chars_per_token: float = 3.5

    def __post_init__(self) -> None:
        if self.max_items < 0 or self.batch_size < 1 or self.max_input_tokens < 16:
            raise ValueError("max_items >= 0, batch_size >= 1 and max_input_tokens >= 16 required")
        if self.tokens_per_second <= 0 or self.chars_per_token <= 0:
            raise ValueError("tokens_per_second and chars_per_token must be positive")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if self.weights_dtype is not None and self.weights_dtype not in _DTYPES:
            raise ValueError(
                f"weights_dtype must be null or one of {sorted(_DTYPES)}, "
                f"got {self.weights_dtype!r}"
            )

    @property
    def model_id(self) -> str:
        """Model name plus pinned revision, hashed into every cache key."""
        return f"{self.model}@{self.revision or 'main'}"

    @property
    def prompt_version(self) -> str:
        """Version tag hashed into every cache key (instructions, options, truncation)."""
        spec = json.dumps([SYSTEM_PROMPT, OPTION_LETTERS, self.max_input_tokens])
        return f"{PROMPT_VERSION}:{hashlib.sha256(spec.encode()).hexdigest()[:12]}"


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


def _check_sources(items: Iterable[LLMItem]) -> None:
    bad = sorted({it.source for it in items} - ALLOWED_SOURCES)
    if bad:
        raise ValueError(f"item source(s) {bad} not allowed; only {sorted(ALLOWED_SOURCES)}")


# --------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, before any scoring."""

    n_items: int
    n_cached: int
    n_to_score: int
    projected_tokens: float
    projected_seconds: float
    scored_before: int

    def describe(self, cfg: LLMConfig) -> str:
        hours = self.projected_seconds / 3600
        return (
            f"local LLM labeling plan: model {cfg.model_id} on {cfg.device}\n"
            f"  items: {self.n_items} total, {self.n_cached} cached, {self.n_to_score} to score\n"
            f"  projected: about {self.projected_tokens:,.0f} prompt tokens, "
            f"{self.projected_seconds:,.0f} s ({hours:.1f} h) at "
            f"{cfg.tokens_per_second:.0f} tokens/s; spend $0 (local)\n"
            f"  item cap {cfg.max_items}; already scored from this cache: {self.scored_before}"
        )


def _keys(items: Sequence[LLMItem], cfg: LLMConfig) -> list[str]:
    return [
        make_key(model=cfg.model_id, prompt_version=cfg.prompt_version, payload=it.text)
        for it in items
    ]


def _ledger_count(cache: DiskCache) -> int:
    path = cache.root / "ledger.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def plan_run(items: Sequence[LLMItem], cfg: LLMConfig, cache: DiskCache) -> RunPlan:
    """Count cached items and project the compute for the rest (nothing is scored)."""
    keys = _keys(items, cfg)
    todo = {k: it for k, it in zip(keys, items, strict=True) if k not in cache}
    tokens = sum(
        min((len(SYSTEM_PROMPT) + len(it.text)) / cfg.chars_per_token, cfg.max_input_tokens)
        for it in todo.values()
    )
    return RunPlan(
        n_items=len(items),
        n_cached=len(items) - sum(1 for k in keys if k in todo),
        n_to_score=len(todo),
        projected_tokens=float(tokens),
        projected_seconds=float(tokens / cfg.tokens_per_second),
        scored_before=_ledger_count(cache),
    )


# --------------------------------------------------------------------------- scoring


def check_probabilities(raw: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Validate (detractor, passive, promoter) probabilities and renormalize.

    Values must be finite and in [0, 1] and sum to 1 within 0.1. Otherwise the
    constant (1/3, 1/3, 1/3) is returned with ``ok=False``.
    """
    p = np.asarray(raw, dtype=np.float64)
    total = float(p.sum()) if p.shape == (3,) else float("nan")
    if not np.isfinite(p).all() or p.shape != (3,) or p.min() < 0 or p.max() > 1:
        return np.array(FAILED_PROBS), False
    if abs(total - 1.0) > _SUM_TOLERANCE:
        return np.array(FAILED_PROBS), False
    return p / total, True


@dataclass(frozen=True)
class LabelResult:
    """Outcome of :func:`label_items`. ``predictions`` is None for a dry run."""

    plan: RunPlan
    predictions: NDArray[np.float64] | None
    class_probs: NDArray[np.float64] | None
    n_scored: int
    n_failed: int
    seconds: float


def label_items(
    items: Sequence[LLMItem],
    cfg: LLMConfig,
    *,
    cache: DiskCache,
    scorer: Scorer | None = None,
    dry_run: bool = True,
    log: Callable[[str], None] = print,
) -> LabelResult:
    """Label ``items`` (cache first), following the rules in the module docstring.

    Returns predictions ``(len(items), 2)`` = [P(promoter), P(detractor)] and class
    probabilities ``(len(items), 3)`` in (detractor, passive, promoter) order.
    """
    _check_sources(items)
    plan = plan_run(items, cfg, cache)
    log(("[dry run] " if dry_run else "") + plan.describe(cfg))
    if dry_run:
        return LabelResult(plan, None, None, 0, 0, 0.0)
    if plan.n_to_score and plan.scored_before + plan.n_to_score > cfg.max_items:
        raise CapError(
            f"{plan.n_to_score} new items (+{plan.scored_before} already scored) exceeds "
            f"max_items = {cfg.max_items}"
        )
    keys = _keys(items, cfg)
    todo = list({k: it for k, it in zip(keys, items, strict=True) if k not in cache}.items())
    # Similar lengths share a batch, so less compute goes to padding. Scores do not
    # depend on batch composition (left padding is masked; tested).
    todo.sort(key=lambda kv: len(kv[1].text))
    if todo and scorer is None:
        raise ValueError("a scorer is required when dry_run=False and items are uncached")

    start = time.perf_counter()
    for b in range(0, len(todo), cfg.batch_size):
        batch = todo[b : b + cfg.batch_size]
        assert scorer is not None
        raw = np.asarray(scorer([it.text for _, it in batch]), dtype=np.float64)
        if raw.shape != (len(batch), 3):
            raise ValueError(f"scorer returned shape {raw.shape}, expected ({len(batch)}, 3)")
        for (key, item), row in zip(batch, raw, strict=True):
            probs, ok = check_probabilities(row)
            cache.put(
                key,
                {
                    "probs": probs.tolist(),
                    "ok": ok,
                    "model": cfg.model_id,
                    "prompt_version": cfg.prompt_version,
                    "item_id": item.item_id,
                    "source": item.source,
                },
            )
            cache.root.mkdir(parents=True, exist_ok=True)
            with open(cache.root / "ledger.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "item_id": item.item_id, "ok": ok}) + "\n")
        done = min(b + cfg.batch_size, len(todo))
        if done % max(cfg.batch_size * 50, 1) < cfg.batch_size or done == len(todo):
            log(f"scored {done}/{len(todo)} items in {time.perf_counter() - start:.0f} s")
    seconds = time.perf_counter() - start

    class_probs = np.empty((len(items), 3))
    n_failed = 0
    for i, key in enumerate(keys):
        entry = cache.get(key)
        if entry is None:  # pragma: no cover - defensive
            raise RuntimeError(f"item {items[i].item_id} is unlabeled")
        class_probs[i] = entry["probs"]
        n_failed += not entry["ok"]
    predictions = np.column_stack([class_probs[:, PROMOTER], class_probs[:, DETRACTOR]])
    log(f"labeled {len(items)} items: {len(todo)} newly scored, {n_failed} invalid")
    return LabelResult(plan, predictions, class_probs, len(todo), n_failed, seconds)


def render_prompt(tok: Any, text: str) -> str:
    """The chat-formatted prompt for one item, ending at the answer position.

    Thinking is switched off (``enable_thinking=False``): the readout needs the answer
    letter as the very next token, not a ``<think>`` block. Templates that do not use
    the flag ignore it.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    out: str = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return out


def upcast_linears(model: Any, *, compute_dtype: Any) -> int:
    """Keep every ``nn.Linear`` weight in its stored dtype, but compute in ``compute_dtype``.

    Each linear layer's weight is cast to ``compute_dtype`` only for its own matrix
    product, so a bf16 checkpoint needs about half the memory of a float32 copy while
    giving float32 results (CPUs without native bf16 run float32 several times
    faster). Every other parameter and buffer is converted to ``compute_dtype``;
    embeddings whose weight is shared with a linear layer (tied) keep it and have
    their output cast. Returns the number of linear layers wrapped.
    """
    import torch
    import torch.nn.functional as F

    class _UpcastLinear(torch.nn.Module):
        def __init__(self, lin: Any) -> None:
            super().__init__()
            self.weight = lin.weight
            self.bias = (
                None
                if lin.bias is None
                else torch.nn.Parameter(lin.bias.detach().to(compute_dtype))
            )

        def forward(self, x: Any) -> Any:
            return F.linear(x, self.weight.to(compute_dtype), self.bias)

    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(module, name, _UpcastLinear(child))
                n += 1
    kept = {id(m.weight) for m in model.modules() if isinstance(m, _UpcastLinear)}
    for param in model.parameters():
        if id(param) not in kept and param.is_floating_point():
            param.data = param.data.to(compute_dtype)
    for buf in model.buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(compute_dtype)
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding) and module.weight.dtype != compute_dtype:
            module.register_forward_hook(lambda _m, _i, out: out.to(compute_dtype))
    return n


def transformers_scorer(cfg: LLMConfig) -> Scorer:
    """A local ``transformers`` scorer (needs the ``text`` extra and the model weights).

    Prompts are rendered with the model's chat template (system + user + generation
    prompt, thinking off) and left-padded, so the last position of every row is its own
    next-token position. The item text is truncated so the full prompt fits
    ``max_input_tokens``. With ``weights_dtype`` set and different from ``dtype``, the
    weights are loaded in ``weights_dtype`` and the compute runs in ``dtype``
    (:func:`upcast_linears`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    tok: Any = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    compute = getattr(torch, cfg.dtype)
    stored = getattr(torch, cfg.weights_dtype or cfg.dtype)
    model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.model, revision=cfg.revision, dtype=stored
    )
    if stored != compute:
        upcast_linears(model, compute_dtype=compute)
    model = model.to(cfg.device)
    model.eval()
    option_ids = []
    for letter in OPTION_LETTERS:
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option {letter!r} is not a single token for {cfg.model}: {ids}")
        option_ids.append(ids[0])

    def render(text: str) -> str:
        return render_prompt(tok, text)

    overhead = len(tok(render(""), add_special_tokens=False)["input_ids"])
    budget = max(cfg.max_input_tokens - overhead, 8)

    def score(texts: list[str]) -> NDArray[np.float64]:
        truncated = 0
        prompts = []
        for text in texts:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > budget:
                text = tok.decode(ids[:budget])
                truncated += 1
            prompts.append(render(text))
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            cfg.device
        )
        with torch.inference_mode():
            # Only the last position's logits are needed; skipping the vocabulary
            # projection elsewhere saves time and about 2 GB per batch for a 150k vocab.
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        picked = logits[:, option_ids].double()
        probs = torch.softmax(picked, dim=-1).cpu().numpy()
        score.last_truncated = truncated  # type: ignore[attr-defined]
        return np.asarray(probs, dtype=np.float64)

    score.last_truncated = 0  # type: ignore[attr-defined]
    return score
