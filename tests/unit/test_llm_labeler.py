"""Tests for the local LLM labeler.

The labeling logic (dry run, caps, cache, sources, failures) is tested with a stub
scorer. The real ``transformers`` scorer is tested end to end on a tiny randomly
initialized model with a locally trained tokenizer, so no download is needed; those
tests need the optional ``text`` extra (PyTorch) and are skipped without it.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from silent97.estimators import nps
from silent97.labelers import llm
from silent97.labelers.cache import DiskCache


def _items(n: int, source: str = "cfpb") -> list[llm.LLMItem]:
    if source != "cfpb":
        return [llm.LLMItem(item_id=str(i), source=source, text=f"text {i}") for i in range(n)]
    return [
        llm.render_cfpb_item(
            complaint_id=str(i),
            narrative=f"I was charged a fee {i} times and nobody called back.",
            product="Credit card",
            issue="Billing disputes",
            state="CA",
        )
        for i in range(n)
    ]


def _cfg(**overrides: Any) -> llm.LLMConfig:
    return llm.LLMConfig(**{"model": "stub/model", "max_items": 100, **overrides})


class StubScorer:
    """Returns fixed class probabilities (detractor, passive, promoter)."""

    def __init__(self, probs: tuple[float, float, float] = (0.6, 0.3, 0.1)) -> None:
        self.probs = np.array(probs, dtype=float)
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.tile(self.probs, (len(texts), 1))

    @property
    def n_scored(self) -> int:
        return sum(len(c) for c in self.calls)


# --------------------------------------------------------------------------- planning


def test_dry_run_scores_nothing_and_prints_plan(tmp_path: Path) -> None:
    lines: list[str] = []
    scorer = StubScorer()
    res = llm.label_items(
        _items(20), _cfg(), cache=DiskCache(tmp_path), scorer=scorer, log=lines.append
    )
    assert res.predictions is None and scorer.n_scored == 0
    assert res.plan.n_items == 20 and res.plan.n_to_score == 20 and res.plan.n_cached == 0
    text = "\n".join(lines).lower()
    assert "dry run" in text and "20" in text and "stub/model" in text


def test_projection(tmp_path: Path) -> None:
    items = _items(10)
    cfg = _cfg(chars_per_token=4.0, tokens_per_second=100.0)
    plan = llm.plan_run(items, cfg, DiskCache(tmp_path))
    tokens = sum(
        min((len(llm.SYSTEM_PROMPT) + len(it.text)) / 4.0, cfg.max_input_tokens) for it in items
    )
    assert plan.projected_tokens == pytest.approx(tokens)
    assert plan.projected_seconds == pytest.approx(tokens / 100.0)


def test_refuses_over_item_cap(tmp_path: Path) -> None:
    scorer = StubScorer()
    with pytest.raises(llm.CapError, match="max_items"):
        llm.label_items(
            _items(11), _cfg(max_items=10), cache=DiskCache(tmp_path), scorer=scorer, dry_run=False
        )
    assert scorer.n_scored == 0


def test_item_cap_is_cumulative(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    llm.label_items(_items(6), _cfg(max_items=10), cache=cache, scorer=StubScorer(), dry_run=False)
    more = [llm.LLMItem(item_id=f"x{i}", source="cfpb", text=f"new {i}") for i in range(5)]
    with pytest.raises(llm.CapError, match="max_items"):
        llm.label_items(more, _cfg(max_items=10), cache=cache, scorer=StubScorer(), dry_run=False)


def test_rejects_disallowed_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        llm.label_items(_items(2, source="customer_db"), _cfg(), cache=DiskCache(tmp_path))


def test_synthetic_source_allowed(tmp_path: Path) -> None:
    res = llm.label_items(
        _items(2, source="synthetic"),
        _cfg(),
        cache=DiskCache(tmp_path),
        scorer=StubScorer(),
        dry_run=False,
    )
    assert res.predictions is not None


def test_scorer_required_for_real_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scorer"):
        llm.label_items(_items(2), _cfg(), cache=DiskCache(tmp_path), scorer=None, dry_run=False)


# --------------------------------------------------------------------------- scoring and cache


def test_labels_batches_and_caches(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    scorer = StubScorer()
    res = llm.label_items(_items(7), _cfg(batch_size=3), cache=cache, scorer=scorer, dry_run=False)
    assert [len(c) for c in scorer.calls] == [3, 3, 1]
    assert res.predictions is not None and res.class_probs is not None
    nps.validate_predictions(res.predictions)
    np.testing.assert_allclose(res.predictions, [[0.1, 0.6]] * 7)
    np.testing.assert_allclose(res.class_probs, [[0.6, 0.3, 0.1]] * 7)
    assert res.n_scored == 7 and res.n_failed == 0
    # Second run: everything is cached, nothing is scored again.
    scorer2 = StubScorer()
    res2 = llm.label_items(
        _items(7), _cfg(batch_size=3), cache=cache, scorer=scorer2, dry_run=False
    )
    assert scorer2.n_scored == 0 and res2.n_scored == 0
    np.testing.assert_array_equal(res2.predictions, res.predictions)


def test_duplicate_texts_scored_once(tmp_path: Path) -> None:
    items = _items(3) + _items(3)  # same texts twice
    scorer = StubScorer()
    res = llm.label_items(items, _cfg(), cache=DiskCache(tmp_path), scorer=scorer, dry_run=False)
    assert scorer.n_scored == 3 and res.predictions is not None and len(res.predictions) == 6


def test_scorer_sees_the_rendered_text_without_outcome(tmp_path: Path) -> None:
    scorer = StubScorer()
    llm.label_items(_items(1), _cfg(), cache=DiskCache(tmp_path), scorer=scorer, dry_run=False)
    text = scorer.calls[0][0]
    assert "Credit card" in text and "Billing disputes" in text and "CA" in text
    assert "disputed" not in text.lower()


def test_cache_key_covers_model_prompt_and_input(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    llm.label_items(_items(2), _cfg(), cache=cache, scorer=StubScorer(), dry_run=False)
    assert llm.plan_run(_items(2), _cfg(), cache).n_to_score == 0
    assert llm.plan_run(_items(2), _cfg(model="other/model"), cache).n_to_score == 2
    assert llm.plan_run(_items(2), _cfg(revision="abc123"), cache).n_to_score == 2
    assert llm.plan_run(_items(2), _cfg(max_input_tokens=128), cache).n_to_score == 2
    edited = _items(2)
    edited[0] = llm.LLMItem(item_id="0", source="cfpb", text=edited[0].text + " Edited.")
    assert llm.plan_run(edited, _cfg(), cache).n_to_score == 1


@pytest.mark.parametrize(
    "bad", [(np.nan, 0.5, 0.5), (-0.1, 0.6, 0.5), (0.0, 0.0, 0.0), (0.5, 0.2, 0.1)]
)
def test_invalid_scores_become_constant_and_are_cached(
    tmp_path: Path, bad: tuple[float, float, float]
) -> None:
    cache = DiskCache(tmp_path)
    res = llm.label_items(_items(2), _cfg(), cache=cache, scorer=StubScorer(bad), dry_run=False)
    assert res.n_failed == 2
    assert res.predictions is not None
    np.testing.assert_allclose(res.predictions, [[1 / 3, 1 / 3]] * 2)
    scorer2 = StubScorer()
    llm.label_items(_items(2), _cfg(), cache=cache, scorer=scorer2, dry_run=False)
    assert scorer2.n_scored == 0


def test_check_probabilities() -> None:
    probs, ok = llm.check_probabilities(np.array([0.5, 0.3, 0.2]))
    assert ok
    np.testing.assert_allclose(probs, [0.5, 0.3, 0.2])
    probs, ok = llm.check_probabilities(np.array([0.5, 0.3, 0.25]))
    assert ok and probs.sum() == pytest.approx(1.0)


def test_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "llm.yaml"
    path.write_text("model: Qwen/Qwen2.5-1.5B-Instruct\nmax_items: 10\nbatch_size: 4\n")
    cfg = llm.load_config(path)
    assert cfg.model == "Qwen/Qwen2.5-1.5B-Instruct" and cfg.batch_size == 4
    path.write_text("modle: x\n")
    with pytest.raises(ValueError, match="unknown"):
        llm.load_config(path)


def test_repo_config_is_local() -> None:
    cfg = llm.load_config(Path("configs/llm.yaml"))
    assert cfg.device == "cpu" and "/" in cfg.model


# --------------------------------------------------------------------------- real scorer


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random-weight Llama with a locally trained BPE tokenizer and chat template."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    out = tmp_path_factory.mktemp("tiny_llm")
    corpus = [
        "A B C detractor passive promoter",
        "Product: Credit card Issue: Billing disputes State: CA Complaint:",
        "I was charged a fee times and nobody called back.",
        "system user assistant answer letter only",
    ] * 50
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
    )
    tok.train_from_iterator(corpus, trainer)
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    fast.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}<|im_end|> "
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
    )
    fast.save_pretrained(out)
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        pad_token_id=fast.pad_token_id,
    )  # fmt: skip
    transformers.LlamaForCausalLM(config).save_pretrained(out)
    return out


def test_real_scorer_probabilities(tiny_model_dir: Path) -> None:
    scorer = llm.transformers_scorer(_cfg(model=str(tiny_model_dir), batch_size=4))
    texts = [it.text for it in _items(5)]
    probs = scorer(texts)
    assert probs.shape == (5, 3)
    assert np.isfinite(probs).all() and (probs > 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_real_scorer_batching_matches_single(tiny_model_dir: Path) -> None:
    """Left padding must not change any item's probabilities."""
    scorer = llm.transformers_scorer(_cfg(model=str(tiny_model_dir)))
    texts = ["short", "a much longer complaint about a credit card fee and nobody called back"] * 2
    batched = scorer(texts)
    single = np.vstack([scorer([t]) for t in texts])
    np.testing.assert_allclose(batched, single, atol=1e-5)


def test_real_scorer_is_deterministic_and_truncates(tiny_model_dir: Path) -> None:
    cfg = _cfg(model=str(tiny_model_dir), max_input_tokens=40)
    scorer = llm.transformers_scorer(cfg)
    long_text = "I was charged a fee " * 200
    a, b = scorer([long_text]), scorer([long_text])
    np.testing.assert_array_equal(a, b)
    assert scorer.last_truncated == 1  # type: ignore[attr-defined]


def test_real_labeling_end_to_end(tiny_model_dir: Path, tmp_path: Path) -> None:
    cfg = _cfg(model=str(tiny_model_dir))
    res = llm.label_items(
        _items(4),
        cfg,
        cache=DiskCache(tmp_path),
        scorer=llm.transformers_scorer(cfg),
        dry_run=False,
    )
    assert res.predictions is not None and res.n_failed == 0
    nps.validate_predictions(res.predictions)


# --------------------------------------------------------------------------- stronger model support


def test_batches_are_sorted_by_length(tmp_path: Path) -> None:
    """Similar lengths share a batch (less padding); results still map to their items."""
    texts = ["x" * n for n in (50, 5, 40, 10, 30, 20)]
    items = [llm.LLMItem(item_id=str(i), source="cfpb", text=t) for i, t in enumerate(texts)]
    seen: list[list[int]] = []

    def scorer(batch: list[str]) -> np.ndarray:
        seen.append([len(t) for t in batch])
        # Encode the text length in the probabilities so the mapping can be checked.
        p = np.array([[len(t) / 100, 0.0, 1 - len(t) / 100] for t in batch])
        return p

    res = llm.label_items(
        items, _cfg(batch_size=2), cache=DiskCache(tmp_path), scorer=scorer, dry_run=False
    )
    assert seen == [[5, 10], [20, 30], [40, 50]]
    assert res.class_probs is not None
    np.testing.assert_allclose(res.class_probs[:, 0], [len(t) / 100 for t in texts])


def test_weights_dtype_validation() -> None:
    with pytest.raises(ValueError, match="weights_dtype"):
        _cfg(weights_dtype="int4")
    with pytest.raises(ValueError, match="dtype"):
        _cfg(dtype="float8")
    assert _cfg(weights_dtype="bfloat16", dtype="float32").weights_dtype == "bfloat16"


def test_render_prompt_disables_thinking() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    fast.chat_template = (
        "{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}[assistant]"
        "{% if enable_thinking is defined and enable_thinking is false %}<think></think>"
        "{% else %}<think>{% endif %}{% endif %}"
    )
    out = llm.render_prompt(fast, "hello")
    assert out.endswith("[assistant]<think></think>")
    assert out.startswith(f"[system]{llm.SYSTEM_PROMPT}[user]hello")


def test_upcast_linears_keeps_weights_and_computes_in_float32() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Embedding(10, 4), torch.nn.Linear(4, 3), torch.nn.LayerNorm(3)
    ).to(torch.bfloat16)
    # Reference: a float32 copy holding exactly the same (bf16-rounded) values.
    ref = torch.nn.Sequential(
        torch.nn.Embedding(10, 4), torch.nn.Linear(4, 3), torch.nn.LayerNorm(3)
    )
    ref.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = ref(ids)
    n = llm.upcast_linears(model, compute_dtype=torch.float32)
    assert n == 1
    assert model[1].weight.dtype == torch.bfloat16  # stored compactly
    assert model[2].weight.dtype == torch.float32  # everything else in the compute dtype
    with torch.no_grad():
        out = model(ids)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def tiny_hybrid_dir(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random Qwen3.5-style hybrid model (linear + full attention layers)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg_cls = getattr(transformers, "Qwen3_5TextConfig", None)
    model_cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if cfg_cls is None or model_cls is None:
        pytest.skip("transformers has no Qwen3.5 support")
    out = tmp_path_factory.mktemp("tiny_hybrid")
    tok = transformers.AutoTokenizer.from_pretrained(tiny_model_dir)
    tok.save_pretrained(out)
    torch.manual_seed(0)
    config = cfg_cls(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, full_attention_interval=2,
        max_position_embeddings=256, pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )  # fmt: skip
    model_cls(config).to(torch.bfloat16).save_pretrained(out)
    return out


def test_hybrid_scorer_batching_matches_single(tiny_hybrid_dir: Path) -> None:
    """Left padding must not leak into the recurrent (linear-attention) state."""
    cfg = _cfg(model=str(tiny_hybrid_dir), dtype="float32", weights_dtype="bfloat16")
    scorer = llm.transformers_scorer(cfg)
    texts = ["short", "a much longer complaint about a credit card fee and nobody called back"] * 2
    batched = scorer(texts)
    single = np.vstack([scorer([t]) for t in texts])
    np.testing.assert_allclose(batched, single, atol=1e-5)
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, atol=1e-9)


def test_hybrid_upcast_matches_float32_of_stored_weights(tiny_hybrid_dir: Path) -> None:
    """bf16 storage + float32 compute equals a float32 model holding the same bf16 weights."""
    texts = ["short", "a much longer complaint about a credit card fee and nobody called back"]
    upcast = llm.transformers_scorer(
        _cfg(model=str(tiny_hybrid_dir), dtype="float32", weights_dtype="bfloat16")
    )(texts)
    full = llm.transformers_scorer(_cfg(model=str(tiny_hybrid_dir), dtype="float32"))(texts)
    np.testing.assert_allclose(upcast, full, atol=1e-6)


# --------------------------------------------------------------------------- per-source prompts


def test_cfpb_cache_keys_are_unchanged_by_per_source_prompts() -> None:
    """The cached CFPB labels stay valid: their prompt version is the original one."""
    cfg = _cfg()
    assert cfg.prompt_version_for("cfpb") == cfg.prompt_version
    assert cfg.prompt_version_for("synthetic") == cfg.prompt_version
    assert cfg.prompt_version_for("amazon_reviews") != cfg.prompt_version
    assert llm.SYSTEM_PROMPTS["cfpb"] == llm.SYSTEM_PROMPT


def test_review_items_render_title_and_text_only() -> None:
    it = llm.render_review_item(review_id="r1", title=" Great card ", text="Worked fine. ")
    assert it.source == "amazon_reviews"
    assert it.text == "Review title: Great card\n\nReview:\nWorked fine."
    assert "complaint" not in llm.SYSTEM_PROMPTS["amazon_reviews"].lower()


def test_scorer_prompt_must_match_item_source(tmp_path: Path) -> None:
    scorer = StubScorer()
    scorer.system_prompt = llm.SYSTEM_PROMPTS["cfpb"]  # type: ignore[attr-defined]
    items = [llm.render_review_item(review_id="r", title="t", text="x")]
    with pytest.raises(ValueError, match="system prompt"):
        llm.label_items(items, _cfg(), cache=DiskCache(tmp_path), scorer=scorer, dry_run=False)
    assert scorer.n_scored == 0


def test_mixed_sources_with_different_prompts_are_refused(tmp_path: Path) -> None:
    items = [*_items(1), llm.render_review_item(review_id="r", title="t", text="x")]
    with pytest.raises(ValueError, match="one system prompt"):
        llm.label_items(
            items, _cfg(), cache=DiskCache(tmp_path), scorer=StubScorer(), dry_run=False
        )


def test_review_keys_differ_from_cfpb_keys_for_the_same_text(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cfpb_item = llm.LLMItem(item_id="1", source="cfpb", text="same text")
    review_item = llm.LLMItem(item_id="1", source="amazon_reviews", text="same text")
    llm.label_items([cfpb_item], _cfg(), cache=cache, scorer=StubScorer(), dry_run=False)
    plan = llm.plan_run([review_item], _cfg(), cache)
    assert plan.n_to_score == 1


def test_render_prompt_uses_the_given_system_prompt() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    fast.chat_template = (
        "{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}[assistant]{% endif %}"
    )
    out = llm.render_prompt(fast, "hello", llm.SYSTEM_PROMPTS["amazon_reviews"])
    assert out.startswith(f"[system]{llm.SYSTEM_PROMPTS['amazon_reviews']}[user]hello")
