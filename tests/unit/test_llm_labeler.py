"""Tests for the LLM labeler. A fake client stands in for the API: no real calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from silent97.estimators import nps
from silent97.labelers import llm
from silent97.labelers.cache import DiskCache


def _items(n: int, source: str = "cfpb") -> list[llm.LLMItem]:
    return [
        llm.render_cfpb_item(
            complaint_id=str(i),
            narrative=f"I was charged a fee {i} times and nobody called back.",
            product="Credit card",
            issue="Billing disputes",
            state="CA",
        )
        if source == "cfpb"
        else llm.LLMItem(item_id=str(i), source=source, text=f"synthetic text {i}")
        for i in range(n)
    ]


def _cfg(**overrides: Any) -> llm.LLMConfig:
    base = {
        "model": "claude-opus-5-5",
        "max_items": 100,
        "max_spend_usd": 5.0,
        "mode": "sync",
        "price_input_per_mtok": 4.0,
        "price_output_per_mtok": 20.0,
    }
    return llm.LLMConfig(**{**base, **overrides})


def _message(text: str, *, stop: str = "end_turn", inp: int = 600, out: int = 120) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


GOOD = json.dumps({"detractor": 0.6, "passive": 0.3, "promoter": 0.1})


class FakeClient:
    """Records calls; answers every sync request with ``reply`` and batches in kind."""

    def __init__(self, reply: str = GOOD, **message_kw: Any) -> None:
        self.reply = reply
        self.message_kw = message_kw
        self.sync_calls: list[dict[str, Any]] = []
        self.batches_created: list[list[dict[str, Any]]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(
            create=self._create,
            batches=SimpleNamespace(
                create=self._batch_create,
                retrieve=self._batch_retrieve,
                results=self._batch_results,
            ),
        )

    def _create(self, **kwargs: Any) -> Any:
        self.sync_calls.append(kwargs)
        return _message(self.reply, **self.message_kw)

    def _batch_create(self, *, requests: list[dict[str, Any]]) -> Any:
        self.batches_created.append(requests)
        return SimpleNamespace(
            id=f"batch_{len(self.batches_created)}", processing_status="in_progress"
        )

    def _batch_retrieve(self, batch_id: str) -> Any:
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def _batch_results(self, batch_id: str) -> Any:
        requests = self.batches_created[int(batch_id.split("_")[1]) - 1]
        for r in requests:
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(
                    type="succeeded", message=_message(self.reply, **self.message_kw)
                ),
            )


# --------------------------------------------------------------------------- dry run and caps


def test_dry_run_makes_no_calls_and_prints_plan(tmp_path: Path) -> None:
    lines: list[str] = []
    res = llm.label_items(
        _items(20), _cfg(), cache=DiskCache(tmp_path), client=None, log=lines.append
    )
    assert res.predictions is None
    assert res.plan.n_items == 20 and res.plan.n_to_call == 20 and res.plan.n_cached == 0
    assert res.plan.projected_cost_usd > 0
    text = "\n".join(lines)
    assert "20" in text and "$" in text and "dry run" in text.lower()


def test_dry_run_is_default(tmp_path: Path) -> None:
    client = FakeClient()
    res = llm.label_items(_items(3), _cfg(), cache=DiskCache(tmp_path), client=client)
    assert res.predictions is None and client.sync_calls == []


def test_projection_formula(tmp_path: Path) -> None:
    items = _items(10)
    cfg = _cfg(chars_per_token=4.0, expected_output_tokens=100)
    plan = llm.plan_run(items, cfg, DiskCache(tmp_path))
    per_item_in = [(len(llm.SYSTEM_PROMPT) + len(it.text)) / 4.0 for it in items]
    expected = (sum(per_item_in) * 4.0 + 10 * 100 * 20.0) / 1e6
    assert plan.projected_cost_usd == pytest.approx(expected)
    batch = llm.plan_run(
        items,
        _cfg(chars_per_token=4.0, expected_output_tokens=100, mode="batch"),
        DiskCache(tmp_path),
    )
    assert batch.projected_cost_usd == pytest.approx(expected * 0.5)


def test_refuses_without_spend_cap(tmp_path: Path) -> None:
    with pytest.raises(llm.CapError, match="max_spend_usd"):
        llm.label_items(
            _items(3),
            _cfg(max_spend_usd=None),
            cache=DiskCache(tmp_path),
            client=FakeClient(),
            dry_run=False,
        )


def test_refuses_over_item_cap(tmp_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(llm.CapError, match="max_items"):
        llm.label_items(
            _items(11), _cfg(max_items=10), cache=DiskCache(tmp_path), client=client, dry_run=False
        )
    assert client.sync_calls == []


def test_refuses_over_projected_spend(tmp_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(llm.CapError, match="projected"):
        llm.label_items(
            _items(50),
            _cfg(max_spend_usd=0.01),
            cache=DiskCache(tmp_path),
            client=client,
            dry_run=False,
        )
    assert client.sync_calls == []


def test_stops_before_actual_spend_could_exceed_cap(tmp_path: Path) -> None:
    """Responses cost far more than projected. Before each call the run reserves the
    worst case (estimated input + max_tokens output), so the cap is never exceeded."""
    cache = DiskCache(tmp_path)
    client = FakeClient(out=1500)  # ~0.032 USD per item, ~6x the projection
    with pytest.raises(llm.SpendCapReached):
        llm.label_items(
            _items(10), _cfg(max_spend_usd=0.1), cache=cache, client=client, dry_run=False
        )
    n_done = len(client.sync_calls)
    assert 1 <= n_done < 10
    assert llm.plan_run(_items(10), _cfg(), cache).n_cached == n_done
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == n_done
    assert sum(e["cost_usd"] for e in ledger) <= 0.1


def test_rejects_disallowed_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        llm.label_items(_items(2, source="customer_db"), _cfg(), cache=DiskCache(tmp_path))


def test_synthetic_source_allowed(tmp_path: Path) -> None:
    res = llm.label_items(
        _items(2, source="synthetic"),
        _cfg(),
        cache=DiskCache(tmp_path),
        client=FakeClient(),
        dry_run=False,
    )
    assert res.predictions is not None


# --------------------------------------------------------------------------- calls and cache


def test_sync_run_labels_and_caches(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    client = FakeClient()
    res = llm.label_items(_items(5), _cfg(), cache=cache, client=client, dry_run=False)
    assert len(client.sync_calls) == 5
    assert res.predictions is not None
    nps.validate_predictions(res.predictions)
    np.testing.assert_allclose(res.predictions, [[0.1, 0.6]] * 5)
    np.testing.assert_allclose(res.class_probs, [[0.6, 0.3, 0.1]] * 5)
    assert res.spent_usd == pytest.approx(5 * (600 * 4 + 120 * 20) / 1e6)
    # Second run: every key is cached, so no call is made even with dry_run=False.
    client2 = FakeClient()
    res2 = llm.label_items(_items(5), _cfg(), cache=cache, client=client2, dry_run=False)
    assert client2.sync_calls == [] and res2.spent_usd == 0.0
    np.testing.assert_array_equal(res2.predictions, res.predictions)


def test_request_shape(tmp_path: Path) -> None:
    client = FakeClient()
    llm.label_items(
        _items(1), _cfg(effort="low"), cache=DiskCache(tmp_path), client=client, dry_run=False
    )
    call = client.sync_calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == llm.SYSTEM_PROMPT
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"] == llm.RESPONSE_SCHEMA
    content = call["messages"][0]["content"]
    assert "Credit card" in content and "Billing disputes" in content and "CA" in content
    assert "disputed" not in content.lower()  # the outcome is never shown


def test_cache_key_covers_model_prompt_and_input(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    llm.label_items(_items(2), _cfg(), cache=cache, client=FakeClient(), dry_run=False)
    assert llm.plan_run(_items(2), _cfg(), cache).n_to_call == 0
    assert llm.plan_run(_items(2), _cfg(model="claude-haiku-5-5"), cache).n_to_call == 2
    assert llm.plan_run(_items(2), _cfg(effort="medium"), cache).n_to_call == 2
    edited = _items(2)
    edited[0] = llm.LLMItem(item_id="0", source="cfpb", text=edited[0].text + " Edited.")
    assert llm.plan_run(edited, _cfg(), cache).n_to_call == 1


@pytest.mark.parametrize(
    "reply, stop",
    [
        ("not json", "end_turn"),
        (json.dumps({"detractor": 2.0, "passive": 0.1, "promoter": 0.1}), "end_turn"),
        (json.dumps({"detractor": -0.1, "passive": 0.6, "promoter": 0.5}), "end_turn"),
        (GOOD, "refusal"),
        (GOOD, "max_tokens"),
    ],
)
def test_failures_get_constant_prediction_and_are_cached(
    tmp_path: Path, reply: str, stop: str
) -> None:
    cache = DiskCache(tmp_path)
    client = FakeClient(reply=reply, stop=stop)
    res = llm.label_items(_items(2), _cfg(), cache=cache, client=client, dry_run=False)
    assert res.n_failed == 2
    np.testing.assert_allclose(res.predictions, [[1 / 3, 1 / 3]] * 2)
    # Failures are cached too: never re-called.
    client2 = FakeClient()
    llm.label_items(_items(2), _cfg(), cache=cache, client=client2, dry_run=False)
    assert client2.sync_calls == []


def test_probabilities_renormalized() -> None:
    probs, ok = llm.parse_probabilities(
        json.dumps({"detractor": 0.5, "passive": 0.26, "promoter": 0.24})
    )
    assert ok
    np.testing.assert_allclose(probs, np.array([0.5, 0.26, 0.24]) / 1.0)
    probs, ok = llm.parse_probabilities(
        json.dumps({"detractor": 0.55, "passive": 0.3, "promoter": 0.2})
    )
    assert ok and probs.sum() == pytest.approx(1.0)


def test_batch_mode(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    client = FakeClient()
    res = llm.label_items(
        _items(7),
        _cfg(mode="batch", batch_size=3),
        cache=cache,
        client=client,
        dry_run=False,
        sleep=lambda s: None,
    )
    assert [len(b) for b in client.batches_created] == [3, 3, 1]
    assert client.sync_calls == []
    assert res.n_called == 7
    np.testing.assert_allclose(res.predictions, [[0.1, 0.6]] * 7)
    assert res.spent_usd == pytest.approx(7 * (600 * 4 + 120 * 20) / 1e6 * 0.5)
    req = client.batches_created[0][0]
    assert len(req["custom_id"]) <= 64
    assert "fallbacks" not in req["params"]  # not accepted by the Batches API


def test_sync_mode_uses_server_side_fallback(tmp_path: Path) -> None:
    client = FakeClient()
    llm.label_items(_items(1), _cfg(), cache=DiskCache(tmp_path), client=client, dry_run=False)
    call = client.sync_calls[0]
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]


# --------------------------------------------------------------------------- credentials


def test_api_key_from_dotenv_and_explicit_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nANTHROPIC_API_KEY=sk-test-123\n")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://session-proxy.invalid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.load_api_key(env) == "sk-test-123"
    client = llm.make_client(_cfg(), env_file=env)
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com"


def test_missing_api_key_message_has_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.CredentialsError, match="ANTHROPIC_API_KEY"):
        llm.load_api_key(tmp_path / "missing.env")


def test_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "llm.yaml"
    path.write_text("model: claude-haiku-5-5\nmax_items: 10\nmax_spend_usd: null\nmode: batch\n")
    cfg = llm.load_config(path)
    assert cfg.model == "claude-haiku-5-5" and cfg.max_spend_usd is None and cfg.mode == "batch"
    path.write_text("modle: x\n")
    with pytest.raises(ValueError, match="unknown"):
        llm.load_config(path)


def test_repo_config_has_no_cap() -> None:
    """The committed config must not authorize spend: a human sets the cap."""
    cfg = llm.load_config(Path("configs/llm.yaml"))
    assert cfg.max_spend_usd is None
