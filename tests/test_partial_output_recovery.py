"""Synthetic output-limit recovery through the actual cache and budget wrappers."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError, RunStopped
from judges.generic.shared_semantics import prepare_topic
from tests.test_partial_grounding_batches import _response, _topic


class Provider:
    def __init__(self, cache, outcomes, *, endpoint="https://openrouter.ai/api/v1", cost=0.02):
        self.cfg = SimpleNamespace(
            model="synthetic-model", cache_dir=str(cache), force_refresh=False,
            base_url=endpoint, max_attempts=1,
        )
        self.outcomes, self.cost, self.calls = list(outcomes), cost, []

    async def generate(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        packet = json.loads(request.messages[1]["content"])
        text = "invalid" if outcome == "invalid" else json.dumps(_response(packet))
        usage = {} if self.cost is None else {"cost": self.cost}
        return MinimaLlmResponse(request.request_id, text, raw={
            "choices": [{"finish_reason": "stop" if outcome == "invalid" else outcome}],
            "usage": usage,
        })

    async def aclose(self):
        pass


def prepare(session, *, ceiling=65536):
    return asyncio.run(prepare_topic(
        _topic(), session, None, coverage_mode="separate",
        resolve_partial_grounding=True, partial_max_output_tokens=ceiling,
    ))


def session(tmp_path, outcomes, *, endpoint="https://openrouter.ai/api/v1", budget=10, cost=0.02):
    provider = Provider(tmp_path / "cache", outcomes, endpoint=endpoint, cost=cost)
    return CompletionSession(None, tmp_path / "budget.sqlite3", budget,
                             factory=lambda _config: provider), provider


def ledger_rows(tmp_path):
    with sqlite3.connect(tmp_path / "budget.sqlite3") as db:
        return db.execute("SELECT reserved,reported,run_id FROM calls ORDER BY id").fetchall()


def test_known_output_limit_recovers_once_and_replays_without_spend(tmp_path):
    live, provider = session(tmp_path, ["length", "stop"])
    result = prepare(live)
    assert [r.max_tokens for r in provider.calls] == [16384, 65536]
    assert provider.calls[0].messages == provider.calls[1].messages
    assert provider.calls[0].extra == provider.calls[1].extra
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    assert len(before) == 2
    assert before[1][0] - before[0][0] == pytest.approx(2 * (65536 - 16384) * 10 / 1_000_000)
    assert before[1][0] >= 2 * 65536 * 10 / 1_000_000

    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_second_output_limit_stops_without_third_attempt(tmp_path):
    live, provider = session(tmp_path, ["length", "length"])
    with pytest.raises(JudgeError, match="output token limit"):
        prepare(live)
    assert [r.max_tokens for r in provider.calls] == [16384, 65536]
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError, match="output token limit"):
        prepare(offline)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("outcome", ["content_filter", RuntimeError("synthetic transport")])
def test_other_failures_do_not_trigger_larger_attempt(tmp_path, outcome):
    live, provider = session(tmp_path, [outcome])
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == 1
    assert provider.calls[0].max_tokens == 16384


def test_larger_request_must_fit_incremental_budget_before_network(tmp_path):
    live, provider = session(tmp_path, ["length", "stop"], budget=0.8)
    with pytest.raises(RunStopped) as stopped:
        prepare(live)
    assert stopped.value.code == "budget_exhausted"
    assert len(provider.calls) == 1
    assert len(ledger_rows(tmp_path)) == 1
    assert len(list(live.directory.glob("*.json"))) == 1
    assert len(list(live.directory.glob("*.started"))) == 1


def test_unknown_original_cost_remains_reserved_during_recovery(tmp_path):
    live, provider = session(tmp_path, ["length", "stop"], cost=None)
    prepare(live)
    rows = ledger_rows(tmp_path)
    assert len(provider.calls) == len(rows) == 2
    assert all(reported is None for _reserved, reported, _owner in rows)
    assert live.summary()["committed_usd"] == pytest.approx(sum(row[0] for row in rows))


@pytest.mark.parametrize("ceiling", [True, 16385, 128000, "65536"])
def test_invalid_output_configuration_fails_before_provider(tmp_path, ceiling):
    live, provider = session(tmp_path, ["stop"])
    with pytest.raises(JudgeError, match="output token"):
        prepare(live, ceiling=ceiling)
    assert provider.calls == []
