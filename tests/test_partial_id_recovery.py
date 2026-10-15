"""Completed ID-routing repair through the real cache and budget wrappers."""

import asyncio
import copy
import json

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError, RunStopped
from judges.generic.shared_semantics import prepare_topic
from tests.test_partial_grounding_batches import (
    _aliased_payload, _request_token_count, _response, _topic,
)
from tests.test_partial_output_recovery import Provider, ledger_rows


class RoutingProvider(Provider):
    async def generate(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        packet = json.loads(request.messages[1]["content"])
        response = _response(packet)
        finish = "stop"
        if outcome == "missing":
            response["links"].pop()
        elif outcome == "duplicate":
            response["links"].append(copy.deepcopy(response["links"][0]))
        elif outcome == "foreign":
            response["links"][0]["item"] = "not-a-checklist-item"
        elif outcome == "wrong_pair":
            response["links"][1]["pair_ids"] = ["P1"]
        elif outcome == "swap":
            response["links"][0]["pair_ids"] = ["P2"]
            response["links"][1]["pair_ids"] = ["P1"]
        elif outcome == "outside_first":
            response["links"][0]["pair_ids"] = ["P3"]
        elif outcome == "outside_single":
            response["links"][0]["pair_ids"] = ["P2"]
        elif outcome == "unknown_pair":
            response["links"][0]["pair_ids"] = ["P99"]
        elif outcome == "duplicate_pairs":
            response["links"][0]["pair_ids"] *= 2
        elif outcome == "empty_pairs":
            response["links"][0]["pair_ids"] = []
        elif outcome == "select_last":
            response["links"][0]["pair_ids"] = response["links"][0]["pair_ids"][-1:]
        elif outcome == "supported":
            for link in response["links"]:
                link["status"] = "supported"
                link["missing_or_unsupported"] = ""
        elif outcome == "wrong_case":
            response["links"][0]["case"] = "other-answer"
        elif outcome == "length":
            finish = "length"
        text = "invalid" if outcome == "invalid" else json.dumps(response)
        usage = {} if self.cost is None else {"cost": self.cost}
        return MinimaLlmResponse(request.request_id, text, raw={
            "choices": [{"finish_reason": finish}], "usage": usage,
        })


def session(tmp_path, outcomes, *, endpoint="https://openrouter.ai/api/v1", budget=10, cost=0.02):
    provider = RoutingProvider(tmp_path / "cache", outcomes, endpoint=endpoint, cost=cost)
    return CompletionSession(None, tmp_path / "budget.sqlite3", budget,
                             factory=lambda _config: provider), provider


def prepare(backend, topic=None, **kwargs):
    return asyncio.run(prepare_topic(
        _topic() if topic is None else topic, backend, None,
        coverage_mode="separate", resolve_partial_grounding=True,
        partial_max_output_tokens=65536, **kwargs,
    ))


def short_id_topic():
    topic = _topic()
    record = topic["answers"][0]
    for number, (item, link, coverage) in enumerate(zip(
        topic["checklist"]["items"], record["evidence"]["links"],
        record["checklist_coverage"]["items"],
    ), 1):
        item["id"] = link["item"] = coverage["id"] = f"I{number}"
    return topic


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign"])
@pytest.mark.parametrize("short_ids", [False, True])
def test_completed_bad_ids_get_one_exact_repair_and_cache_only_replay(tmp_path, failure, short_ids):
    topic = short_id_topic() if short_ids else _topic()
    original = copy.deepcopy(topic)
    live, provider = session(tmp_path, [failure, "stop"])
    result = prepare(live, topic)
    assert [call.max_tokens for call in provider.calls] == [16384, 65536]
    first, second = [json.loads(call.messages[1]["content"]) for call in provider.calls]
    assert [item["id"] for item in second["items"]] == ["I1", "I2"]
    for old, new in zip(first["items"], second["items"]):
        assert {k: v for k, v in old.items() if k != "id"} == {
            k: v for k, v in new.items() if k != "id"
        }
    assert second["answer"] == first["answer"]
    assert second["excerpts"] == first["excerpts"]
    assert second["question"] == first["question"]
    assert provider.calls[0].messages[0] != provider.calls[1].messages[0]
    resolved = result["answers"][0]["resolved_links"]
    assert [link["item"] for link in resolved] == [
        item["id"] for item in original["checklist"]["items"]
    ]
    assert [link["pair_ids"] for link in resolved] == [
        [pair["id"]] for pair in original["answers"][0]["evidence"]["pairs"]
    ]
    assert topic == original
    assert len(list(live.directory.glob("*.json"))) == 2
    assert len(list(live.directory.glob("*.started"))) == 2
    before = ledger_rows(tmp_path)
    assert len(before) == 2
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline, topic) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "length"])
def test_failed_id_repair_is_saved_and_stops_without_another_attempt(tmp_path, failure):
    live, provider = session(tmp_path, ["missing", failure, "stop"])
    with pytest.raises(JudgeError):
        prepare(live)
    assert [call.max_tokens for call in provider.calls] == [16384, 65536]
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_known_length_then_bad_ids_is_bounded_to_three_distinct_receipts(tmp_path):
    live, provider = session(tmp_path, ["length", "missing", "stop"])
    result = prepare(live)
    assert [call.max_tokens for call in provider.calls] == [16384, 65536, 65536]
    assert provider.calls[0].messages == provider.calls[1].messages
    assert provider.calls[1].messages != provider.calls[2].messages
    assert len(list(live.directory.glob("*.json"))) == 3
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_already_short_item_ids_at_same_output_cap_still_have_distinct_cache_keys(tmp_path):
    live, provider = session(tmp_path, ["missing", "stop"])
    result = asyncio.run(prepare_topic(
        short_id_topic(), live, None, coverage_mode="separate",
        resolve_partial_grounding=True,
    ))
    assert [call.max_tokens for call in provider.calls] == [16384, 16384]
    assert provider.calls[0].messages[1] == provider.calls[1].messages[1]
    assert provider.calls[0].messages[0] != provider.calls[1].messages[0]
    assert len(list(live.directory.glob("*.json"))) == 2
    assert [link["item"] for link in result["answers"][0]["resolved_links"]] == ["I1", "I2"]


@pytest.mark.parametrize("failure", [RuntimeError("synthetic transport")])
def test_non_id_failures_do_not_trigger_item_repair(tmp_path, failure):
    live, provider = session(tmp_path, [failure, "stop"])
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == 1


def test_completed_wrong_case_uses_isolation_not_batch_id_repair(tmp_path):
    live, provider = session(tmp_path, ["wrong_case", "stop"])
    result = prepare(live)
    assert len(provider.calls) == 2
    assert provider.calls[1].request_id == "partial-grounding-isolated-item-v1"
    assert result["answers"][0]["resolved_links"][0]["case"] == "answer-A"


def test_restored_item_ids_cannot_borrow_another_items_pair(tmp_path):
    live, provider = session(tmp_path, ["missing", "wrong_pair", "unknown_pair"])
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == 3


def test_id_repair_must_fit_budget_before_provider(tmp_path):
    live, provider = session(tmp_path, ["missing", "stop"], budget=0.8)
    with pytest.raises(RunStopped) as stopped:
        prepare(live)
    assert stopped.value.code == "budget_exhausted"
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_id_repair_keeps_unknown_original_cost_reserved(tmp_path):
    live, provider = session(tmp_path, ["missing", "stop"], cost=None)
    prepare(live)
    rows = ledger_rows(tmp_path)
    assert len(provider.calls) == len(rows) == 2
    assert all(reported is None for _reserved, reported, _owner in rows)
    assert live.summary()["committed_usd"] == pytest.approx(sum(row[0] for row in rows))


def test_repair_prompt_at_input_boundary_fails_without_trimming_or_second_call(tmp_path):
    topic = short_id_topic()
    payload, _mapping = _aliased_payload(topic)
    live, provider = session(tmp_path, ["missing", "stop"])
    with pytest.raises(JudgeError, match="context budget"):
        prepare(live, topic, max_input_tokens=_request_token_count(payload))
    assert len(provider.calls) == 1
    assert len(list(live.directory.glob("*.json"))) == 1
