"""Completed contract failures recover; ambiguous calls never do."""

import copy
import json

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError
from tests.test_partial_id_recovery import RoutingProvider, prepare
from tests.test_partial_grounding_batches import _topic
from tests.test_partial_output_recovery import ledger_rows


class ContractProvider(RoutingProvider):
    async def generate(self, request):
        change = self.outcomes[0]
        response = await super().generate(request)
        if not isinstance(change, dict):
            return response
        data = json.loads(response.text)
        # Corrupt only the last link: the first valid sibling must survive.
        data["links"][-1].update(change)
        return MinimaLlmResponse(request.request_id, json.dumps(data), raw=response.raw)


def session(tmp_path, outcomes, *, endpoint="https://openrouter.ai/api/v1", cost=0.02):
    provider = ContractProvider(tmp_path / "cache", outcomes, endpoint=endpoint, cost=cost)
    return CompletionSession(None, tmp_path / "budget.sqlite3", 10,
                             factory=lambda _config: provider), provider


FAILURES = [
    {"mentioned": False},
    {"pair_ids": []},
    {"supported_portion": "   "},
    {"pair_ids": ["P99"]},
    {"pair_ids": ["P2", "P2"]},
    {"pair_ids": ["P1"]},
    {"case": "another-answer"},
    {"status": "supported", "missing_or_unsupported": "Missing an essential part."},
]


@pytest.mark.parametrize("change", FAILURES)
def test_completed_contract_failure_repairs_only_invalid_item_and_replays(tmp_path, change):
    topic = _topic()
    original = copy.deepcopy(topic)
    live, provider = session(tmp_path, [change, "stop"])
    result = prepare(live, topic)
    assert len(provider.calls) == 2
    packet = json.loads(provider.calls[1].messages[1]["content"])
    assert [i["id"] for i in packet["items"]] == ["checklist-item-2"]
    assert len(packet["answer"]["pairs"]) == 1
    assert packet["answer"]["pairs"][0]["text"] == "Synthetic factual statement 2."
    assert len(packet["excerpts"]) == 2
    links = result["answers"][0]["resolved_links"]
    assert links[0]["supported_portion"] == "Synthetic fact 1 is stated accurately."
    assert [link["pair_ids"] for link in links] == [[p["id"]] for p in original["answers"][0]["evidence"]["pairs"]]
    assert topic == original
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline, topic) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("change", FAILURES[:5] + [FAILURES[-1]])
def test_invalid_repair_stops_and_replay_never_resends(tmp_path, change):
    # P1 is the only local pair in a repair packet.
    if change == {"pair_ids": ["P2", "P2"]}:
        change = {"pair_ids": ["P1", "P1"]}
    live, provider = session(tmp_path, [{"mentioned": False}, change, "stop"])
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == 2
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("outcome", ["invalid", {"mentioned": "yes"}, {"status": "unknown"}])
def test_completed_bad_schema_gets_one_isolated_call_per_item(tmp_path, outcome):
    live, provider = session(tmp_path, [outcome, "stop", "stop"])
    result = prepare(live)
    assert len(provider.calls) == 3
    assert all(len(json.loads(c.messages[1]["content"])["items"]) == 1 for c in provider.calls[1:])
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("where", ["original", "repair"])
def test_transport_failure_is_never_repaired_or_resent(tmp_path, where):
    outcomes = [RuntimeError("synthetic transport")]
    if where == "repair":
        outcomes.insert(0, {"mentioned": False})
    live, provider = session(tmp_path, outcomes)
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == (1 if where == "original" else 2)
    before = ledger_rows(tmp_path)
    assert before[-1][1] is None
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_new_repair_keeps_unknown_cost_reservations(tmp_path):
    live, provider = session(tmp_path, [{"mentioned": False}, "stop"], cost=None)
    prepare(live)
    rows = ledger_rows(tmp_path)
    assert len(rows) == len(provider.calls) == 2
    assert all(reported is None for _reserved, reported, _owner in rows)
    assert live.summary()["committed_usd"] == pytest.approx(sum(row[0] for row in rows))
