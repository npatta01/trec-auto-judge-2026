"""A completed misroute is isolated once through the real cache/budget path."""

import copy
import json

import pytest

from judges.generic.models import JudgeError, RunStopped
from tests.test_partial_id_recovery import prepare, session
from tests.test_partial_output_recovery import ledger_rows
from tests.test_partial_grounding_batches import (
    _aliased_payload, _project_items, _request_token_count, _topic,
)


def test_only_misrouted_item_is_isolated_and_replays_without_spend(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "stop"])
    result = prepare(live)

    assert len(provider.calls) == 2
    first, isolated = [json.loads(r.messages[1]["content"]) for r in provider.calls]
    assert len(isolated["items"]) == 1
    item = isolated["items"][0]
    assert item["id"] == "checklist-item-2"
    assert item["pair_ids"] == ["P1"]
    assert [p["id"] for p in isolated["answer"]["pairs"]] == ["P1"]
    pair = isolated["answer"]["pairs"][0]
    assert pair["text"] == "Synthetic factual statement 2."
    assert set(isolated["excerpts"]) == set(pair["excerpt_ids"])
    assert list(isolated["excerpts"].values()) == [
        "Synthetic retained evidence for item 2. " * 5,
        "Second retained excerpt 2.",
    ]
    assert isolated["question"] == first["question"]
    assert {k: v for k, v in item.items() if k != "pair_ids"} == {
        k: v for k, v in first["items"][1].items() if k != "pair_ids"
    }
    assert provider.calls[0].messages[0] != provider.calls[1].messages[0]
    links = result["answers"][0]["resolved_links"]
    assert links[0]["supported_portion"] == "Synthetic fact 1 is stated accurately."
    assert links[1]["supported_portion"] == "Synthetic fact 2 is stated accurately."
    assert [link["pair_ids"] for link in links] == [
        ["original-pair-id-1-" + "x" * 28],
        ["original-pair-id-2-" + "x" * 28],
    ]
    assert len(list(live.directory.glob("*.json"))) == 2
    assert len(list(live.directory.glob("*.started"))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_completed_id_repair_with_misroute_has_one_isolated_recovery(tmp_path):
    live, provider = session(tmp_path, ["missing", "wrong_pair", "stop"])
    result = prepare(live)
    assert len(provider.calls) == 3
    assert [link["item"] for link in result["answers"][0]["resolved_links"]] == [
        "checklist-item-1", "checklist-item-2",
    ]
    assert len(json.loads(provider.calls[-1].messages[1]["content"])["items"]) == 1


@pytest.mark.parametrize("failure", [
    "missing", "duplicate", "length", "invalid", "unknown_pair",
    "duplicate_pairs", "empty_pairs",
])
def test_failed_isolation_is_terminal_and_cached(tmp_path, failure):
    live, provider = session(tmp_path, ["wrong_pair", failure, "stop"])
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


@pytest.mark.parametrize("failure", ["foreign", "wrong_case"])
def test_code_owns_case_and_item_in_one_item_context(tmp_path, failure):
    live, provider = session(tmp_path, ["wrong_pair", failure])
    result = prepare(live)
    assert result["answers"][0]["resolved_links"][1]["item"] == "checklist-item-2"
    assert result["answers"][0]["resolved_links"][1]["case"] == "answer-A"
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result
    assert unused.calls == []


def test_isolation_keeps_unknown_costs_reserved(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "stop"], cost=None)
    prepare(live)
    rows = ledger_rows(tmp_path)
    assert len(rows) == len(provider.calls) == 2
    assert all(reported is None for _reserved, reported, _owner in rows)
    assert live.summary()["committed_usd"] == pytest.approx(sum(row[0] for row in rows))


def test_isolation_checks_budget_before_provider(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "stop"], budget=0.8)
    with pytest.raises(RunStopped) as stopped:
        prepare(live)
    assert stopped.value.code == "budget_exhausted"
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_each_misroute_gets_one_isolated_call(tmp_path):
    live, provider = session(tmp_path, ["swap", "stop", "stop"])
    result = prepare(live)
    assert len(provider.calls) == 3
    assert [link["pair_ids"] for link in result["answers"][0]["resolved_links"]] == [
        ["original-pair-id-1-" + "x" * 28],
        ["original-pair-id-2-" + "x" * 28],
    ]
    assert all(len(json.loads(r.messages[1]["content"])["items"]) == 1
               for r in provider.calls[1:])


def test_overlapping_routes_do_not_trigger_recovery(tmp_path):
    topic = _topic()
    topic["answers"][0]["checklist_coverage"]["items"][1]["claim_ids"] = ["C1", "C2"]
    live, provider = session(tmp_path, ["wrong_pair"])
    result = prepare(live, topic)
    assert len(provider.calls) == 1
    assert result["answers"][0]["resolved_links"][1]["pair_ids"] == [
        "original-pair-id-1-" + "x" * 28,
    ]


def test_recovery_retains_all_permitted_evidence_but_only_credits_selected_pair(tmp_path):
    topic = _topic(3)
    topic["answers"][0]["checklist_coverage"]["items"][0]["claim_ids"] = ["C1", "C2"]
    live, provider = session(tmp_path, ["outside_first", "select_last"])
    result = prepare(live, topic)
    packet = json.loads(provider.calls[1].messages[1]["content"])
    assert len(packet["answer"]["pairs"]) == 2
    assert len(packet["excerpts"]) == 4
    assert result["answers"][0]["resolved_links"][0]["pair_ids"] == [
        "original-pair-id-2-" + "x" * 28,
    ]


def test_recovery_cannot_promote_partial_coverage_to_full_credit(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "supported"])
    result = prepare(live)
    link = result["answers"][0]["resolved_links"][1]
    assert link["status"] == "partially_supported"
    assert link["missing_or_unsupported"] == "A required synthetic detail is unresolved for 2."


def test_recovery_checks_actual_prompt_size_without_trimming(tmp_path):
    payload, _ = _aliased_payload(_topic())
    limit = max(_request_token_count(_project_items(payload, [item]))
                for item in payload["items"])
    live, provider = session(tmp_path, ["outside_single", "stop"])
    with pytest.raises(JudgeError, match="context budget"):
        prepare(live, max_input_tokens=limit)
    assert len(provider.calls) == 1
    assert len(list(live.directory.glob("*.json"))) == 1


def test_ambiguous_recovery_transport_is_never_resent(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", RuntimeError("synthetic transport")])
    with pytest.raises(JudgeError):
        prepare(live)
    before = ledger_rows(tmp_path)
    assert len(provider.calls) == len(before) == 2
    assert before[-1][1] is None
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_isolation_does_not_change_successful_request_bytes(tmp_path):
    from tests.test_partial_id_recovery import short_id_topic
    from tests.test_partial_grounding_batches import _aliased_payload

    topic = short_id_topic()
    original = copy.deepcopy(topic)
    payload, _ = _aliased_payload(topic)
    live, provider = session(tmp_path, ["stop"])
    prepare(live, topic)
    assert len(provider.calls) == 1
    assert provider.calls[0].messages[1]["content"] == json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
    )
    assert topic == original
