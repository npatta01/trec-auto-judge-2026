"""Known completed partial-contract correction, with real cache and accounting."""

import copy
import json

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError, RunStopped
from judges.generic.shared_semantics import _validated_partial_link, restore_pair_ids
from tests.test_partial_grounding_batches import _topic, _response
from tests.test_partial_id_recovery import RoutingProvider, prepare
from tests.test_partial_output_recovery import ledger_rows


def failing_location(error):
    trace = error.__traceback__
    while trace.tb_next:
        trace = trace.tb_next
    return trace.tb_frame.f_code.co_filename.rsplit("/", 1)[-1], trace.tb_lineno


@pytest.mark.parametrize("gap", ["mentioned", "pair_ids", "supported_portion"])
def test_three_positive_gaps_reach_same_frozen_guard(gap):
    answer = _topic()["answers"][0]["evidence"]
    link = dict(answer["links"][0], status="partially_supported", mentioned=True,
        pair_ids=[answer["pairs"][0]["id"]], supported_portion="Synthetic stated fact.")
    link[gap] = {"mentioned": False, "pair_ids": [], "supported_portion": ""}[gap]
    with pytest.raises(JudgeError) as stopped:
        _validated_partial_link(link, answer, link["item"], {link["item"]: set(link["pair_ids"])})
    assert failing_location(stopped.value) == ("shared_pipeline.py", 129)


def test_duplicate_pointer_is_a_different_frozen_guard():
    answer = _topic()["answers"][0]["evidence"]
    link = dict(answer["links"][0], status="partially_supported", mentioned=True,
        pair_ids=[answer["pairs"][0]["id"]] * 2, supported_portion="Synthetic fact.")
    with pytest.raises(JudgeError) as stopped:
        _validated_partial_link(link, answer, link["item"], {link["item"]: set(link["pair_ids"])})
    assert failing_location(stopped.value) == ("shared_pipeline.py", 122)
    with pytest.raises(JudgeError, match="Unknown request-local"):
        restore_pair_ids([dict(link, pair_ids=["P99"])], {"P1": answer["pairs"][0]["id"]})


class ContractProvider(RoutingProvider):
    def __init__(self, *args, gap="mentioned", **kwargs):
        super().__init__(*args, **kwargs)
        self.gap = gap

    async def generate(self, request):
        outcome = self.outcomes[0]
        result = await super().generate(request)
        if request.request_id == "partial-grounding-isolated-item-v1":
            body = json.loads(result.text)
            body["links"][0][self.gap] = {
                "mentioned": False, "pair_ids": [], "supported_portion": "",
            }[self.gap]
            return MinimaLlmResponse(result.request_id, json.dumps(body), raw=result.raw)
        if request.request_id != "partial-grounding-keyed-contract-v1" or outcome == "invalid":
            return result
        packet = json.loads(request.messages[1]["content"])
        original = _response(packet)["links"][0]
        ids = packet["items"][0]["pair_ids"]
        link = dict(status="partially_supported", primary_pair_id=ids[0],
            additional_contributions={pid: False for pid in ids},
            supported_portion=original["supported_portion"],
            missing_or_unsupported=original["missing_or_unsupported"])
        if outcome == "foreign_primary":
            link["primary_pair_id"] = "P99"
        elif outcome == "empty_portion":
            link["supported_portion"] = ""
        elif outcome == "full_with_gap":
            link["status"] = "supported"
        elif outcome == "full_clean":
            link["status"] = "supported"
            link["missing_or_unsupported"] = ""
        elif outcome == "select_all":
            link["primary_pair_id"] = ids[-1]
            link["additional_contributions"] = {pid: True for pid in ids}
        elif outcome == "negative":
            link["status"] = "uncertain"
            link.pop("primary_pair_id")
            link["supported_portion"] = ""
        return MinimaLlmResponse(result.request_id, json.dumps(dict(link=link)), raw=result.raw)


def session(tmp_path, outcomes, *, gap="mentioned", endpoint="https://openrouter.ai/api/v1", cost=0.02, budget=10):
    provider = ContractProvider(tmp_path / "cache", outcomes, gap=gap, endpoint=endpoint, cost=cost)
    return CompletionSession(None, tmp_path / "budget.sqlite3", budget,
        factory=lambda _config: provider), provider


def receipts(directory):
    return {p.name: p.read_bytes() for p in directory.glob("*.json")}


@pytest.mark.parametrize("gap", ["mentioned", "pair_ids", "supported_portion"])
def test_saved_positive_gap_corrects_once_and_replays_exactly(tmp_path, gap):
    old, old_provider = session(tmp_path, ["wrong_pair", "stop"], gap=gap)
    with pytest.raises(JudgeError):
        prepare(old)
    before = receipts(old.directory)
    live, provider = session(tmp_path, ["stop"], gap=gap)
    result = prepare(live, repair_partial_contract=True)
    assert [r.request_id for r in provider.calls] == ["partial-grounding-keyed-contract-v1"]
    assert all(receipts(old.directory)[name] == body for name, body in before.items())
    link = result["answers"][0]["resolved_links"][1]
    assert link["case"] == "answer-A" and link["item"] == "checklist-item-2"
    assert link["mentioned"] is True
    assert link["status"] == "partially_supported"
    assert link["pair_ids"] == ["original-pair-id-2-" + "x" * 28]
    assert link["supported_portion"] == "Synthetic fact 2 is stated accurately."
    rows, saved = ledger_rows(tmp_path), receipts(old.directory)
    offline, unused = session(tmp_path, [], gap=gap, endpoint="EMPTY")
    assert prepare(offline, repair_partial_contract=True) == result
    assert unused.calls == [] and ledger_rows(tmp_path) == rows
    assert receipts(old.directory) == saved


@pytest.mark.parametrize("bad", ["invalid", "length", "foreign_primary", "empty_portion", "full_with_gap", RuntimeError("synthetic transport")])
def test_invalid_correction_is_terminal_without_second_attempt(tmp_path, bad):
    live, provider = session(tmp_path, ["wrong_pair", "stop", bad, "stop"])
    with pytest.raises(JudgeError):
        prepare(live, repair_partial_contract=True)
    assert len(provider.calls) == 3
    rows, saved = ledger_rows(tmp_path), receipts(live.directory)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline, repair_partial_contract=True)
    assert unused.calls == [] and ledger_rows(tmp_path) == rows
    assert receipts(live.directory) == saved


@pytest.mark.parametrize("bad", [1, "true", None])
def test_setting_rejects_non_boolean_before_spending(tmp_path, bad):
    live, provider = session(tmp_path, [])
    with pytest.raises(JudgeError, match="boolean"):
        prepare(live, repair_partial_contract=bad)
    assert provider.calls == []


def test_uncertain_correction_is_not_promoted(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "stop", "negative"])
    result = prepare(live, repair_partial_contract=True)
    link = result["answers"][0]["resolved_links"][1]
    assert link["status"] == "uncertain" and link["pair_ids"] == []
    assert link["supported_portion"] == ""


def test_already_valid_requests_and_links_do_not_change(tmp_path):
    def ordinary(path):
        provider = RoutingProvider(path / "cache", ["wrong_pair", "stop"])
        return CompletionSession(None, path / "budget.sqlite3", 10,
            factory=lambda _: provider), provider
    old, old_provider = ordinary(tmp_path / "old")
    new, new_provider = ordinary(tmp_path / "new")
    assert prepare(new, repair_partial_contract=True) == prepare(old)
    assert [(r.messages, r.extra, r.max_tokens) for r in new_provider.calls] == [
        (r.messages, r.extra, r.max_tokens) for r in old_provider.calls]
    assert len(new_provider.calls) == 2


@pytest.mark.parametrize("failure", ["invalid", "length", RuntimeError("synthetic ambiguous transport")])
def test_isolated_schema_length_transport_failures_do_not_enter_correction(tmp_path, failure):
    provider = RoutingProvider(tmp_path / "cache", ["wrong_pair", failure, "stop"])
    live = CompletionSession(None, tmp_path / "budget.sqlite3", 10,
        factory=lambda _: provider)
    with pytest.raises(JudgeError):
        prepare(live, repair_partial_contract=True)
    assert len(provider.calls) == 2
    before = ledger_rows(tmp_path)
    offline = CompletionSession(None, tmp_path / "budget.sqlite3", 10,
        factory=lambda _: RoutingProvider(tmp_path / "cache", [], endpoint="EMPTY"))
    with pytest.raises(JudgeError):
        prepare(offline, repair_partial_contract=True)
    assert ledger_rows(tmp_path) == before


def test_full_correction_cannot_promote_partial_coverage(tmp_path):
    live, provider = session(tmp_path, ["wrong_pair", "stop", "full_clean"])
    result = prepare(live, repair_partial_contract=True)
    link = result["answers"][0]["resolved_links"][1]
    assert link["status"] == "partially_supported"
    assert link["missing_or_unsupported"] == "A required synthetic detail is unresolved for 2."


def test_primary_and_multiple_contributions_survive_both_exact_maps(tmp_path):
    topic = _topic(3)
    topic["answers"][0]["checklist_coverage"]["items"][1]["claim_ids"] = ["C2", "C3"]
    live, provider = session(tmp_path, ["wrong_pair", "stop", "select_all"])
    result = prepare(live, topic, repair_partial_contract=True)
    assert result["answers"][0]["resolved_links"][1]["pair_ids"] == [
        "original-pair-id-2-" + "x" * 28,
        "original-pair-id-3-" + "x" * 28,
    ]


def test_shared_budget_denies_correction_before_purchase(tmp_path):
    old, provider = session(tmp_path, ["wrong_pair", "stop"])
    with pytest.raises(JudgeError):
        prepare(old)
    rows = ledger_rows(tmp_path)
    live, unused = session(tmp_path, ["stop"], budget=0.05)
    with pytest.raises(RunStopped) as stopped:
        prepare(live, repair_partial_contract=True)
    assert stopped.value.code == "budget_exhausted"
    assert unused.calls == [] and ledger_rows(tmp_path) == rows


def test_keyed_schema_is_deterministic_and_rejects_duplicate_object_keys():
    from judges.generic.partial_completion import keyed_partial
    payload = dict(items=[dict(pair_ids=["P1", "P2"])],
        answer=dict(pairs=[dict(id="P1"), dict(id="P2")]))
    schema = keyed_partial(payload)
    assert schema.model_json_schema() == keyed_partial(copy.deepcopy(payload)).model_json_schema()
    raw = '{"link":{"status":"uncertain","additional_contributions":{"P1":false,"P1":true,"P2":false},"supported_portion":"","missing_or_unsupported":"Unknown."}}'
    with pytest.raises(ValueError):
        schema.model_validate_json(raw)


def test_unified_setting_is_forwarded_and_saved(tmp_path, monkeypatch):
    from autojudge_base import Request
    from types import SimpleNamespace
    from judges.generic import unified
    from tests.test_unified_judge import SemanticBackend
    from tests.report_fixtures import report
    original, observed = unified.prepare_topic, []
    async def capture(*args, **kwargs):
        observed.append(kwargs["repair_partial_contract"])
        return await original(*args, **kwargs)
    monkeypatch.setattr(unified, "prepare_topic", capture)
    backend = SemanticBackend(tmp_path / "cache")
    unified.UnifiedJudge(backend_factory=lambda _: backend).judge(
        [report([dict(text="Blue.", citations={"d": 100})], {"d": dict(id="d", text="Blue.")})],
        [Request(request_id="t", title="What color?", word_limit=20)], SimpleNamespace(),
        method="evidence", outdir=tmp_path / "out", repair_partial_contract=True)
    assert observed == [True]
    assert json.loads((tmp_path / "out/unified.stages.json").read_text())["repair_partial_contract"] is True


def test_correction_counts_full_request_and_schema_before_spending(tmp_path):
    from judges.generic.document_pipeline import token_count
    old, old_provider = session(tmp_path, ["wrong_pair", "stop"])
    with pytest.raises(JudgeError):
        prepare(old)
    limit = max(token_count(json.dumps(r.messages)) for r in old_provider.calls)
    before = ledger_rows(tmp_path)
    live, unused = session(tmp_path, ["stop"])
    with pytest.raises(JudgeError, match="context budget"):
        prepare(live, repair_partial_contract=True, max_input_tokens=limit)
    assert unused.calls == [] and ledger_rows(tmp_path) == before


def test_isolated_refusal_does_not_trigger_contract_call(tmp_path):
    class RefusalProvider(RoutingProvider):
        async def generate(self, request):
            result = await super().generate(request)
            if len(self.calls) == 2:
                raw = dict(result.raw, choices=[dict(finish_reason="content_filter")])
                result = MinimaLlmResponse(result.request_id, result.text, raw=raw)
            return result
    provider = RefusalProvider(tmp_path / "cache", ["wrong_pair", "stop", "stop"])
    live = CompletionSession(None, tmp_path / "budget.sqlite3", 10,
        factory=lambda _: provider)
    with pytest.raises(JudgeError):
        prepare(live, repair_partial_contract=True)
    assert len(provider.calls) == 2


def test_keyed_correction_refusal_is_terminal_and_replayed_without_purchase(tmp_path):
    class RefusalProvider(ContractProvider):
        async def generate(self, request):
            result = await super().generate(request)
            if request.request_id == "partial-grounding-keyed-contract-v1":
                result = MinimaLlmResponse(result.request_id, result.text,
                    raw=dict(result.raw, choices=[dict(finish_reason="content_filter")]))
            return result
    provider = RefusalProvider(tmp_path / "cache", ["wrong_pair", "stop", "stop", "stop"])
    live = CompletionSession(None, tmp_path / "budget.sqlite3", 10,
        factory=lambda _: provider)
    with pytest.raises(JudgeError):
        prepare(live, repair_partial_contract=True)
    assert len(provider.calls) == 3
    rows, saved = ledger_rows(tmp_path), receipts(live.directory)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline, repair_partial_contract=True)
    assert unused.calls == [] and ledger_rows(tmp_path) == rows
    assert receipts(live.directory) == saved
