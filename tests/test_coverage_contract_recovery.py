"""Completed coverage contract recovery; real receipts, joins and budget."""

import asyncio
import copy
import json

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError
from judges.generic.shared_semantics import prepare_topic
from tests.test_partial_output_recovery import Provider, ledger_rows
from tests.test_shared_track_judge import sample
from tests.test_single_checklist import item


def topic():
    answer = sample()
    answer.pop("links")
    answer["pairs"][0]["label"] = "supported"
    return dict(pool=dict(question=dict(title="State the color."),
                         evidence=[dict(id="e", text="Blue.")]),
                checklist=dict(items=[item("criterion-1")], gaps=[]),
                answers=[dict(run_id="anonymous-A", evidence=answer)])


class CoverageProvider(Provider):
    async def generate(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        packet = json.loads(request.messages[1]["content"])
        entry = dict(status="covered", claim_ids=["C1"], reason="Expressed.")
        if outcome == "foreign":
            entry["claim_ids"] = ["C99"]
        elif outcome == "duplicate":
            entry["claim_ids"] *= 2
        elif outcome == "missing_claim":
            entry["status"] = "missing"
        elif outcome == "partial_empty":
            entry.update(status="partial", claim_ids=[])
        elif outcome == "omission":
            entry["claim_ids"] = []
        elif outcome == "partial":
            entry["status"] = "partial"
        present = outcome != "presence"
        if request.request_id == "checklist-coverage-keyed-contract-v1":
            value = dict(answer_present=present, items={"I1": entry})
            if outcome == "missing_key":
                value["items"] = {}
            elif outcome == "extra_key":
                value["items"]["I2"] = copy.deepcopy(entry)
            text = json.dumps(value)
            if outcome == "duplicate_key":
                text = text.replace('"I1":', '"I1": ' + json.dumps(entry) + ', "I1":', 1)
            elif outcome == "nested_duplicate":
                text = text.replace('"reason":', '"reason": "SYNTHETIC_PRIVATE_SENTINEL", "reason":', 1)
        else:
            value = dict(answer_present=present,
                         items=[dict(id="foreign-item" if outcome == "foreign_item" else
                                     packet["items"][0]["id"], **entry)])
            text = json.dumps(value)
        if outcome == "invalid":
            text = "not-json"
        return MinimaLlmResponse(request.request_id, text, raw={
            "choices": [{"finish_reason": "length" if outcome == "length" else
                        "content_filter" if outcome == "refusal" else "stop"}],
            "usage": {} if self.cost is None else {"cost": self.cost},
        })


def session(tmp_path, outcomes, *, endpoint="https://openrouter.ai/api/v1", budget=10, cost=.02):
    provider = CoverageProvider(tmp_path / "cache", outcomes, endpoint=endpoint, cost=cost)
    return CompletionSession(None, tmp_path / "budget.sqlite3", budget,
                             factory=lambda _cfg: provider), provider


def prepare(backend, value=None, *, enabled=True, **kwargs):
    return asyncio.run(prepare_topic(topic() if value is None else value,
        backend, None, coverage_mode="separate", repair_coverage_contract=enabled,
        **kwargs))


@pytest.mark.parametrize("bad", ["foreign", "duplicate", "missing_claim", "partial_empty"])
def test_synthetic_reproducer_reaches_the_original_claim_selection_guard(tmp_path, bad):
    live, provider = session(tmp_path, [bad])
    with pytest.raises(JudgeError, match="Invalid coverage claim IDs"):
        asyncio.run(prepare_topic(topic(), live, None, coverage_mode="separate"))
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


@pytest.mark.parametrize("bad", ["foreign", "duplicate", "missing_claim", "partial_empty"])
def test_completed_bad_claim_selections_get_one_keyed_repair_and_exact_replay(tmp_path, bad):
    original = topic()
    saved = copy.deepcopy(original)
    live, provider = session(tmp_path, [bad, "stop"])
    result = prepare(live, original)
    assert result["answers"][0]["checklist_coverage"] == dict(answer_present=True,
        items=[dict(id="criterion-1", status="covered", claim_ids=["C1"], reason="Expressed.")])
    assert result["answers"][0]["evidence"]["links"][0]["status"] == "supported"
    assert original == saved
    assert [r.request_id for r in provider.calls] == [
        "checklist-coverage-v5", "checklist-coverage-keyed-contract-v1"]
    first, repair = [json.loads(r.messages[1]["content"]) for r in provider.calls]
    assert first["answer"] == repair["answer"]
    assert first["question"] == repair["question"]
    assert first["items"][0]["id"] == "criterion-1"
    assert repair["items"][0]["id"] == "I1"
    assert {k:v for k,v in first["items"][0].items() if k != "id"} == {
        k:v for k,v in repair["items"][0].items() if k != "id"}
    assert len(list(live.directory.glob("*.json"))) == 2
    before = ledger_rows(tmp_path)
    old_receipts = {p.name:p.read_bytes() for p in live.directory.glob("*.json")}
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline, saved) == result
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before
    assert {p.name:p.read_bytes() for p in live.directory.glob("*.json")} == old_receipts


@pytest.mark.parametrize("bad", ["foreign", "duplicate", "missing_claim", "partial_empty"])
def test_default_disabled_keeps_invalid_contract_fail_closed(tmp_path, bad):
    live, provider = session(tmp_path, [bad, "stop"])
    with pytest.raises(JudgeError):
        prepare(live, enabled=False)
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_enabling_repair_does_not_change_valid_original_request_or_score(tmp_path):
    first, p1 = session(tmp_path / "old", ["stop"])
    original = prepare(first, enabled=False)
    second, p2 = session(tmp_path / "new", ["stop"])
    assert prepare(second) == original
    assert len(p1.calls) == len(p2.calls) == 1
    assert p1.calls[0] == p2.calls[0]


@pytest.mark.parametrize("bad", ["foreign", "duplicate", "missing_claim", "partial_empty",
    "missing_key", "extra_key", "duplicate_key", "nested_duplicate", "presence", "length", "refusal", "invalid",
    RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")])
def test_failed_repair_is_terminal_and_saved_across_resumes(tmp_path, bad):
    live, provider = session(tmp_path, ["foreign", bad, "stop"])
    with pytest.raises(JudgeError) as stopped:
        prepare(live)
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in str(stopped.value)
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    with pytest.raises(JudgeError):
        prepare(offline)
    assert unused.calls == [] and ledger_rows(tmp_path) == before


@pytest.mark.parametrize("bad", ["length", "refusal", "invalid", RuntimeError("synthetic transport")])
def test_unusable_initial_response_never_triggers_contract_repair(tmp_path, bad):
    live, provider = session(tmp_path, [bad, "stop"])
    with pytest.raises(JudgeError):
        prepare(live)
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_covered_by_omission_remains_valid_but_not_grounded(tmp_path):
    live, provider = session(tmp_path, ["duplicate", "omission"])
    result = prepare(live)
    coverage = result["answers"][0]["checklist_coverage"]["items"][0]
    link = result["answers"][0]["evidence"]["links"][0]
    assert coverage["status"] == "covered" and coverage["claim_ids"] == []
    assert link["status"] == "not_established" and link["pair_ids"] == []


def test_unknown_original_charge_is_preserved_when_repair_settles(tmp_path):
    live, provider = session(tmp_path, ["foreign", "stop"], cost=None)
    prepare(live)
    rows = ledger_rows(tmp_path)
    assert len(rows) == 2 and all(row[1] is None for row in rows)
    assert live.summary()["committed_usd"] == pytest.approx(sum(row[0] for row in rows))


def test_existing_short_id_repair_is_reused_before_new_contract(tmp_path):
    live, provider = session(tmp_path, ["foreign_item", "foreign", "stop"])
    result = prepare(live)
    assert [r.request_id for r in provider.calls] == ["checklist-coverage-v5",
        "checklist-coverage-short-ids-v4", "checklist-coverage-keyed-contract-v1"]
    assert provider.calls[1] != provider.calls[2]
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline) == result and not unused.calls
    assert ledger_rows(tmp_path) == before


def test_successful_old_short_id_repair_never_buys_new_contract(tmp_path):
    live, provider = session(tmp_path, ["foreign_item", "stop"])
    result = prepare(live)
    assert len(provider.calls) == 2
    offline, unused = session(tmp_path, [], endpoint="EMPTY")
    assert prepare(offline, enabled=False) == result and not unused.calls


def test_partial_coverage_keeps_half_grounding_credit(tmp_path):
    live, provider = session(tmp_path, ["duplicate", "partial"])
    result = prepare(live)
    assert result["answers"][0]["evidence"]["links"][0]["status"] == "partially_supported"


def test_dynamic_schema_is_shared_deterministic_and_handles_zero_claims():
    from judges.generic.coverage_completion import keyed_coverage, restore_coverage
    from judges.generic.coverage_join import coverage_packet
    from pydantic import ValidationError
    packet = coverage_packet({}, [item("original"), item("another")], topic()["answers"][0]["evidence"])
    aliased, schema, mapping = keyed_coverage(packet)
    value = schema.model_validate_json(json.dumps(dict(answer_present=True, items={
        "I1": dict(status="covered", claim_ids=["C1"], reason="Expressed."),
        "I2": dict(status="partial", claim_ids=["C2"], reason="Some."),
    })))
    restored = restore_coverage(value, mapping)
    assert [i.id for i in restored.items] == ["original", "another"]
    assert restored.items[1].claim_ids == ["C2"]
    assert schema.model_json_schema() == keyed_coverage(packet)[1].model_json_schema()
    assert schema.model_json_schema()["$defs"]["CoverageClaimId"]["enum"] == ["C1", "C2"]
    packet["answer"]["claims"] = []
    _, empty, _ = keyed_coverage(packet)
    assert "CoverageClaimId" not in empty.model_json_schema()["$defs"]
    payload = dict(answer_present=False, items={"I1": dict(status="missing", claim_ids=[], reason="Empty."),
                                              "I2": dict(status="covered", claim_ids=[], reason="Omitted.")})
    empty.model_validate(payload)
    payload["items"]["I1"]["claim_ids"] = ["invented"]
    with pytest.raises(ValidationError):
        empty.model_validate(payload)


def test_enlarged_schema_context_guard_refuses_before_purchase(tmp_path):
    from judges.generic.coverage_join import coverage_packet, COVERAGE_PROMPT
    from judges.generic.document_pipeline import token_count
    packet = coverage_packet(topic()["pool"]["question"], topic()["checklist"]["items"], topic()["answers"][0]["evidence"])
    messages = [dict(role="system", content=COVERAGE_PROMPT),
                dict(role="user", content=json.dumps(packet, ensure_ascii=False, sort_keys=True))]
    limit = token_count(json.dumps(messages))
    live, provider = session(tmp_path, ["foreign", "stop"])
    with pytest.raises(JudgeError, match="Input exceeds context"):
        prepare(live, max_input_tokens=limit)
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_full_response_format_wrapper_counts_before_purchase(tmp_path):
    from judges.generic.coverage_completion import keyed_coverage, FORMAT
    from judges.generic.coverage_join import coverage_packet, COVERAGE_PROMPT
    from judges.generic.document_pipeline import token_count
    packet = coverage_packet(topic()['pool']['question'],topic()['checklist']['items'],topic()['answers'][0]['evidence'])
    aliased, schema, _ = keyed_coverage(packet)
    messages = [dict(role='system',content=COVERAGE_PROMPT+FORMAT),
                dict(role='user',content=json.dumps(aliased,ensure_ascii=False,sort_keys=True))]
    undercount = token_count(json.dumps(dict(messages=messages,schema=schema.model_json_schema())))
    extra = dict(reasoning=dict(effort='high'),response_format=dict(type='json_schema',
        json_schema=dict(name=schema.__name__,strict=True,schema=schema.model_json_schema())))
    assert token_count(json.dumps(dict(messages=messages,extra=extra))) > undercount
    live, provider = session(tmp_path,['foreign','stop'])
    with pytest.raises(JudgeError,match='Input exceeds context'):
        prepare(live,max_input_tokens=undercount)
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1


def test_budget_denial_never_purchases_correction(tmp_path):
    from judges.generic.models import RunStopped
    live, provider = session(tmp_path, ["foreign", "stop"], budget=.01, cost=.02)
    # Use a settled cache from another owner with a larger allowance.
    seed, p = session(tmp_path, ["foreign"])
    with pytest.raises(JudgeError):
        prepare(seed, enabled=False)
    with pytest.raises(RunStopped) as stopped:
        prepare(live)
    assert stopped.value.code == "budget_exhausted"
    assert provider.calls == [] and len(ledger_rows(tmp_path)) == 1


def test_unified_runner_wires_and_records_opt_in(tmp_path, monkeypatch):
    from judges.generic import unified
    from tests.test_unified_judge import SemanticBackend
    from tests.report_fixtures import report
    from autojudge_base import Request
    from types import SimpleNamespace
    original = unified.prepare_topic
    observed = []
    async def capture(*args, **kwargs):
        observed.append(kwargs['repair_coverage_contract'])
        return await original(*args, **kwargs)
    monkeypatch.setattr(unified, 'prepare_topic', capture)
    backend = SemanticBackend(tmp_path/'cache')
    reports = [report([dict(text='Blue.',citations={'d':100})],{'d':dict(id='d',text='Blue.')})]
    unified.UnifiedJudge(backend_factory=lambda _:backend).judge(reports,
        [Request(request_id='t',title='What color?',word_limit=20)],SimpleNamespace(),
        method='evidence',outdir=tmp_path/'out',repair_coverage_contract=True)
    assert observed == [True]
    saved = json.loads((tmp_path/'out/unified.stages.json').read_text())
    assert saved['repair_coverage_contract'] is True


@pytest.mark.parametrize('value', ['true', 1, None])
def test_opt_in_requires_a_real_boolean_before_purchase(tmp_path, value):
    live, provider = session(tmp_path, [])
    with pytest.raises(JudgeError, match='must be boolean'):
        prepare(live, enabled=value)
    assert provider.calls == []
