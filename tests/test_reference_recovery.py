"""Synthetic output failures through the real receipt and budget boundary."""

import asyncio
import copy
import hashlib
import json
import traceback

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError, RunStopped
from judges.generic.reference_answer import generate_answer, make_request, parse_answer
from tests.test_partial_output_recovery import Provider, ledger_rows
from tests.test_reference_answer import bundle


def packet():
    return dict(question=dict(title='Price and duration?'), claims=[],
                evidence=[dict(id='e1', text='Elm costs $2.')])


class ReferenceProvider(Provider):
    async def generate(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        value = bundle()
        payload = json.loads(request.messages[1]['content'])
        evidence_id = payload['evidence'][0]['id']
        for row in [*value['answer'], *value['rubric']]:
            row['evidence_ids'] = [evidence_id]
        finish = 'stop'
        if outcome == 'foreign':
            value['answer'][0]['evidence_ids'] = ['SECRET-CANARY']
        elif outcome == 'duplicate':
            value['answer'][0]['evidence_ids'] *= 2
        elif outcome == 'duplicate_item':
            value['rubric'] *= 2
        elif outcome == 'no_core':
            value['rubric'][0]['priority'] = 'optional'
        elif outcome == 'length':
            finish = 'length'
        elif outcome == 'refusal':
            finish = 'content_filter'
        elif outcome == 'overlong':
            value['answer'][0]['text'] = 'X' * 100
        elif outcome == 'wrong_namespace':
            value['answer'][0]['evidence_ids'] = ['E1']
        text = 'SECRET-CANARY' if outcome == 'invalid' else json.dumps(value)
        usage = {} if self.cost is None else dict(cost=self.cost)
        return MinimaLlmResponse(request.request_id, text,
            raw=dict(choices=[dict(finish_reason=finish)], usage=usage))


def session(tmp_path, outcomes, *, endpoint='https://openrouter.ai/api/v1', budget=10, cost=.02):
    provider = ReferenceProvider(tmp_path/'cache', outcomes, endpoint=endpoint, cost=cost)
    backend = CompletionSession(None, tmp_path/'budget.sqlite3', budget,
                                factory=lambda _: provider)
    return backend, provider


def generate(backend, source=None, **kwargs):
    return asyncio.run(generate_answer(backend, packet() if source is None else source,
                                      repair_length=True, **kwargs))[0]


@pytest.mark.parametrize('outcome,code', [
    ('foreign', 'foreign_evidence'), ('duplicate', 'duplicate_evidence'),
    ('duplicate_item', 'invalid_items'), ('no_core', 'invalid_items'),
    ('invalid', 'schema'), ('length', 'truncated'), ('refusal', 'incomplete'),
])
def test_safe_failure_categories_never_echo_values(tmp_path, outcome, code):
    from judges.generic.structured_completion import StructuredOutputError
    backend, provider = session(tmp_path, [outcome])
    result = asyncio.run(backend.generate(make_request(packet())))
    with pytest.raises(StructuredOutputError) as caught:
        parse_answer(result, {'e1'})
    assert caught.value.code == code
    assert 'SECRET-CANARY' not in ''.join(traceback.format_exception(caught.value))
    assert len(provider.calls) == 1


@pytest.mark.parametrize('failure', ['foreign', 'duplicate', 'duplicate_item', 'no_core', 'invalid', 'length'])
def test_completed_invalid_reference_gets_one_exact_repair_and_replay(tmp_path, failure):
    source = packet()
    original = copy.deepcopy(source)
    live, provider = session(tmp_path, [failure, 'stop'])
    answer = generate(live, source)
    assert answer.model_dump() == bundle()
    assert source == original
    assert len(provider.calls) == 2
    first, repair = [json.loads(r.messages[1]['content']) for r in provider.calls]
    assert first == source
    assert repair['question'] == source['question']
    assert repair == source
    assert 'SECRET-CANARY' not in provider.calls[1].messages[1]['content']
    assert provider.calls[0].messages != provider.calls[1].messages
    assert len(list(live.directory.glob('*.json'))) == 2
    receipts = {p.name: p.read_bytes() for p in live.directory.glob('*.json')}
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint='EMPTY')
    assert generate(offline, source) == answer
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before
    assert {p.name: p.read_bytes() for p in live.directory.glob('*.json')} == receipts


@pytest.mark.parametrize('failure', ['foreign', 'duplicate', 'invalid', 'length', 'overlong', 'wrong_namespace'])
def test_failed_repair_is_terminal_even_when_failure_changes(tmp_path, failure):
    live, provider = session(tmp_path, ['foreign', failure, 'stop'])
    with pytest.raises(JudgeError):
        generate(live, dict(packet(), question=dict(limit=50)))
    assert len(provider.calls) == 2
    assert len(list(live.directory.glob('*.json'))) == 2
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint='EMPTY')
    with pytest.raises(JudgeError):
        generate(offline, dict(packet(), question=dict(limit=50)))
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_refusal_is_not_repaired(tmp_path):
    live, provider = session(tmp_path, ['refusal', 'stop'])
    with pytest.raises(JudgeError):
        generate(live)
    assert len(provider.calls) == 1


def test_budget_refusal_before_repair_sends_no_call_or_new_reservation(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'stop'], budget=.05)
    with pytest.raises(RunStopped) as caught:
        generate(live)
    assert caught.value.code == 'budget_exhausted'
    # The original request itself needs >$0.32 reserved. No network is allowed.
    assert provider.calls == []
    assert ledger_rows(tmp_path) == []


def test_ambiguous_repair_preserves_receipt_marker_and_reservation(tmp_path):
    live, provider = session(tmp_path, ['foreign', RuntimeError('SECRET-CANARY'), 'stop'])
    with pytest.raises(RunStopped) as caught:
        generate(live)
    assert caught.value.code == 'transport'
    assert 'SECRET-CANARY' not in ''.join(traceback.format_exception(caught.value))
    assert len(provider.calls) == 2
    assert len([p for p in live.directory.glob('*.json') if len(p.stem) == 64]) == 1
    assert len(list(live.directory.glob('*.started'))) == 2
    before = ledger_rows(tmp_path)
    assert len(before) == 2 and before[1][1] is None
    resumed, unused = session(tmp_path, ['stop'])
    with pytest.raises(JudgeError):
        generate(resumed)
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before


def test_repair_input_overflow_stops_before_new_call(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'stop'])
    with pytest.raises(JudgeError, match='input'):
        generate(live, max_input_tokens=1)
    assert len(provider.calls) == 1
    assert len(ledger_rows(tmp_path)) == 1


def test_valid_reference_keeps_original_key_and_zero_extra_calls(tmp_path):
    live, provider = session(tmp_path, ['stop'])
    saved = asyncio.run(live.generate(make_request(packet())))
    receipts = {p.name: p.read_bytes() for p in live.directory.glob('*.json')}
    assert generate(live).model_dump() == bundle()
    assert len(provider.calls) == 1
    assert {p.name: p.read_bytes() for p in live.directory.glob('*.json')} == receipts
    assert saved.text == json.dumps(bundle())


@pytest.mark.parametrize('joint,compact,expected', [
    (False, False, 'c9f96428ae05111c2789aa41f1b768acccec18928e2a1c0aeab0366e9bdc7df9'),
    (True, False, 'b299085f15b20523022aa19715acd39e66ef8811fce0db3db5ecbbf2e1fe2c7d'),
    (True, True, 'ecb9164948130cff6d37728dff3e587b3ec2c10c0148333f31233af83d467ae6'),
])
def test_legacy_request_keys_are_byte_identical(joint, compact, expected):
    # Golden fingerprints measured from frozen execution commit 2019f03.
    request = make_request(packet(), joint_checklist=joint, validated_claims=compact)
    assert fingerprint(request) == expected


def fingerprint(request):
    return hashlib.sha256(json.dumps(dict(model='synthetic-model',
        messages=request.messages, extra=request.extra, max_tokens=request.max_tokens,
        temperature=request.temperature), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_legacy_length_repair_key_is_unchanged():
    calls = []

    class Backend:
        async def generate(self, request):
            calls.append(request)
            value = bundle()
            if len(calls) == 2:
                value['answer'][0]['text'] = 'Elm $2.'
            return MinimaLlmResponse(request.request_id, json.dumps(value),
                                    raw=dict(choices=[dict(finish_reason='stop')]))

    generate(Backend(), dict(packet(), question=dict(title='Price?', limit=8)))
    assert fingerprint(calls[1]) == '84cb25766df00f2e82869f6caaf4a25922cde76f8dd69b9b09548f6bdc282032'


def test_budget_refuses_only_the_repair_before_sending(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'stop'], budget=.36)
    with pytest.raises(RunStopped) as caught:
        generate(live)
    assert caught.value.code == 'budget_exhausted'
    assert len(provider.calls) == len(ledger_rows(tmp_path)) == 1
    assert len(list(live.directory.glob('*.started'))) == 1


def test_unknown_original_and_repair_costs_stay_reserved(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'stop'], cost=None)
    generate(live)
    rows = ledger_rows(tmp_path)
    assert len(provider.calls) == len(rows) == 2
    assert all(reported is None for _, reported, _ in rows)
    assert live.summary()['committed_usd'] == pytest.approx(sum(row[0] for row in rows))


@pytest.mark.parametrize('compact', [False, True])
def test_joint_repair_keeps_all_exact_ids_without_modifying_input(compact):
    from tests.test_single_checklist import item
    calls = []
    source = dict(question=dict(title='Compare the color and price.'),
                  evidence=[dict(id='0001', text='Blue.'), dict(id='E1', text='Elm $2.')],
                  partial_claims=[dict(text='Blue and round.', label='partially_supported')])
    original = copy.deepcopy(source)

    class Backend:
        async def generate(self, request):
            calls.append(request)
            data = json.loads(request.messages[1]['content'])
            ids = [row['id'] for row in data['evidence']]
            value = dict(answer=[dict(text='Blue; Elm $2.', evidence_ids=ids)],
                         checklist=[dict(item('F'), evidence_ids=ids)], gaps=[])
            if len(calls) == 1:
                value['checklist'][0]['evidence_ids'] = ['foreign']
            return MinimaLlmResponse(request.request_id, json.dumps(value),
                                    raw=dict(choices=[dict(finish_reason='stop')]))

    answer = generate(Backend(), source, joint_checklist=True, validated_claims=compact)
    assert answer.answer[0].evidence_ids == answer.checklist[0].evidence_ids == ['0001', 'E1']
    assert source == original
    repaired = json.loads(calls[1].messages[1]['content'])
    assert [e['id'] for e in repaired['evidence']] == ['0001', 'E1']
    assert repaired['partial_claims'] == source['partial_claims']


@pytest.mark.parametrize('raw', [None, {}, {'choices': None}, {'choices': [{}]},
                               {'choices': [None]}, {'choices': [{}, {}]},
                               {'choices': [{'finish_reason': 'stop', 'message': {'refusal': 'SECRET-CANARY'}}]}])
def test_missing_or_refused_completion_is_not_repairable(raw):
    from judges.generic.structured_completion import StructuredOutputError, REPAIRABLE
    response = MinimaLlmResponse('r', json.dumps(bundle()), raw=raw)
    with pytest.raises(StructuredOutputError) as caught:
        parse_answer(response, {'e1'})
    assert caught.value.code not in REPAIRABLE
    assert 'SECRET-CANARY' not in str(caught.value)


def test_repair_does_not_leave_ambiguous_aliases_in_gap_prose():
    calls = []
    source = dict(question=dict(title='Compare sources.'), evidence=[
        dict(id='0001', text='Elm costs $2.'), dict(id='E1', text='Elm costs $3.')])

    class Backend:
        async def generate(self, request):
            calls.append(request)
            payload = json.loads(request.messages[1]['content'])
            ids = [row['id'] for row in payload['evidence']]
            value = bundle()
            value['answer'][0]['evidence_ids'] = value['rubric'][0]['evidence_ids'] = ids
            value['gaps'] = [f'{ids[0]} conflicts with {ids[1]}; unresolved.']
            if len(calls) == 1:
                value['answer'][0]['evidence_ids'] = ['foreign']
            return MinimaLlmResponse(request.request_id, json.dumps(value),
                                    raw=dict(choices=[dict(finish_reason='stop')]))

    answer = generate(Backend(), source)
    assert answer.gaps == ['0001 conflicts with E1; unresolved.']


@pytest.mark.parametrize('field', ['question', 'requirement', 'request_basis', 'answer_text'])
@pytest.mark.parametrize('blank', ['', '  \n '])
def test_joint_reference_enforces_downstream_nonblank_contract(field, blank):
    from tests.test_single_checklist import item
    from judges.generic.structured_completion import StructuredOutputError
    value = dict(answer=[dict(text='Blue.', evidence_ids=['e'])],
                 checklist=[item('F')], gaps=[])
    if field == 'answer_text':
        value['answer'][0]['text'] = blank
    else:
        value['checklist'][0][field] = blank
    response = MinimaLlmResponse('r', json.dumps(value), raw=dict(choices=[dict(finish_reason='stop')]))
    with pytest.raises(StructuredOutputError):
        parse_answer(response, {'e'}, joint_checklist=True)


@pytest.mark.parametrize('track', ['rag', 'ragtime'])
def test_blank_checklist_recovery_and_audit_through_full_judge(tmp_path, track):
    from types import SimpleNamespace
    from autojudge_base import Request
    from judges.generic.unified import UnifiedJudge
    from tests.test_unified_judge import SemanticBackend, report

    class BlankChecklistBackend(SemanticBackend):
        async def generate(self, request):
            result = await super().generate(request)
            if request.request_id == 'reference-checklist-v3':
                value = json.loads(result.text)
                value['checklist'][0]['request_basis'] = '   '
                return MinimaLlmResponse(result.request_id, json.dumps(value), raw=result.raw)
            return result

    reports = [report([dict(text='Blue.', citations={'d': 100})],
                      {'d': dict(id='d', text='Blue.')})]
    question = Request(request_id='t', title='What color?', word_limit=20)
    provider = BlankChecklistBackend(tmp_path/'cache')
    judge = UnifiedJudge(backend_factory=lambda _: provider)
    judge.judge(reports, [question], SimpleNamespace(), outdir=tmp_path/'first',
                track=track, method='evidence', repair_reference_output=True)
    saved = json.loads((tmp_path/'first/unified.stages.json').read_text())
    assert saved['repair_reference_output'] is True
    audit = saved['topics'][0]['evidence']['reference_audit']
    assert audit['repaired'] and audit['repair_failure_code'] == 'invalid_items'
    assert audit['accepted_request_id'] == 'reference-output-repair-v1'
    assert provider.calls.count('ReferenceChecklist') == 2
    offline = SemanticBackend(tmp_path/'cache', offline=True)
    UnifiedJudge(backend_factory=lambda _: offline).judge(reports, [question], SimpleNamespace(),
        outdir=tmp_path/'offline', track=track, method='evidence', repair_reference_output=True)
    assert offline.calls == []
    replay = json.loads((tmp_path/'offline/unified.stages.json').read_text())
    assert saved['topics'] == replay['topics']


def test_workflow_can_disable_reference_output_repair(tmp_path):
    from types import SimpleNamespace
    from autojudge_base import Request
    from judges.generic.unified import UnifiedJudge
    from tests.test_unified_judge import SemanticBackend, report

    class InvalidReferenceBackend(SemanticBackend):
        async def generate(self, request):
            result = await super().generate(request)
            if request.request_id == 'reference-checklist-v3':
                value = json.loads(result.text)
                value['answer'][0]['evidence_ids'] = ['foreign']
                return MinimaLlmResponse(result.request_id, json.dumps(value), raw=result.raw)
            return result

    provider = InvalidReferenceBackend(tmp_path/'cache')
    reports = [report([dict(text='Blue.', citations={'d': 100})], {'d': dict(id='d', text='Blue.')})]
    with pytest.raises(JudgeError):
        UnifiedJudge(backend_factory=lambda _: provider).judge(reports,
            [Request(request_id='t', title='What color?')], SimpleNamespace(),
            outdir=tmp_path/'out', method='evidence', repair_reference_output=False)
    assert provider.calls.count('ReferenceChecklist') == 1


@pytest.mark.parametrize('outcome,repaired', [('stop', False), ('foreign', True)])
def test_reference_provenance_is_code_owned(tmp_path, outcome, repaired):
    live, provider = session(tmp_path, [outcome, 'stop'])
    audit = {}
    generate(live, audit=audit)
    assert audit == dict(initial_request_id='reference-and-rubric-v5',
        accepted_request_id='reference-output-repair-v1' if repaired else 'reference-and-rubric-v5',
        repair_failure_code='foreign_evidence' if repaired else None,
        repaired=repaired)
