"""A bounded reference-ID recovery through real cache and budget ownership."""
import copy
import json
from dataclasses import replace

import pytest

from judges.generic.document_pipeline import token_count
from judges.generic.completion_session import CompletionSession
from judges.generic.models import JudgeError, RunStopped
from judges.generic.structured_completion import StructuredOutputError
from judges.generic.unified import UnifiedJudge
from tests.test_reference_recovery import ReferenceProvider, generate, session
from tests.test_partial_output_recovery import ledger_rows


def sparse_packet():
    return dict(question=dict(title='Price?'), claims=[],
                evidence=[dict(id='E00007', text='Elm costs $2.'),
                          dict(id='E00002', text='Elm is available.')],
                partial_claims=[dict(id='hint1', text='Unverified hint.')])


def test_completed_foreign_repair_recovers_once_and_replays_without_new_charges(tmp_path):
    source = sparse_packet()
    original = copy.deepcopy(source)
    live, provider = session(tmp_path, ['foreign', 'foreign', 'stop'])
    audit = {}
    answer = generate(live, source, repair_reference_ids=True, audit=audit)
    assert answer.answer[0].evidence_ids == ['E00007']
    assert source == original
    assert len(provider.calls) == 3
    fallback = provider.calls[2]
    schema = fallback.extra['response_format']['json_schema']['schema']
    assert schema['$defs']['EvidenceId']['enum'] == ['E00007', 'E00002']
    for name in ['AnswerStatement', 'Criterion']:
        assert schema['$defs'][name]['properties']['evidence_ids']['items'] == {
            '$ref': '#/$defs/EvidenceId'}
    assert json.loads(fallback.messages[1]['content']) == source
    assert fallback.extra != provider.calls[1].extra
    assert audit['id_contract_fallback'] is True
    assert audit['accepted_request_id'] == fallback.request_id
    receipts = {p.name: p.read_bytes() for p in live.directory.glob('*.json')}
    before = ledger_rows(tmp_path)
    assert len(before) == 3 and len(receipts) == 3
    offline, unused = session(tmp_path, [], endpoint='EMPTY')
    assert generate(offline, source, repair_reference_ids=True) == answer
    assert unused.calls == []
    assert ledger_rows(tmp_path) == before
    assert {p.name: p.read_bytes() for p in live.directory.glob('*.json')} == receipts


@pytest.mark.parametrize('outcome,code', [('foreign', 'foreign_evidence'),
    ('duplicate', 'duplicate_evidence'), ('overlong', 'answer_length'),
    ('refusal', 'incomplete')])
def test_invalid_id_contract_fallback_is_terminal_and_replayable(tmp_path, outcome, code):
    live, provider = session(tmp_path, ['foreign', 'foreign', outcome, 'stop'])
    source = dict(sparse_packet(), question=dict(title='Price?', limit=50))
    with pytest.raises(StructuredOutputError) as caught:
        generate(live, source, repair_reference_ids=True)
    assert caught.value.code == code
    assert len(provider.calls) == 3
    before = ledger_rows(tmp_path)
    offline, unused = session(tmp_path, [], endpoint='EMPTY')
    with pytest.raises(StructuredOutputError) as caught:
        generate(offline, source, repair_reference_ids=True)
    assert caught.value.code == code
    assert unused.calls == [] and ledger_rows(tmp_path) == before


@pytest.mark.parametrize('first,second', [('foreign', 'duplicate'),
    ('foreign', 'refusal'), ('overlong', 'foreign')])
def test_other_failures_and_length_repair_do_not_get_id_regeneration(tmp_path, first, second):
    live, provider = session(tmp_path, [first, second, 'stop'])
    with pytest.raises(JudgeError):
        generate(live, dict(sparse_packet(), question=dict(limit=50)), repair_reference_ids=True)
    assert len(provider.calls) == 2


def test_id_regeneration_never_retries_ambiguous_transport(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'foreign', RuntimeError('private'), 'stop'])
    with pytest.raises(RunStopped) as caught:
        generate(live, sparse_packet(), repair_reference_ids=True)
    assert caught.value.code == 'transport'
    before = ledger_rows(tmp_path)
    assert len(before) == 3 and before[-1][1] is None
    resumed, unused = session(tmp_path, ['stop'])
    with pytest.raises(JudgeError):
        generate(resumed, sparse_packet(), repair_reference_ids=True)
    assert unused.calls == [] and ledger_rows(tmp_path) == before


def test_disabled_recovery_preserves_two_call_terminal_contract(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'foreign', 'stop'])
    with pytest.raises(StructuredOutputError):
        generate(live, sparse_packet(), repair_reference_ids=False)
    assert len(provider.calls) == 2


def test_schema_overhead_is_counted_before_fallback_purchase(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'foreign', 'stop'])
    with pytest.raises(StructuredOutputError):
        generate(live, sparse_packet())
    limit = max(token_count(json.dumps(r.messages)) for r in provider.calls)
    before = ledger_rows(tmp_path)
    with pytest.raises(StructuredOutputError) as caught:
        generate(live, sparse_packet(), repair_reference_ids=True, max_input_tokens=limit)
    assert caught.value.code == 'input_overflow'
    assert len(provider.calls) == 2 and ledger_rows(tmp_path) == before


def test_offline_missing_fallback_does_not_purchase_or_modify_receipts(tmp_path):
    live, provider = session(tmp_path, ['foreign', 'foreign'])
    with pytest.raises(StructuredOutputError):
        generate(live, sparse_packet())
    before = ledger_rows(tmp_path)
    receipts = {p.name: p.read_bytes() for p in live.directory.glob('*.json')}
    offline, unused = session(tmp_path, [], endpoint='EMPTY')
    with pytest.raises(JudgeError, match='cache miss'):
        generate(offline, sparse_packet(), repair_reference_ids=True)
    assert unused.calls == [] and ledger_rows(tmp_path) == before
    assert {p.name: p.read_bytes() for p in live.directory.glob('*.json')} == receipts


@pytest.mark.parametrize('compact', [False, True])
def test_joint_checklist_id_contract_retains_nonfactual_criteria_and_partial_hints(tmp_path, compact):
    class JointProvider(ReferenceProvider):
        async def generate(self, request):
            result = await super().generate(request)
            value = json.loads(result.text)
            criterion = value.pop('rubric')[0]
            value['checklist'] = [dict(id=criterion['id'], question='Price?',
                requirement=criterion['requirement'], acceptable_alternatives=['Equivalent price'],
                evidence_ids=criterion['evidence_ids'], request_basis='Asked for price',
                kind='factual', priority='core')]
            return replace(result, text=json.dumps(value))

    provider = JointProvider(tmp_path/'cache', ['foreign', 'foreign', 'stop'], cost=.02)
    backend = CompletionSession(None, tmp_path/'budget.sqlite3', 10, factory=lambda _: provider)
    answer = generate(backend, sparse_packet(), joint_checklist=True,
                      validated_claims=compact, repair_reference_ids=True)
    assert answer.checklist[0].acceptable_alternatives == ['Equivalent price']
    schema = provider.calls[2].extra['response_format']['json_schema']['schema']
    assert schema['$defs']['EvidenceId']['enum'] == ['E00007', 'E00002']
    assert schema['$defs']['QuestionItem']['properties']['evidence_ids']['items'] == {
        '$ref': '#/$defs/EvidenceId'}
    assert 'enum' not in schema['$defs']['QuestionItem']['properties']['id']


def test_unified_rejects_nonboolean_id_recovery_before_admitting_work(tmp_path):
    with pytest.raises(JudgeError, match='Reference ID repair must be boolean'):
        UnifiedJudge().judge([], [], None, repair_reference_ids='yes', outdir=tmp_path)
