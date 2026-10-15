"""Compact references retain claims and provenance, not whole source chunks."""
import asyncio
import copy
import hashlib
import json

import pytest

from judges.generic.models import JudgeError


def topic():
    return dict(
        pool=dict(question=dict(title='Price and duration?', background='Budget $3', limit=2000),
            evidence=[dict(id='E00001', excerpt='Elm costs $2.', claims=[], pair_ids=['p1', 'p2']),
                      dict(id='E00002', excerpt='A second source.', claims=[], pair_ids=['p3'])]),
        answers=[dict(run_id='a', evidence=dict(id='A001', pairs=[
            dict(id='p1', sentence_index=0, document_id='d1', text='Elm costs $2.',
                 label='supported', uncertain=False, excerpts=['Elm costs $2.']),
            dict(id='p2', sentence_index=1, document_id='d1', text='Elm costs $2 and takes an hour.',
                 label='partially_supported', uncertain=False, excerpts=['Elm costs $2.'])])),
            dict(run_id='b', evidence=dict(id='A002', pairs=[
                dict(id='p3', sentence_index=0, document_id='d2', text='Elm costs $2.',
                     label='supported', uncertain=False, excerpts=['A second source.'])]))])


def test_compaction_keeps_exact_claims_and_all_source_links_without_mutation():
    from judges.generic.evidence_bridge import compact_claims
    original = topic()
    before = copy.deepcopy(original)
    packet, audit = compact_claims(original)
    assert original == before
    assert packet['question'] == before['pool']['question']
    assert packet['evidence'] == [dict(id='E00001', claims=[dict(text='Elm costs $2.', label='supported')])]
    assert packet['partial_claims'] == [dict(text='Elm costs $2 and takes an hour.', label='partially_supported')]
    assert set(audit[0]['evidence_ids']) == {'E00001', 'E00002'}
    assert [(p['pair_id'], p['run_id']) for p in audit[0]['sources']] == [('p1', 'a'), ('p3', 'b')]
    assert [p['pair_id'] for p in audit[1]['sources']] == ['p2']
    assert audit[1]['label'] == 'partially_supported' and audit[1]['reference_evidence_id'] is None


def test_equal_text_with_different_support_labels_is_not_promoted():
    from judges.generic.evidence_bridge import compact_claims
    value = topic()
    value['answers'][0]['evidence']['pairs'][1]['text'] = 'Elm costs $2.'
    packet, _ = compact_claims(value)
    assert packet['evidence'][0]['claims'][0]['label'] == 'supported'
    assert packet['partial_claims'][0]['label'] == 'partially_supported'


def test_uncertain_and_unsupported_pairs_do_not_become_reference_facts():
    from judges.generic.evidence_bridge import compact_claims
    value = topic()
    value['answers'][0]['evidence']['pairs'][0]['uncertain'] = True
    value['answers'][1]['evidence']['pairs'][0]['label'] = 'unsupported'
    with pytest.raises(JudgeError, match='fully supported'):
        compact_claims(value)


@pytest.mark.parametrize('excerpts', [[], ['Not in the canonical pool.']])
def test_compaction_rejects_untraceable_positive_claim(excerpts):
    from judges.generic.evidence_bridge import compact_claims
    value = topic()
    value['answers'][0]['evidence']['pairs'][0]['excerpts'] = excerpts
    with pytest.raises(JudgeError, match='source'):
        compact_claims(value)


def test_fitting_reference_keeps_sources_but_invalidates_old_prompt_cache():
    from judges.generic.reference_answer import reference_input, make_request, JOINT_PROMPT
    assert hashlib.sha256(JOINT_PROMPT.encode()).hexdigest() != (
        '82f0e0360737df4a4069b2c5a6eb5650d7642c797a7b6ebabc7cfc012e61ccf4')
    value = topic()
    packet, compact, audit = reference_input(value, 180000)
    assert packet == value['pool'] and not compact
    assert audit['mode'] == 'source_excerpts'
    assert make_request(packet, joint_checklist=True).messages == make_request(value['pool'], joint_checklist=True).messages


def test_large_document_pool_falls_back_without_source_text_or_claim_loss():
    from judges.generic.reference_answer import reference_input, make_request
    value = topic()
    long_source = 'Source text repeated. ' * 20000
    value['pool']['evidence'][0]['excerpt'] = long_source
    for pair in value['answers'][0]['evidence']['pairs']:
        pair['excerpts'] = [long_source]
    packet, compact, audit = reference_input(value, 6000)
    assert compact and audit['input_tokens'] <= 6000 < audit['source_input_tokens']
    assert len(audit['claims']) == 2
    req = make_request(packet, joint_checklist=True, validated_claims=True)
    assert req.request_id == 'reference-validated-claims-v3'
    assert long_source not in req.messages[1]['content']
    assert 'partially_supported' in req.messages[1]['content']
    assert 'which portion' in req.messages[0]['content']
    assert req.messages != make_request(packet, joint_checklist=True).messages


def test_compact_overflow_rejects_without_silent_top_k_or_truncation():
    from judges.generic.reference_answer import reference_input
    with pytest.raises(JudgeError, match='Compact reference input'):
        reference_input(topic(), 1)


@pytest.mark.parametrize('compact', [False, True])
def test_length_revision_keeps_assertion_policy_and_compact_partial_contract(compact):
    from judges.generic.reference_answer import generate_answer
    from minima_llm import MinimaLlmResponse
    requests = []
    class Backend:
        async def generate(self, req):
            requests.append(req)
            p = json.loads(req.messages[1]['content'])
            if len(requests) == 1 and compact:
                assert p['partial_claims'][0]['label'] == 'partially_supported'
            answer = dict(answer=[dict(text='Too long.' if len(requests) == 1 else 'Gap.', evidence_ids=['E00002'])],
                checklist=[dict(id='N1', question='Price?', requirement='Explain price.',
                    acceptable_alternatives=[], evidence_ids=['E00002'], request_basis='Asked.',
                    kind='factual', priority='core')], gaps=['Supported portion is unknown.'])
            return MinimaLlmResponse(req.request_id, json.dumps(answer), raw={'choices': [{'finish_reason': 'stop'}]})
    packet = dict(question=dict(limit=4), evidence=[dict(id='E00002', claims=[
        dict(text='Elm costs $2.', label='supported')])], partial_claims=[
        dict(text='Elm costs $2 and takes an hour.', label='partially_supported')])
    answer, _ = asyncio.run(generate_answer(Backend(), packet, joint_checklist=True,
        validated_claims=compact, repair_length=True))
    assert answer.answer[0].text == 'Gap.' and len(requests) == 2
    assert requests[1].request_id == 'reference-length-revision-v2'
    assert 'attached to that statement' in requests[1].messages[0]['content']
    if compact:
        assert 'which portion' in requests[1].messages[0]['content']


def test_partial_only_source_id_is_not_citable():
    from judges.generic.evidence_bridge import compact_claims
    from judges.generic.reference_answer import parse_answer
    from minima_llm import MinimaLlmResponse
    value = topic()
    value['answers'][0]['evidence']['pairs'][0]['uncertain'] = True
    packet, _ = compact_claims(value)
    assert {e['id'] for e in packet['evidence']} == {'E00002'}
    reply = dict(answer=[dict(text='Elm costs $2 and takes an hour.', evidence_ids=['E00001'])],
        checklist=[dict(id='N1', question='Price?', requirement='State price.',
            acceptable_alternatives=[], evidence_ids=[], request_basis='Asked.', kind='factual', priority='core')], gaps=[])
    with pytest.raises(JudgeError, match='evidence references'):
        parse_answer(MinimaLlmResponse('r', json.dumps(reply), raw={'choices': [{'finish_reason': 'stop'}]}),
            {e['id'] for e in packet['evidence']}, joint_checklist=True)


def test_citation_heavy_pool_preserves_every_pair_and_exact_claim():
    from judges.generic.evidence_bridge import compact_claims
    value = dict(pool=dict(question=dict(title='Summarize', limit=2000), evidence=[]), answers=[])
    for i in range(520):
        excerpt = f'Source {i}. ' * 100
        value['pool']['evidence'].append(dict(id=f'E{i:05d}', excerpt=excerpt))
        value['answers'].append(dict(run_id=str(i), evidence=dict(id=f'A{i}', pairs=[
            dict(id=f'p{i}', sentence_index=0, document_id=f'd{i}', text=f'Fact {i%17}.',
                label='supported' if i%2 else 'partially_supported', uncertain=False,
                excerpts=[excerpt])])))
    value['answers'][0]['evidence']['pairs'][0]['excerpts'].append(value['pool']['evidence'][1]['excerpt'])
    before = copy.deepcopy(value)
    packet, audit = compact_claims(value)
    assert value == before
    assert compact_claims(value) == (packet, audit)
    assert len(audit) == 34 and sum(len(c['sources']) for c in audit) == 520
    assert len(audit[0]['sources'][0]['evidence_ids']) == 2
    assert len(packet['partial_claims']) == 17
    assert sum(len(e['claims']) for e in packet['evidence']) == 17


@pytest.mark.parametrize('track', ['rag', 'ragtime'])
def test_compact_overflow_runs_through_same_grades_scores_and_offline_cache(tmp_path, monkeypatch, track):
    from judges.generic import unified
    from tests.test_unified_judge import SemanticBackend, report
    from autojudge_base import Request
    from types import SimpleNamespace
    original = unified.build_topic
    def enlarged(*args):
        result = original(*args)
        long_source = 'Long retained evidence. ' * 20000
        result['pool']['evidence'][0]['excerpt'] = long_source
        for record in result['answers']:
            for pair in record['evidence']['pairs']:
                pair['excerpts'] = [long_source]
        return result
    monkeypatch.setattr(unified, 'build_topic', enlarged)
    reports = [report([dict(text='Blue.', citations={'d': 100})], {'d': dict(id='d', text='Blue.')})]
    question = Request(request_id='t', title='What color?', word_limit=20)
    backend = SemanticBackend(tmp_path/'cache')
    for out, client in [('first', backend), ('replay', SemanticBackend(tmp_path/'cache', offline=True))]:
        unified.UnifiedJudge(backend_factory=lambda _: client).judge(reports, [question], SimpleNamespace(),
            method='evidence', track=track, outdir=tmp_path/out, max_input_tokens=6000)
    first = json.loads((tmp_path/'first/unified.stages.json').read_text())['topics']
    replay = json.loads((tmp_path/'replay/unified.stages.json').read_text())['topics']
    assert first == replay
    assert first[0]['evidence']['reference_input']['mode'] == 'validated_claims'
    assert backend.calls.count('ReferenceChecklist') == 1
    assert backend.calls.count('ChecklistCoverage') == 1


@pytest.mark.parametrize('bad_repair', [False, True])
def test_completed_incomplete_checklist_ids_get_one_exact_mapped_repair(tmp_path, bad_repair):
    from judges.generic.unified import UnifiedJudge
    from tests.test_unified_judge import SemanticBackend, report
    from autojudge_base import Request
    from minima_llm import MinimaLlmResponse
    from types import SimpleNamespace
    requests = []
    class Backend(SemanticBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.extra['response_format']['json_schema']['name'] == 'ChecklistCoverage':
                requests.append(req)
                value = json.loads(result.text)
                if len(requests) == 1:
                    value['items'] = []
                else:
                    payload = json.loads(req.messages[1]['content'])
                    assert payload['items'][0]['id'] == 'I1'
                    assert payload['answer']['claims'][0]['id'] == 'C1'
                    if bad_repair:
                        value['items'][0]['id'] = 'invented'
                return MinimaLlmResponse(req.request_id, json.dumps(value), raw={'choices': [{'finish_reason': 'stop'}]})
            return result
    client = Backend(tmp_path/'cache')
    judge = UnifiedJudge(backend_factory=lambda _: client)
    args = ([report([dict(text='Blue.', citations={'d': 100})], {'d': dict(id='d', text='Blue.')})],
            [Request(request_id='t', title='What color?')], SimpleNamespace())
    if bad_repair:
        with pytest.raises(JudgeError, match='aliased checklist'):
            judge.judge(*args, method='evidence', outdir=tmp_path/'out')
    else:
        judge.judge(*args, method='evidence', outdir=tmp_path/'out')
        topic = json.loads((tmp_path/'out/unified.stages.json').read_text())['topics'][0]['evidence']
        assert topic['answers'][0]['checklist_coverage']['items'][0]['id'] == topic['checklist']['items'][0]['id']
    assert len(requests) == 2
