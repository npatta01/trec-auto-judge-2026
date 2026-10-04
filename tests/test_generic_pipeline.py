import asyncio
import json
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmResponse

from tests.test_generic_judge import report


BRIEF = dict(explicit_needs=[], inferred_needs=[], ambiguities=[], questionable_premises=[])


class SyntheticBackend:
    def __init__(self, *args):
        self.requests = []
        self.closed = False

    async def generate(self, req):
        self.requests.append(req)
        phase = req.messages[0]['content'].splitlines()[0]
        data = json.loads(req.messages[1]['content'])
        if phase == 'PHASE: interpret':
            result = BRIEF
        elif phase == 'PHASE: audit':
            result = {'evidence': [dict(sentence_id=s['sentence_id'], document_id=data['document_id'],
                        status='supported', quote=data['document']) for s in data['sentences']]}
        elif phase == 'PHASE: audit_spans':
            result = {'evidence': [dict(sentence_id=s['sentence_id'], document_id=data['document_id'],
                        status='supported', source_ids=[u['source_id'] for u in data['document_units']])
                        for s in data['sentences']]}
        else:
            assessment = dict(usefulness=3, request_coverage=2, ambiguity_handling=4,
                sentences=[dict(sentence_id=s['sentence_id'], needs_citation=True)
                           for s in data['answer']['sentences']])
            result = assessment
            if phase == 'PHASE: direct':
                result = dict(assessment=assessment, evidence=[
                    dict(sentence_id=s['sentence_id'], document_id=d, status='supported', quote=text)
                    for s in data['answer']['sentences']
                    for d, text in data['answer']['documents'].items() if d in s['citations']])
        return MinimaLlmResponse(req.request_id, json.dumps(result))

    async def aclose(self):
        self.closed = True


def test_json_retries_change_prompt_and_fail_safely(capsys):
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief, JudgeError

    class Broken(SyntheticBackend):
        async def generate(self, req):
            self.requests.append(req)
            print('SECRET provider body')
            return MinimaLlmResponse(req.request_id, 'SECRET malformed')

    backend = Broken()
    client = JsonClient(backend, max_prompt_chars=10000, max_tokens=100, schema_attempts=2)
    with pytest.raises(JudgeError, match='structured output') as err:
        asyncio.run(client.ask('interpret', {}, Brief))
    assert 'SECRET' not in str(err.value)
    assert len(backend.requests) == 2
    assert backend.requests[0].messages != backend.requests[1].messages
    captured = capsys.readouterr()
    assert 'SECRET' not in captured.out + captured.err


@pytest.mark.parametrize('mode,calls', [('direct', 1), ('staged', 4)])
def test_pipeline(mode, calls, monkeypatch):
    from judges.generic import judge
    backend = SyntheticBackend()
    monkeypatch.setattr(judge, 'make_backend', lambda _: backend)
    result = judge.GenericJudge().judge([report()], [Request(request_id='t', title='Compare')],
                                      SimpleNamespace(raw={}), mode=mode)
    assert len(backend.requests) == calls
    assert backend.closed
    rows = {(m, e.topic_id): v for e in result.entries for m, v in e.values.items()}
    assert rows['USEFULNESS', 't'] == .75
    assert rows['CITE_PRECISION_PROXY', 't'] == 1
    assert rows['CITE_RECALL_PROXY', 't'] == pytest.approx(1/3)


def test_empty_report_does_not_construct_backend(monkeypatch):
    from judges.generic import judge
    monkeypatch.setattr(judge, 'make_backend', lambda _: pytest.fail('LLM called'))
    result = judge.GenericJudge().judge([report(sentences=[])], [Request(request_id='t', title='')],
                                      SimpleNamespace(raw={}))
    assert all(v == 0 for e in result.entries for v in e.values.values())


def test_prompt_budget_fails_before_network():
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief, JudgeError
    backend = SyntheticBackend()
    client = JsonClient(backend, max_prompt_chars=1, max_tokens=100, schema_attempts=2)
    with pytest.raises(JudgeError, match='budget'):
        asyncio.run(client.ask('interpret', {}, Brief))
    assert not backend.requests


def test_missing_topic_cell_fails(monkeypatch):
    from judges.generic import judge
    from judges.generic.models import JudgeError
    monkeypatch.setattr(judge, 'make_backend', lambda _: SyntheticBackend())
    with pytest.raises(JudgeError, match='coverage'):
        judge.GenericJudge().judge([report(sentences=[])],
            [Request(request_id='t', title=''), Request(request_id='other', title='')], SimpleNamespace(raw={}))


@pytest.mark.parametrize('mode', ['direct', 'staged'])
def test_missing_documents_preserve_zero_availability(mode, monkeypatch):
    from judges.generic import judge
    monkeypatch.setattr(judge, 'make_backend', lambda _: SyntheticBackend())
    result = judge.GenericJudge().judge([report(documents={})], [Request(request_id='t', title='Compare')],
                                      SimpleNamespace(raw={}), mode=mode)
    row = next(e.values for e in result.entries if e.topic_id == 't')
    assert row['EVIDENCE_AVAILABILITY'] == row['CITE_PRECISION_PROXY'] == 0
    assert row['USEFULNESS'] == .75


def test_chunk_merge_does_not_hide_contradiction():
    from judges.generic.judge import chunks, merge_evidence
    from judges.generic.models import Evidence
    text = 'ABCDEFGHIJ'
    assert list(chunks(text, 6, 2)) == ['ABCDEF', 'EFGHIJ']
    labels = [Evidence(sentence_id=0, document_id='d', status=s, quote=q)
              for s, q in [('supported', 'ABC'), ('contradicted', 'HIJ')]]
    assert merge_evidence(labels).status == 'partial'


def test_incomplete_audit_fails_and_closes(monkeypatch):
    from judges.generic import judge
    from judges.generic.models import JudgeError

    class DropsLabels(SyntheticBackend):
        async def generate(self, req):
            if req.messages[0]['content'].startswith('PHASE: audit'):
                return MinimaLlmResponse(req.request_id, '{"evidence": []}')
            return await super().generate(req)

    backend = DropsLabels()
    monkeypatch.setattr(judge, 'make_backend', lambda _: backend)
    with pytest.raises(JudgeError, match='labels'):
        judge.GenericJudge().judge([report()], [Request(request_id='t', title='Compare')], SimpleNamespace(raw={}))
    assert backend.closed


def test_request_interpretation_cache_uses_background(monkeypatch):
    from judges.generic import judge
    backend = SyntheticBackend()
    monkeypatch.setattr(judge, 'make_backend', lambda _: backend)
    reports, topics = [], []
    for i, background in enumerate(['School', 'Hospital', 'School']):
        r = report()
        r.metadata.topic_id = str(i)
        reports.append(r)
        topics.append(Request(request_id=str(i), title='Compare', background=background))
    judge.GenericJudge().judge(reports, topics, SimpleNamespace(raw={}))
    assert sum(r.messages[0]['content'].startswith('PHASE: interpret') for r in backend.requests) == 2


def test_transport_failure_is_not_a_grade():
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief, JudgeError
    from minima_llm import MinimaLlmFailure

    class Fails(SyntheticBackend):
        async def generate(self, req):
            return MinimaLlmFailure(req.request_id, 'error', 'SECRET body', 1)

    client = JsonClient(Fails(), max_prompt_chars=10000, max_tokens=100, schema_attempts=2)
    with pytest.raises(JudgeError) as err:
        asyncio.run(client.ask('interpret', {}, Brief))
    assert 'SECRET' not in str(err.value)


def test_audit_has_surrounding_answer_context(monkeypatch):
    from judges.generic import judge
    backend = SyntheticBackend()
    monkeypatch.setattr(judge, 'make_backend', lambda _: backend)
    judge.GenericJudge().judge([report()], [Request(request_id='t', title='Compare')], SimpleNamespace(raw={}))
    audit = next(r for r in backend.requests if r.messages[0]['content'].startswith('PHASE: audit'))
    assert len(json.loads(audit.messages[1]['content'])['answer_context']) == 3


def test_semantically_invalid_result_gets_bounded_retry():
    from judges.generic.client import JsonClient
    from judges.generic.models import Audit, JudgeError
    from judges.generic.scoring import validate_evidence

    class Repairable(SyntheticBackend):
        async def generate(self, req):
            self.requests.append(req)
            entries = [] if len(self.requests) == 1 else [dict(
                sentence_id=0, document_id='d', status='supported', quote='blue')]
            return MinimaLlmResponse(req.request_id, json.dumps({'evidence': entries}))

    backend = Repairable()
    client = JsonClient(backend, max_prompt_chars=20000, max_tokens=100, schema_attempts=2)
    audit = asyncio.run(client.ask('audit', {}, Audit,
        validator=lambda x: validate_evidence(x.evidence, {(0, 'd')}, {'d': 'blue'})))
    assert audit.evidence[0].status == 'supported'
    assert len(backend.requests) == 2


def test_structured_output_contract_is_forwarded():
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief

    class SchemaEndpoint(SyntheticBackend):
        async def generate(self, req):
            contract = req.extra['response_format']
            assert contract['type'] == 'json_schema'
            assert contract['json_schema']['strict'] is True
            assert 'explicit_needs' in contract['json_schema']['schema']['properties']
            return MinimaLlmResponse(req.request_id, json.dumps(BRIEF))

    client = JsonClient(SchemaEndpoint(), max_prompt_chars=10000, max_tokens=100,
                        schema_attempts=2, structured_output=True)
    assert asyncio.run(client.ask('interpret', {}, Brief)).explicit_needs == []


def test_budget_error_is_not_misreported_as_transport_failure():
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief, JudgeError

    class Backend:
        async def generate(self, req):
            raise JudgeError('Experiment budget exhausted before network call.')

    client = JsonClient(Backend(), max_prompt_chars=10000, max_tokens=100, schema_attempts=2)
    with pytest.raises(JudgeError, match='budget'):
        asyncio.run(client.ask('interpret', {}, Brief))


def test_request_options_survive_in_cache_key_request():
    from judges.generic.client import JsonClient
    from judges.generic.models import Brief

    class OptionsEndpoint(SyntheticBackend):
        async def generate(self, req):
            assert req.extra['reasoning'] == {'enabled': False}
            assert req.extra['provider']['max_price']['prompt'] == 1
            assert req.extra['response_format']['type'] == 'json_schema'
            return MinimaLlmResponse(req.request_id, json.dumps(BRIEF))

    client = JsonClient(OptionsEndpoint(), max_prompt_chars=10000, max_tokens=100,
        schema_attempts=2, structured_output=True,
        request_extra={'reasoning': {'enabled': False}, 'provider': {'max_price': {'prompt': 1}}})
    assert asyncio.run(client.ask('interpret', {}, Brief)).explicit_needs == []
