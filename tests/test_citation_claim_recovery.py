"""Completed attribution repair is bounded and never guesses claim mappings."""

import json
import asyncio
from dataclasses import replace

import pytest

from tests.test_citation_completion import CitationBackend, run_judge
from judges.generic.models import CitationStageError


def keyed_result(text, aliases=('C1',)):
    from minima_llm import MinimaLlmRequest, MinimaLlmResponse
    from judges.generic.citation_recovery import recover_citation_ids

    class Backend:
        requests = []

        async def generate(self, request):
            self.requests.append(request)
            return MinimaLlmResponse(request.request_id, text,
                raw={'choices': [{'finish_reason': 'stop'}]})

    backend = Backend()
    value = asyncio.run(recover_citation_ids(backend,
        MinimaLlmRequest('fixture', [], max_tokens=32768), expected_ids=aliases,
        context_tokens=64000, max_output_tokens=32768))
    return value, backend.requests


@pytest.mark.parametrize('count', [1, 5])
def test_keyed_schema_requires_every_claim_and_preserves_distinct_labels(count):
    labels = ['supported', 'partially_supported', 'unsupported', 'supported', 'unsupported']
    aliases = [f'C{i+1}' for i in range(count)]
    output = {alias: dict(eligibility='claim', label=labels[i], uncertain=False,
                         contradiction=False) for i, alias in reversed(list(enumerate(aliases)))}
    value, requests = keyed_result(json.dumps(output), aliases)
    assert [(v.id, v.label) for v in value.claims] == list(zip(aliases, labels[:count]))
    schema = requests[0].extra['response_format']['json_schema']['schema']
    assert schema['required'] == aliases and schema['additionalProperties'] is False
    assessment = schema['$defs']['ClaimAssessment']
    assert assessment['additionalProperties'] is False
    assert set(assessment['properties']) == {'eligibility', 'label', 'uncertain', 'contradiction'}


@pytest.mark.parametrize('failure', ['missing', 'foreign', 'case', 'id_field', 'label',
                                   'duplicate_key', 'duplicate_field'])
def test_invalid_keyed_completion_is_terminal_and_sanitized(failure):
    from judges.generic.structured_completion import StructuredOutputError
    row = dict(eligibility='claim', label='supported', uncertain=False, contradiction=False)
    output = {'C1': row}
    if failure == 'missing':
        output = {}
    elif failure in ('foreign', 'case'):
        output['C999' if failure == 'foreign' else 'c1'] = output.pop('C1')
    elif failure == 'id_field':
        row['id'] = 'synthetic-sensitive-value'
    elif failure == 'label':
        row['label'] = 'synthetic-sensitive-value'
    text = json.dumps(output)
    if failure == 'duplicate_key':
        text = '{"C1":' + json.dumps(row) + ',"C1":' + json.dumps(row) + '}'
    elif failure == 'duplicate_field':
        text = '{"C1":' + json.dumps(row)[:-1] + ',"label":"unsupported"}}'
    with pytest.raises(StructuredOutputError) as raised:
        keyed_result(text)
    assert raised.value.code == 'schema'
    assert 'synthetic-sensitive-value' not in str(raised.value)


@pytest.mark.parametrize('bad_ids', ['missing', 'duplicate', 'foreign'])
def test_completed_alias_id_failure_recovers_once_and_replays(tmp_path, bad_ids):
    # Without the recovery boundary the real judge stops before producing scores.
    # Replace only the external model; exercise the cache, mapping and pipeline.
    class BadInitial(CitationBackend):
        async def generate(self, request):
            result = await super().generate(request)
            if request.request_id == 'document-support-short-ids-v1':
                value = json.loads(result.text)
                if bad_ids == 'missing':
                    value['claims'] = []
                elif bad_ids == 'duplicate':
                    value['claims'].append(dict(value['claims'][0]))
                else:
                    value['claims'][0]['id'] = 'C999'
                return replace(result, text=json.dumps(value))
            return result

    backend = BadInitial(tmp_path / 'cache', [])
    result = run_judge(backend, tmp_path / 'live')
    result.verify(on_missing='error', expected_topic_ids=['t'])
    assert len(backend.citation_requests) == 2
    first, correction = backend.citation_requests
    schema = correction.extra['response_format']['json_schema']['schema']
    assert schema['required'] == ['C1']
    assert schema['additionalProperties'] is False
    # A request_id alone is not part of the production fingerprint.
    assert (first.messages, first.extra, first.max_tokens) != (
        correction.messages, correction.extra, correction.max_tokens)
    saved = json.loads((tmp_path / 'live/unified.stages.json').read_text())
    pair = saved['topics'][0]['evidence']['answers'][0]['evidence']['pairs'][0]
    assert pair['label'] == 'supported' and pair['text'] == 'Blue.'
    assert pair['document_id'] == 'd' and len(pair['id']) == 64
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    replay = run_judge(offline, tmp_path / 'offline')
    replay.verify(on_missing='error', expected_topic_ids=['t'])
    assert not offline.calls
    assert saved['topics'] == json.loads(
        (tmp_path / 'offline/unified.stages.json').read_text())['topics']
    assert receipts == {p.name: p.read_bytes() for p in
                        (tmp_path / 'cache/unified-completions').glob('*.json')}


def test_unresolved_id_correction_is_never_resent(tmp_path):
    class BadInitial(CitationBackend):
        async def generate(self, request):
            result = await super().generate(request)
            if request.request_id == 'document-support-short-ids-v1':
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'C999'
                return replace(result, text=json.dumps(value))
            return result

    backend = BadInitial(tmp_path / 'cache', [])
    run_judge(backend, tmp_path / 'first')
    from judges.generic.completion_session import CompletionSession
    session = CompletionSession(None, tmp_path / 'ledger', 3, factory=lambda _: backend)
    _, receipt, _ = session._cached_receipt(backend.citation_requests[1])
    # Delete only this synthetic receipt to emulate an unknown provider outcome.
    receipt.unlink()
    marker = receipt.with_suffix('.started')
    before = marker.read_bytes()
    calls = len(backend.citation_requests)
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path / 'blocked')
    assert len(backend.citation_requests) == calls
    assert marker.read_bytes() == before


def test_id_correction_has_context_headroom_guard(monkeypatch):
    import asyncio
    from minima_llm import MinimaLlmRequest, MinimaLlmResponse
    from judges.generic import citation_recovery
    from judges.generic.structured_completion import StructuredOutputError
    # The original fits, but adding the correction would exceed input+output.
    monkeypatch.setattr(citation_recovery, 'token_count', lambda _: 53761)

    class Backend:
        calls = 0

        async def generate(self, request):
            self.calls += 1
            return MinimaLlmResponse(request.request_id, '{"claims": []}',
                raw={'choices': [{'finish_reason': 'stop'}]})

    backend = Backend()
    with pytest.raises(StructuredOutputError) as raised:
        asyncio.run(citation_recovery.recover_citation_ids(backend,
            MinimaLlmRequest('document-support', [], max_tokens=32768),
            context_tokens=64000, max_output_tokens=32768,
            expected_ids=['C1']))
    assert raised.value.code == 'input_overflow'
    assert backend.calls == 0


def test_transport_failure_during_id_correction_remains_unresolved(tmp_path):
    from judges.generic.models import RunStopped

    class FailedCorrection(CitationBackend):
        async def generate(self, request):
            if request.request_id == 'document-support-keyed-ids-v1':
                self.citation_requests.append(request)
                raise RunStopped('transport')
            result = await super().generate(request)
            if request.request_id == 'document-support-short-ids-v1':
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'C999'
                return replace(result, text=json.dumps(value))
            return result

    backend = FailedCorrection(tmp_path / 'cache', [])
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path / 'first')
    assert len(backend.citation_requests) == 2
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    assert len(receipts) == 1
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path / 'blocked')
    assert len(backend.citation_requests) == 2
    assert receipts == {p.name: p.read_bytes() for p in
                        (tmp_path / 'cache/unified-completions').glob('*.json')}
