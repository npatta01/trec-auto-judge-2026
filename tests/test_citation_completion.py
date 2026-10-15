"""Completed citation repairs preserve original requests and never resend transport failures."""

import asyncio
import json
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmRequest, MinimaLlmResponse

from judges.generic.models import CitationStageError, JudgeError, RunStopped
from judges.generic.structured_completion import StructuredOutputError
from judges.generic.unified import UnifiedJudge
from tests.report_fixtures import report
from tests.test_unified_judge import SemanticBackend


def cited_reports():
    return [report([dict(text="Blue.", citations={"d": 100})],
                   {"d": dict(id="d", text="Blue.")})]


def run_judge(backend, outdir, **kwargs):
    kwargs.setdefault('citation_max_output_tokens', 32768)
    return UnifiedJudge(backend_factory=lambda _: backend).judge(
        cited_reports(), [Request(request_id="t", title="What color?", word_limit=20)],
        SimpleNamespace(), method="evidence", outdir=outdir, **kwargs)


class CitationBackend(SemanticBackend):
    def __init__(self, cache, failures, *, offline=False):
        super().__init__(cache, offline=offline)
        self.failures = list(failures)
        self.citation_requests = []

    async def generate(self, request):
        result = await super().generate(request)
        if request.extra['response_format']['json_schema']['name'] not in ('ClaimSupport', 'KeyedClaimSupport'):
            return result
        self.citation_requests.append(request)
        failure = self.failures.pop(0) if self.failures else None
        if failure == 'length':
            result.raw['choices'][0]['finish_reason'] = 'length'
        elif failure == 'missing':
            result = replace(result, raw={})
        elif failure == 'multiple':
            result.raw['choices'].append(dict(finish_reason='stop'))
        elif failure == 'refusal':
            result.raw['choices'][0]['message'] = dict(refusal='synthetic-sensitive-value')
        elif failure == 'schema':
            result = replace(result, text='{synthetic-sensitive-value')
        return result


@pytest.mark.parametrize('track', ['rag', 'ragtime'])
def test_completed_truncation_is_repaired_once_and_replays_without_network(tmp_path, track):
    backend = CitationBackend(tmp_path/'cache', ['length'])
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path/'legacy', citation_max_output_tokens=8192,
                  repair_citation_output=False)
    result = run_judge(backend, tmp_path/'live', track=track)
    result.verify(on_missing='error', expected_topic_ids=['t'])
    first, repaired = backend.citation_requests
    assert [first.max_tokens, repaired.max_tokens] == [8192, 32768]
    assert repaired.messages == first.messages
    assert repaired.extra == first.extra
    receipts = {p.name: p.read_bytes() for p in (tmp_path/'cache/unified-completions').glob('*.json')}
    assert len(receipts) >= 2
    offline = CitationBackend(tmp_path/'cache', [], offline=True)
    replay = run_judge(offline, tmp_path/'offline', track=track)
    replay.verify(on_missing='error', expected_topic_ids=['t'])
    assert not offline.calls
    assert {p.name: p.read_bytes() for p in (tmp_path/'cache/unified-completions').glob('*.json')} == receipts
    live = json.loads((tmp_path/'live/unified.stages.json').read_text())
    cached = json.loads((tmp_path/'offline/unified.stages.json').read_text())
    assert live['topics'] == cached['topics']


@pytest.mark.parametrize('failure,code,attempts', [
    ('length', 'truncated', 2), ('missing', 'incomplete', 1),
    ('multiple', 'incomplete', 1), ('refusal', 'incomplete', 1),
    ('schema', 'schema', 1),
])
def test_invalid_completion_stops_with_safe_category_and_bounded_calls(tmp_path, failure, code, attempts):
    backend = CitationBackend(tmp_path/'cache', [failure, failure])
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path/'legacy', citation_max_output_tokens=8192,
                  repair_citation_output=False)
    with pytest.raises(CitationStageError) as raised:
        run_judge(backend, tmp_path/'out')
    assert len(backend.citation_requests) == attempts
    assert raised.value.diagnostics['completion_code'] == code
    assert 'synthetic-sensitive' not in str(raised.value) + json.dumps(raised.value.diagnostics)
    assert not list((tmp_path/'out').glob('*.stages.json'))


def request():
    return MinimaLlmRequest('document-support', [], max_tokens=8192)


def response(finish='stop'):
    return MinimaLlmResponse('document-support', '{"claims": []}',
                             raw={'choices': [{'finish_reason': finish}]})


def complete(backend, **kwargs):
    from judges.generic.citation_completion import complete_citation
    return asyncio.run(complete_citation(backend, request(), context_tokens=64000,
                                        max_output_tokens=32768, **kwargs))


def test_original_success_uses_identical_request_only_once():
    class Backend:
        seen = []

        async def generate(self, req):
            self.seen.append(req)
            return response()

    backend = Backend()
    assert complete(backend).claims == []
    assert len(backend.seen) == 1
    assert backend.seen[0] == request()


def test_repair_respects_context_headroom(monkeypatch):
    from judges.generic import citation_completion
    monkeypatch.setattr(citation_completion, 'token_count', lambda _: 30000)

    class Backend:
        seen = []

        async def generate(self, req):
            self.seen.append(req)
            return response('length' if len(self.seen) == 1 else 'stop')

    backend = Backend()
    assert complete(backend).claims == []
    assert [r.max_tokens for r in backend.seen] == [8192, 31952]


def test_no_context_headroom_stops_without_repair(monkeypatch):
    from judges.generic import citation_completion
    monkeypatch.setattr(citation_completion, 'token_count', lambda _: 53760)

    class Backend:
        calls = 0

        async def generate(self, req):
            self.calls += 1
            return response('length')

    backend = Backend()
    with pytest.raises(StructuredOutputError) as raised:
        complete(backend)
    assert raised.value.code == 'truncated'
    assert backend.calls == 1


def test_diagnostic_mode_does_not_attempt_output_repair():
    class Backend:
        calls = 0

        async def generate(self, req):
            self.calls += 1
            return response('length')

    backend = Backend()
    with pytest.raises(StructuredOutputError) as raised:
        complete(backend, repair_output=False)
    assert raised.value.code == 'truncated'
    assert backend.calls == 1


def test_transport_failure_is_never_caught_or_resent():
    class Backend:
        calls = 0

        async def generate(self, req):
            self.calls += 1
            raise RunStopped('transport')

    backend = Backend()
    with pytest.raises(RunStopped) as raised:
        complete(backend)
    assert raised.value.code == 'transport'
    assert backend.calls == 1


def test_truncated_short_id_fallback_keeps_exact_mapping_and_cached_receipts(tmp_path):
    class WrongOriginalId(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.request_id == 'document-support':
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'wrong-original-id'
                return replace(result, text=json.dumps(value))
            return result

    backend = WrongOriginalId(tmp_path/'cache', [None, 'length'])
    from tests.citation_fixtures import seed_legacy_receipts
    seed_legacy_receipts(backend, tmp_path/'fixture-ledger', cited_reports())
    seed_legacy_receipts(backend, tmp_path/'fixture-ledger', cited_reports(), alias=True)
    run_judge(backend, tmp_path/'live')
    original, aliased, repaired = backend.citation_requests
    assert [r.max_tokens for r in backend.citation_requests] == [8192, 8192, 32768]
    assert aliased.request_id == 'document-support-short-ids-v1'
    assert repaired.request_id == 'document-support-short-ids-v1-output-repair-v1'
    original_id = json.loads(original.messages[1]['content'])['claims'][0]['id']
    assert original_id != 'C1'
    assert json.loads(aliased.messages[1]['content'])['claims'][0]['id'] == 'C1'
    assert repaired.messages == aliased.messages
    live = json.loads((tmp_path/'live/unified.stages.json').read_text())
    pair = live['topics'][0]['evidence']['answers'][0]['evidence']['pairs'][0]
    assert pair['label'] == 'supported'
    receipts = {p.name: p.read_bytes() for p in (tmp_path/'cache/unified-completions').glob('*.json')}
    assert len(receipts) >= 3
    offline = CitationBackend(tmp_path/'cache', [], offline=True)
    run_judge(offline, tmp_path/'offline')
    assert not offline.calls
    assert {p.name: p.read_bytes() for p in (tmp_path/'cache/unified-completions').glob('*.json')} == receipts
    replay = json.loads((tmp_path/'offline/unified.stages.json').read_text())
    assert live['topics'] == replay['topics']


@pytest.mark.parametrize('setting', [
    {'citation_max_output_tokens': 8191}, {'citation_max_output_tokens': 65537},
    {'citation_max_output_tokens': True}, {'repair_citation_output': 'yes'},
])
def test_invalid_settings_rejected_before_backend(tmp_path, setting):
    backend = CitationBackend(tmp_path/'cache', [])
    with pytest.raises(JudgeError, match='Invalid citation completion'):
        run_judge(backend, tmp_path/'out', **setting)
    assert not backend.calls
    assert not (tmp_path/'out').exists()


def test_fresh_citation_starts_with_larger_output_budget(tmp_path):
    backend = CitationBackend(tmp_path/'cache', [])
    run_judge(backend, tmp_path/'out')
    assert len(backend.citation_requests) == 1
    assert backend.citation_requests[0].max_tokens == 32768


def test_invalid_repaired_alias_ids_get_one_keyed_correction(tmp_path):
    class WrongRepairId(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if len(self.citation_requests) == 2:
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'wrong-repaired-id'
                return replace(result, text=json.dumps(value))
            return result

    backend = WrongRepairId(tmp_path/'cache', ['length'])
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path/'legacy', citation_max_output_tokens=8192,
                  repair_citation_output=False)
    result = run_judge(backend, tmp_path/'out')
    result.verify(on_missing='error', expected_topic_ids=['t'])
    assert len(backend.citation_requests) == 3
    assert backend.citation_requests[-1].request_id == 'document-support-keyed-ids-v1'
    offline = CitationBackend(tmp_path/'cache', [], offline=True)
    replay = run_judge(offline, tmp_path/'offline')
    replay.verify(on_missing='error', expected_topic_ids=['t'])
    assert not offline.calls


def test_unresolved_legacy_marker_does_not_authorize_larger_call(tmp_path):
    from judges.generic.completion_session import CompletionSession

    backend = CitationBackend(tmp_path/'cache', [])
    session = CompletionSession(None, tmp_path/'ledger', 1, factory=lambda _: backend)
    _, path, _ = session._cached_receipt(request())
    path.with_suffix('.started').write_text('historical-unresolved-marker')
    from judges.generic.citation_completion import complete_citation
    with pytest.raises(JudgeError, match='no automatic resend'):
        asyncio.run(complete_citation(session, request(), context_tokens=64000,
                                      max_output_tokens=65536))
    assert not backend.calls
    assert path.with_suffix('.started').read_text() == 'historical-unresolved-marker'


@pytest.mark.parametrize('raw', [{'usage': None}, {'usage': []}, {'usage': 'invalid'}, []])
def test_bad_cost_metadata_keeps_received_completion_and_unknown_hold(tmp_path, raw):
    from judges.generic.completion_session import CompletionSession

    class Backend:
        cfg = SimpleNamespace(cache_dir=str(tmp_path/'cache'), model='fixture',
            base_url='https://openrouter.ai/api/v1', force_refresh=False, max_attempts=1)
        calls = 0

        async def generate(self, req):
            self.calls += 1
            return replace(response(), raw=raw)

        async def aclose(self):
            pass

    backend = Backend()
    session = CompletionSession(None, tmp_path/'ledger', 1, factory=lambda _: backend)
    received = asyncio.run(session.generate(request()))
    assert received.text == '{"claims": []}'
    assert len(list(session.directory.glob('*.json'))) == 1
    assert session.summary()['unknown_cost_calls'] == 1
    before = session.summary()
    assert asyncio.run(session.generate(request())).cached
    assert session.summary() == before
    assert backend.calls == 1


def test_default_budget_admits_larger_initial_call_through_real_guard(tmp_path):
    import yaml
    from judges.generic.completion_session import CompletionSession

    settings = yaml.safe_load((Path(__file__).parents[1]/'judges/generic/unified-workflow.yml').read_text())['settings']
    default = inspect.signature(UnifiedJudge.judge).parameters['run_budget_usd'].default
    assert default == settings['run_budget_usd']

    class Backend:
        cfg = SimpleNamespace(cache_dir=str(tmp_path/'cache'), model='fixture',
            base_url='https://openrouter.ai/api/v1', force_refresh=False, max_attempts=1)
        seen = []

        async def generate(self, req):
            self.seen.append(req)
            assert req.extra['provider']['max_price']['completion'] == 10
            return replace(response(), raw=dict(choices=[dict(finish_reason='stop')],
                                               usage=dict(cost=0.001)))

        async def aclose(self):
            pass

    from judges.generic.citation_completion import complete_citation
    backend = Backend()
    session = CompletionSession(None, tmp_path/'ledger', default, factory=lambda _: backend)
    result = asyncio.run(complete_citation(session, request(), context_tokens=64000,
        max_output_tokens=settings['citation_max_output_tokens'], expected_ids=set()))
    assert result.claims == []
    assert [r.max_tokens for r in backend.seen] == [32768]
    assert session.summary()['calls'] == 1
    assert session.summary()['reported_usd'] == 0.001
    assert session.summary()['unknown_cost_calls'] == 0
