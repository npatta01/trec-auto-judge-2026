"""Exact identifier plumbing without exposing source fingerprints to new calls."""

import json
from dataclasses import replace

import pytest
from autojudge_base import Request
from types import SimpleNamespace

from judges.generic.models import CitationStageError
from judges.generic.completion_session import CompletionSession
from judges.generic.unified import UnifiedJudge
from tests.report_fixtures import report
from tests.citation_fixtures import seed_legacy_receipts
from tests.test_citation_completion import CitationBackend, cited_reports, run_judge


def seed_legacy_receipt(backend, tmp_path, *, repair=False):
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', cited_reports(), repair=repair)


def test_valid_legacy_receipt_keeps_original_request_and_judgment(tmp_path):
    backend = CitationBackend(tmp_path / 'cache', [])
    seed_legacy_receipt(backend, tmp_path)
    first_request = backend.citation_requests[0]
    assert first_request.max_tokens == 8192
    assert len(json.loads(first_request.messages[1]['content'])['claims'][0]['id']) == 64
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    run_judge(backend, tmp_path / 'live')
    assert len(backend.citation_requests) == 1
    assert all((tmp_path / 'cache/unified-completions' / name).read_bytes() == data
               for name, data in receipts.items())


@pytest.mark.parametrize('alias', [False, True])
def test_cached_32k_repair_replays_when_output_ceiling_increases(tmp_path, alias):
    # Raising the permitted ceiling must not spend again or break offline replay
    # when the previous completed repair is already usable.
    backend = CitationBackend(tmp_path / 'cache', ['length'])
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', cited_reports(),
                         alias=alias, repair=True)
    assert [r.max_tokens for r in backend.citation_requests] == [8192, 32768]
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    result = run_judge(backend, tmp_path / 'live', citation_max_output_tokens=65536)
    result.verify(on_missing='error', expected_topic_ids=['t'])
    assert len(backend.citation_requests) == 2
    assert all((tmp_path / 'cache/unified-completions' / name).read_bytes() == data
               for name, data in receipts.items())
    # Reference and grading receipts are new on the first full workflow pass.
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    run_judge(offline, tmp_path / 'replay', citation_max_output_tokens=65536)
    assert not offline.calls
    live = json.loads((tmp_path / 'live/unified.stages.json').read_text())
    replay = json.loads((tmp_path / 'replay/unified.stages.json').read_text())
    assert live['topics'] == replay['topics']
    assert {p.name: p.read_bytes() for p in
            (tmp_path / 'cache/unified-completions').glob('*.json')} == receipts


def test_cached_larger_legacy_initial_still_reuses_original_ids(tmp_path):
    backend = CitationBackend(tmp_path / 'cache', [])
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', cited_reports(), max_tokens=32768)
    run_judge(backend, tmp_path / 'live', citation_max_output_tokens=65536)
    assert len(backend.citation_requests) == 1
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    run_judge(offline, tmp_path / 'offline', citation_max_output_tokens=65536)
    assert not offline.calls


@pytest.mark.parametrize('alias,repair', [(False, False), (True, False), (False, True), (True, True)])
def test_unresolved_larger_initial_or_old_repair_blocks_new_namespace(tmp_path, alias, repair):
    backend = CitationBackend(tmp_path / 'cache', ['length'] if repair else [])
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', cited_reports(),
                         alias=alias, repair=repair, max_tokens=8192 if repair else 32768)
    session = CompletionSession(None, tmp_path / 'fixture-ledger', 3, factory=lambda _: backend)
    _, path, _ = session._cached_receipt(backend.citation_requests[-1])
    # Only synthetic fixture receipts are removed to emulate an unacknowledged
    # provider result; the production cache is never involved.
    path.unlink()
    marker = path.with_suffix('.started')
    before = marker.read_bytes()
    count = len(backend.citation_requests)
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path / 'blocked', citation_max_output_tokens=65536)
    assert len(backend.citation_requests) == count
    assert marker.read_bytes() == before
    assert not list((tmp_path / 'blocked').glob('*.stages.json'))


@pytest.mark.parametrize('bad_alias_repair', [False, True])
def test_legacy_and_alias_repairs_use_at_most_one_keyed_recovery(tmp_path, bad_alias_repair):
    class RepairIds(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.request_id == 'document-support-output-repair-v1' or (
                    bad_alias_repair and req.request_id == 'document-support-short-ids-v1-output-repair-v1'):
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'foreign'
                result = replace(result, text=json.dumps(value))
            return result

    backend = RepairIds(tmp_path / 'cache', ['length', None, 'length', None])
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', cited_reports(), repair=True)
    run_judge(backend, tmp_path / 'live', citation_max_output_tokens=65536)
    assert len(backend.citation_requests) == (5 if bad_alias_repair else 4)
    assert [r.request_id for r in backend.citation_requests] == [
        'document-support', 'document-support-output-repair-v1',
        'document-support-short-ids-v1', 'document-support-short-ids-v1-output-repair-v1'
    ] + (['document-support-keyed-ids-v1'] if bad_alias_repair else [])
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    run_judge(offline, tmp_path / 'offline', citation_max_output_tokens=65536)
    assert not offline.calls


def test_reordered_alias_verdicts_restore_exact_claims_not_output_positions(tmp_path):
    class Reordered(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.extra['response_format']['json_schema']['name'] == 'ClaimSupport':
                value = json.loads(result.text)
                value['claims'][1]['label'] = 'partially_supported'
                value['claims'].reverse()
                result = replace(result, text=json.dumps(value))
            return result

    reports = [report([dict(text='Blue.', citations={'d': 100}),
                       dict(text='Round.', citations={'d': 100})],
                      {'d': dict(id='d', text='Blue and slightly curved.')})]
    backend = Reordered(tmp_path / 'cache', [])
    UnifiedJudge(backend_factory=lambda _: backend).judge(
        reports, [Request(request_id='t', title='Describe color and shape.', word_limit=20)],
        SimpleNamespace(), method='evidence', outdir=tmp_path / 'live')
    payload = json.loads(backend.citation_requests[0].messages[1]['content'])
    assert [c['id'] for c in payload['claims']] == ['C1', 'C2']
    saved = json.loads((tmp_path / 'live/unified.stages.json').read_text())
    pairs = saved['topics'][0]['evidence']['answers'][0]['evidence']['pairs']
    assert [(p['text'], p['sentence_index'], p['label']) for p in pairs] == [
        ('Blue.', 0, 'supported'), ('Round.', 1, 'partially_supported')]
    assert len({p['id'] for p in pairs}) == 2
    assert all(len(p['id']) == 64 and p['document_id'] == 'd' for p in pairs)


def test_saved_invalid_completed_repair_recovers_once_with_exact_aliases(tmp_path):
    class WrongRepair(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.request_id == 'document-support-output-repair-v1':
                value = json.loads(result.text)
                value['claims'][0]['id'] = 'wrong-repaired-id'
                return replace(result, text=json.dumps(value))
            return result

    backend = WrongRepair(tmp_path / 'cache', ['length'])
    seed_legacy_receipt(backend, tmp_path, repair=True)
    assert len(backend.citation_requests) == 2
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    result = run_judge(backend, tmp_path / 'recovered')
    result.verify(on_missing='error', expected_topic_ids=['t'])
    assert len(backend.citation_requests) == 3
    alias_request = backend.citation_requests[-1]
    assert [c['id'] for c in json.loads(alias_request.messages[1]['content'])['claims']] == ['C1']
    assert all((tmp_path / 'cache/unified-completions' / name).read_bytes() == data
               for name, data in receipts.items())
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    run_judge(offline, tmp_path / 'offline')
    assert not offline.calls
    first = json.loads((tmp_path / 'recovered/unified.stages.json').read_text())
    replay = json.loads((tmp_path / 'offline/unified.stages.json').read_text())
    assert first['topics'] == replay['topics']


def test_fresh_citation_uses_request_local_ids_and_restores_source_links(tmp_path):
    # This fails if new network calls still ask the model to reproduce hash IDs.
    # The fake replaces only the external model; the real cache, judge, mapping,
    # document aggregation, reference and scoring pipeline all execute.
    backend = CitationBackend(tmp_path / 'cache', [])
    result = run_judge(backend, tmp_path / 'live')
    result.verify(on_missing='error', expected_topic_ids=['t'])
    payload = json.loads(backend.citation_requests[0].messages[1]['content'])
    assert [claim['id'] for claim in payload['claims']] == ['C1']
    saved = json.loads((tmp_path / 'live/unified.stages.json').read_text())
    pair = saved['topics'][0]['evidence']['answers'][0]['evidence']['pairs'][0]
    assert len(pair['id']) == 64
    assert pair['label'] == 'supported'
    assert pair['sentence_index'] == 0
    assert pair['document_id'] == 'd'
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    run_judge(offline, tmp_path / 'replay')
    assert not offline.calls
    replay = json.loads((tmp_path / 'replay/unified.stages.json').read_text())
    assert saved['topics'] == replay['topics']
    assert {p.name: p.read_bytes() for p in
            (tmp_path / 'cache/unified-completions').glob('*.json')} == receipts


@pytest.mark.parametrize('bad_ids', ['missing', 'duplicate', 'foreign'])
def test_invalid_alias_correction_stops_without_guessing_or_more_calls(tmp_path, bad_ids):
    class InvalidAliases(CitationBackend):
        async def generate(self, req):
            result = await super().generate(req)
            if req.request_id not in ('document-support-short-ids-v1', 'document-support-keyed-ids-v1'):
                return result
            value = json.loads(result.text)
            if req.request_id == 'document-support-keyed-ids-v1':
                value['C999'] = value.pop('C1')
                return replace(result, text=json.dumps(value))
            if bad_ids == 'missing':
                value['claims'] = []
            elif bad_ids == 'duplicate':
                value['claims'].append(dict(value['claims'][0]))
            else:
                value['claims'][0]['id'] = 'C999'
            return replace(result, text=json.dumps(value))

    backend = InvalidAliases(tmp_path / 'cache', [])
    with pytest.raises(CitationStageError):
        run_judge(backend, tmp_path / 'out')
    assert len(backend.citation_requests) == 2
    assert not list((tmp_path / 'out').glob('*.stages.json'))
    receipts = {p.name: p.read_bytes() for p in
                (tmp_path / 'cache/unified-completions').glob('*.json')}
    offline = CitationBackend(tmp_path / 'cache', [], offline=True)
    with pytest.raises(CitationStageError):
        run_judge(offline, tmp_path / 'offline')
    assert not offline.calls
    assert {p.name: p.read_bytes() for p in
            (tmp_path / 'cache/unified-completions').glob('*.json')} == receipts
