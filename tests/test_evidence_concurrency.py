import asyncio
import json
from types import SimpleNamespace

from autojudge_base import Request
from minima_llm import MinimaLlmResponse
import pytest

from tests.report_fixtures import report
from tests.test_unified_judge import SemanticBackend


def reports(count=4):
    result = []
    for i in range(count):
        doc = f'd{i}'
        r = report([dict(text='Blue.', citations={doc: 100})],
                   {doc: dict(id=doc, text=f'Blue. Source {i}.')})
        r.metadata.run_id = f'run{i}'
        result.append(r)
    return result


class SlowBackend(SemanticBackend):
    def __init__(self, cache, offline=False):
        super().__init__(cache, offline)
        self.active = self.peak = 0
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001)
            return await super().generate(request)
        finally:
            self.active -= 1


@pytest.mark.parametrize('count,concurrency', [(4, 2), (40, 32)])
def test_evidence_runs_concurrently_and_replays_serially_without_rejudging(tmp_path, count, concurrency):
    from judges.generic.unified import UnifiedJudge

    live = SlowBackend(tmp_path / 'cache')
    args = dict(method='evidence', track='rag', coverage_mode='separate')
    question = Request(request_id='t', title='What color?', word_limit=20)
    UnifiedJudge(backend_factory=lambda _: live).judge(
        reports(count), [question], SimpleNamespace(), outdir=tmp_path / 'parallel',
        evidence_concurrency=concurrency, **args)
    assert live.peak == concurrency
    assert live.active == 0
    replay = SlowBackend(tmp_path / 'cache', offline=True)
    UnifiedJudge(backend_factory=lambda _: replay).judge(
        reports(count), [question], SimpleNamespace(), outdir=tmp_path / 'serial',
        evidence_concurrency=1, **args)
    assert replay.requests == []
    first = json.loads((tmp_path / 'parallel/unified.stages.json').read_text())
    second = json.loads((tmp_path / 'serial/unified.stages.json').read_text())
    assert first['topics'] == second['topics']


def test_bad_legacy_claim_ids_get_exact_alias_repair_and_valid_calls_are_kept(tmp_path):
    from judges.generic.unified import UnifiedJudge

    class BadIds(SlowBackend):
        async def generate(self, request):
            response = await super().generate(request)
            if request.request_id == 'document-support':
                value = json.loads(response.text)
                # Only one legacy batch has a corrupted ID. All others are valid.
                if len([r for r in self.requests if r.request_id == 'document-support']) == 1:
                    value['claims'][0]['id'] = 'not-the-requested-id'
                    response = MinimaLlmResponse(request.request_id, json.dumps(value), raw=response.raw)
            return response

    backend = BadIds(tmp_path / 'cache')
    from tests.citation_fixtures import seed_legacy_receipts
    seed_legacy_receipts(backend, tmp_path / 'fixture-ledger', reports())
    args = dict(method='evidence', track='rag', coverage_mode='separate')
    question = Request(request_id='t', title='What color?', word_limit=20)
    judge = UnifiedJudge(backend_factory=lambda _: backend)
    judge.judge(reports(), [question], SimpleNamespace(), outdir=tmp_path / 'out', **args)
    aliases = [r for r in backend.requests if r.request_id == 'document-support-short-ids-v1']
    assert len(aliases) == 1
    assert json.loads(aliases[0].messages[1]['content'])['claims'][0]['id'] == 'C1'
    stage = json.loads((tmp_path / 'out/unified.stages.json').read_text())
    pairs = stage['topics'][0]['evidence']['citation_audit']['pairs']
    assert len(pairs) == 4 and all(p['status'] == 'complete' for p in pairs)
    assert all(len(p['id']) == 64 for p in pairs)
    offline = SlowBackend(tmp_path / 'cache', offline=True)
    UnifiedJudge(backend_factory=lambda _: offline).judge(
        reports(), [question], SimpleNamespace(), outdir=tmp_path / 'replay', **args)
    assert offline.requests == []


def test_bad_alias_ids_stop_without_guessing_or_repeated_repairs(tmp_path):
    from judges.generic.models import JudgeError
    from judges.generic.unified import UnifiedJudge

    class WrongEveryTime(SlowBackend):
        async def generate(self, request):
            response = await super().generate(request)
            if request.request_id.startswith('document-support'):
                value = json.loads(response.text)
                if request.request_id == 'document-support-keyed-ids-v1':
                    value['unknown-id'] = value.pop('C1')
                    return MinimaLlmResponse(request.request_id, json.dumps(value), raw=response.raw)
                value['claims'][0]['id'] = 'unknown-id'
                return MinimaLlmResponse(request.request_id, json.dumps(value), raw=response.raw)
            return response

    backend = WrongEveryTime(tmp_path / 'cache')
    with pytest.raises(JudgeError, match='Citation stage incomplete'):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports(), [Request(request_id='t', title='What color?', word_limit=20)],
            SimpleNamespace(), outdir=tmp_path / 'out', method='evidence')
    assert [r.request_id for r in backend.requests] == [
        'document-support-short-ids-v1',
        'document-support-keyed-ids-v1']


def test_concurrent_provider_failure_output_is_suppressed(tmp_path, capsys):
    from judges.generic.models import JudgeError
    from judges.generic.unified import UnifiedJudge
    import sys

    class NoisyFailure(SlowBackend):
        async def generate(self, request):
            await asyncio.sleep(0.001)
            print('synthetic-provider-body')
            print('synthetic-provider-body', file=sys.stderr)
            raise RuntimeError('synthetic provider failure')

    backend = NoisyFailure(tmp_path / 'cache')
    with pytest.raises(JudgeError):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports(), [Request(request_id='t', title='What color?', word_limit=20)],
            SimpleNamespace(), outdir=tmp_path / 'out', method='evidence',
            evidence_concurrency=2)
    captured = capsys.readouterr()
    assert 'synthetic-provider-body' not in captured.out + captured.err
