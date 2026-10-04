"""Real CLI + HTTP + SQLite replay; all content is manufactured test data."""
import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from minima_llm import MinimaLlmRequest

from tests.test_generic_judge import report
from tests.test_generic_pipeline import SyntheticBackend


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / 'judges/generic/workflow.yml'


@pytest.mark.parametrize('mode,expected_calls', [('direct', 1), ('staged', 4), ('staged_spans', 4), ('staged_spans_openrouter', 4)])
def test_cli_and_offline_replay(tmp_path, mode, expected_calls):
    backend = SyntheticBackend()
    models = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            models.append(data['model'])
            result = asyncio.run(backend.generate(MinimaLlmRequest('test', data['messages'])))
            payload = json.dumps({'choices': [{'message': {'content': result.text},
                                              'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'synthetic.jsonl').write_text(report().model_dump_json(exclude_none=True) + '\n')
    topics = tmp_path / 'topics.jsonl'
    topics.write_text(json.dumps(dict(request_id='t', title='Compare the sample', background='For a school')) + '\n')
    env = dict(os.environ, OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
               OPENAI_MODEL='synthetic-injected-model', OPENAI_API_KEY='synthetic-test-only',
               CACHE_DIR=str(tmp_path / 'cache'), MAX_ATTEMPTS='1', TIMEOUT_S='2', RPM='0',
               CACHE_FORCE_REFRESH='0', MINIMA_DEBUG='', MINIMA_TRACE_FILE='')
    command = [sys.executable, '-B', '-m', 'autojudge_base.cli', 'run', '--workflow', str(WORKFLOW),
               '--variant', mode, '--rag-responses', str(runs), '--rag-topics', str(topics)]
    try:
        online = subprocess.run(command + ['--out-dir', str(tmp_path / 'online')], cwd=REPO,
                                env=env, capture_output=True, text=True, timeout=60)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert online.returncode == 0, online.stdout + online.stderr
    assert models == ['synthetic-injected-model'] * expected_calls
    env['OPENAI_BASE_URL'] = 'EMPTY'
    env['OPENAI_API_KEY'] = 'EMPTY'
    offline = subprocess.run(command + ['--out-dir', str(tmp_path / 'offline')], cwd=REPO,
                             env=env, capture_output=True, text=True, timeout=60)
    assert offline.returncode == 0, offline.stdout + offline.stderr
    first = list((tmp_path / 'online').glob('*.eval.txt'))
    second = list((tmp_path / 'offline').glob('*.eval.txt'))
    assert len(first) == len(second) == 1
    assert first[0].read_text() == second[0].read_text()
    assert 'USEFULNESS' in first[0].read_text()
    assert '0.75' in first[0].read_text()
    assert list((tmp_path / 'online').glob('*.eval.measures.yml'))
    assert list((tmp_path / 'online').glob('*.config.yml'))


def test_generic_empty_cli(tmp_path):
    # Explicitly cover the untracked workflow; organizer discovery uses git ls-files.
    from tests.test_empty_reports import test_judge_scores_empty_report
    test_judge_scores_empty_report(WORKFLOW, tmp_path)
