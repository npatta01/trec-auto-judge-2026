import json
from types import SimpleNamespace
from minima_llm import MinimaLlmResponse
from autojudge_base import Request
from tests.report_fixtures import report
import pytest


class Backend:
    async def generate(self, req):
        payload = json.loads(req.messages[1]["content"])
        return MinimaLlmResponse(
            request_id=req.request_id,
            text=json.dumps(
                {
                    "claims": [
                        dict(
                            id=c["id"],
                            eligibility="claim",
                            label="supported",
                            uncertain=False,
                            contradiction=False,
                        )
                        for c in payload["claims"]
                    ]
                }
            ),
        )

    async def aclose(self):
        pass


def test_framework_adapter_exports_scores_and_provenance(tmp_path):
    from judges.generic.document_judge import DocumentJudge

    judge = DocumentJudge(backend_factory=lambda _: Backend())
    board = judge.judge(
        [report()],
        [Request(request_id="t", title="Test")],
        SimpleNamespace(),
        outdir=tmp_path,
    )
    assert board is not None
    result = json.loads((tmp_path / "document.support.json").read_text())
    assert result["answers"][0]["score"] == 1
    assert (tmp_path / "document.support.json").stat().st_mode & 0o777 == 0o600
    assert (
        tmp_path / "document.supported-claims.jsonl"
    ).stat().st_mode & 0o777 == 0o600
    assert len(result["supported_claims"]) == 2
    assert (
        len((tmp_path / "document.supported-claims.jsonl").read_text().splitlines())
        == 2
    )


def test_adapter_retains_exclusions_without_exporting_them(tmp_path):
    from judges.generic.document_judge import DocumentJudge

    class ExclusionBackend(Backend):
        async def generate(self, req):
            payload = json.loads(req.messages[1]["content"])
            result = {
                "claims": [
                    dict(
                        id=c["id"],
                        eligibility="incomplete",
                        label="unsupported",
                        uncertain=False,
                        contradiction=False,
                    )
                    for c in payload["claims"]
                ]
            }
            return MinimaLlmResponse(request_id=req.request_id, text=json.dumps(result))

    judge = DocumentJudge(backend_factory=lambda _: ExclusionBackend())
    a = report(
        [{"text": "The study found that 73% of peo", "citations": {"d": 100}}],
        {"d": {"id": "d", "text": "A complete source."}},
    )
    board = judge.judge(
        [a], [Request(request_id="t", title="Test")], SimpleNamespace(), outdir=tmp_path
    )
    assert board is not None
    result = json.loads((tmp_path / "document.support.json").read_text())
    assert result["answers"][0]["score"] is None
    assert result["answers"][0]["excluded_pairs"] == 1
    assert result["pairs"][0]["eligibility"] == "incomplete"
    assert (tmp_path / "document.supported-claims.jsonl").read_text() == ""


def test_document_cli_http_roundtrip_and_offline_replay(tmp_path):
    import subprocess, sys, os, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body["model"])
            data = json.loads(body["messages"][1]["content"])
            result = {
                "claims": [
                    dict(
                        id=c["id"],
                        eligibility="claim",
                        label="supported",
                        uncertain=False,
                        contradiction=False,
                    )
                    for c in data["claims"]
                ]
            }
            wire = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": json.dumps(result)},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "r.jsonl").write_text(report().model_dump_json(exclude_none=True) + "\n")
    topics = tmp_path / "topics.jsonl"
    topics.write_text(Request(request_id="t", title="Test").model_dump_json() + "\n")
    env = dict(
        os.environ,
        OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
        OPENAI_MODEL="injected-test-model",
        OPENAI_API_KEY="test-only",
        CACHE_DIR=str(tmp_path / "cache"),
        MAX_ATTEMPTS="1",
        TIMEOUT_S="5",
        RPM="0",
        MINIMA_DEBUG="",
        MINIMA_TRACE_FILE="",
        CACHE_FORCE_REFRESH="0",
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "autojudge_base.cli",
                "run",
                "--workflow",
                "judges/generic/workflow.yml",
                "--rag-responses",
                str(runs),
                "--rag-topics",
                str(topics),
                "--out-dir",
                str(tmp_path / "out"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert seen and set(seen) == {"injected-test-model"}
    assert list((tmp_path / "out").glob("*.eval.txt"))
    assert list((tmp_path / "out").glob("*.supported-claims.jsonl"))
    # The server is closed: the identical request must replay from the cache.
    env.update(OPENAI_BASE_URL="EMPTY", OPENAI_API_KEY="EMPTY")
    replay_command = list(proc.args)
    replay_command[-1] = str(tmp_path / "offline")
    replay = subprocess.run(
        replay_command, env=env, capture_output=True, text=True, timeout=60
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    for suffix in (".eval.txt", ".supported-claims.jsonl", ".support.json"):
        online = next((tmp_path / "out").glob("*" + suffix))
        offline = next((tmp_path / "offline").glob("*" + suffix))
        assert online.read_text() == offline.read_text()


@pytest.mark.parametrize("limit, succeeds", [(0.001, False), (0.5, True)])
def test_normal_workflow_enforces_openrouter_budget(tmp_path, limit, succeeds):
    from judges.generic.document_judge import DocumentJudge
    from judges.generic.models import JudgeError

    class PaidBackend(Backend):
        cfg = SimpleNamespace(
            base_url="https://openrouter.ai/api/v1", max_attempts=1, model="test-model"
        )
        hits = 0
        closed = False

        async def generate(self, req):
            self.hits += 1
            assert req.extra["reasoning"] == {"effort": "high"}
            assert req.extra["provider"]["max_price"]["completion"] == 4
            return await super().generate(req)

        async def aclose(self):
            self.closed = True

    backend = PaidBackend()
    judge = DocumentJudge(backend_factory=lambda _: backend)

    def run():
        return judge.judge(
            [report()],
            [Request(request_id="t", title="Test")],
            SimpleNamespace(),
            outdir=tmp_path,
            budget_ledger=str(tmp_path / "budget.db"),
            run_budget_usd=limit,
        )

    if succeeds:
        assert run() is not None
    else:
        with pytest.raises(JudgeError):
            run()
        assert backend.hits == 0
    assert backend.closed
    saved = json.loads((tmp_path / "document.support.json").read_text())
    assert saved["budget"]["calls"] == backend.hits
    if not succeeds:
        assert saved["run_failure"] == "budget_exhausted"


def test_transport_failure_stops_and_saves_only_safe_diagnostics(tmp_path):
    from judges.generic.document_judge import DocumentJudge
    from judges.generic.models import JudgeError

    class FailingBackend(Backend):
        hits = 0
        closed = False

        async def generate(self, req):
            self.hits += 1
            raise RuntimeError("PRIVATE_SENTINEL")

        async def aclose(self):
            self.closed = True

    backend = FailingBackend()
    with pytest.raises(JudgeError):
        DocumentJudge(backend_factory=lambda _: backend).judge(
            [report()],
            [Request(request_id="t", title="Test")],
            SimpleNamespace(),
            outdir=tmp_path,
        )
    assert backend.hits == 1 and backend.closed
    saved = (tmp_path / "document.support.json").read_text()
    assert json.loads(saved)["run_failure"] == "transport"
    assert "PRIVATE_SENTINEL" not in saved
