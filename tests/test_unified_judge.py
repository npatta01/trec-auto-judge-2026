import pytest
import json
from types import SimpleNamespace
from minima_llm import MinimaLlmResponse
from autojudge_base import Request
from tests.report_fixtures import report


@pytest.mark.parametrize("phase", ["pairwise", "semantic"])
def test_stage_preserves_safe_offline_error(phase):
    import asyncio
    from minima_llm import MinimaLlmRequest
    from judges.generic.models import JudgeError
    from judges.generic.pairwise import _call
    from judges.generic.shared_semantics import prepare_topic

    class Offline:
        cfg = SimpleNamespace(model="test")

        async def generate(self, request):
            raise JudgeError("Offline completion cache miss.")

    task = (
        _call(MinimaLlmRequest("test", []), Offline(), None)
        if phase == "pairwise"
        else prepare_topic(
            dict(answers=[], pool=dict(question={}, evidence=[])), Offline(), None
        )
    )
    with pytest.raises(JudgeError, match="^Offline completion cache miss"):
        asyncio.run(task)


def test_fusion_is_fixed_bounded_and_track_specific():
    from judges.generic.unified import fuse

    values = dict(
        RAG_REQUEST_COVERAGE_PROXY=1.0,
        RAG_CITATION_PRECISION_UNWEIGHTED_PROXY=0.5,
        RAG_CITATION_RECALL_UNWEIGHTED_PROXY=0.0,
    )
    evidence, combined = fuse(values, 0.75, "rag", 0.5)
    assert evidence == 0.5
    assert combined == 0.625
    with pytest.raises(ValueError):
        fuse(values, 0.75, "rag", 2)


def test_ragtime_partial_grounding_changes_main_score():
    from judges.generic.unified import fuse

    values = {
        "RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE": 0.0,
        "RAGTIME_GROUNDED_COVERAGE_PARTIAL_PROXY": 0.5,
        "RAGTIME_SENTENCE_SUPPORT_PROXY": 0.25,
    }
    assert fuse(values, 0.75, "ragtime", 0.5) == (0.375, 0.5625)


def test_reordered_semantic_answers_cannot_be_scored_as_other_runs(
    tmp_path, monkeypatch
):
    from judges.generic import unified
    from judges.generic.models import JudgeError

    original = unified.prepare_topic

    async def reordered(*args, **kwargs):
        topic = await original(*args, **kwargs)
        topic["answers"].reverse()
        return topic

    monkeypatch.setattr(unified, "prepare_topic", reordered)
    reports = [
        report(
            [dict(text="Blue.", citations={"d": 100})],
            {"d": dict(id="d", text="Blue.")},
        )
        for _ in range(2)
    ]
    reports[1].metadata.run_id = "second"
    backend = SemanticBackend(tmp_path / "cache")
    with pytest.raises(JudgeError, match="Answer/run alignment"):
        unified.UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            [Request(request_id="t", title="What color?", word_limit=20)],
            SimpleNamespace(),
            outdir=tmp_path / "out",
            method="evidence",
        )


def test_bridge_keeps_partial_evidence_and_deduplicates_pool():
    from judges.generic.evidence_bridge import build_topic
    from judges.generic.document_pipeline import prepare_reports

    r = report(
        [dict(text="Blue and round.", citations={"d": 100})],
        {"d": dict(id="d", text="Blue.")},
    )
    prepared = prepare_reports([r])
    p = prepared.pairs[0]
    p.update(
        status="complete",
        eligibility="claim",
        chunks=[dict(chunk_id="c", start=0, end=5)],
        aggregation=dict(
            complete=True,
            label="partially_supported",
            uncertain=False,
            conflict=False,
            winning_chunk_ids=["c"],
        ),
    )
    topic = build_topic(
        [r],
        Request(request_id="t", title="Describe it"),
        dict(pairs=[p], run_failure=None),
    )
    assert topic["answers"][0]["evidence"]["pairs"][0]["label"] == "partially_supported"
    assert topic["pool"]["evidence"][0]["excerpt"] == "Blue."
    assert topic["answers"][0]["evidence"]["sentences"][0]["eligible"] is None


def test_bridge_does_not_promote_conflicting_support():
    from judges.generic.evidence_bridge import build_topic
    from judges.generic.document_pipeline import prepare_reports

    r = report(
        [dict(text="Blue.", citations={"d": 100})], {"d": dict(id="d", text="Blue.")}
    )
    p = prepare_reports([r]).pairs[0]
    p.update(
        status="complete",
        eligibility="claim",
        chunks=[dict(chunk_id="c", start=0, end=5)],
        aggregation=dict(
            complete=True,
            label="supported",
            uncertain=True,
            conflict=True,
            winning_chunk_ids=["c"],
        ),
    )
    t = build_topic(
        [r],
        Request(request_id="t", title="Describe"),
        dict(pairs=[p], run_failure=None),
    )
    assert not t["pool"]["evidence"]
    assert t["answers"][0]["evidence"]["pairs"][0]["uncertain"]


def test_pool_stores_shared_document_once_without_losing_claims():
    from judges.generic.evidence_bridge import build_topic
    from judges.generic.document_pipeline import prepare_reports

    r = report(
        [
            dict(text="Blue.", citations={"d": 100}),
            dict(text="Round.", citations={"d": 100}),
        ],
        {"d": dict(id="d", text="Blue and round.")},
    )
    ps = prepare_reports([r]).pairs
    for p in ps:
        p.update(
            status="complete",
            eligibility="claim",
            chunks=[dict(chunk_id="c", start=0, end=15)],
            aggregation=dict(
                complete=True,
                label="supported",
                uncertain=False,
                conflict=False,
                winning_chunk_ids=["c"],
            ),
        )
    topic = build_topic(
        [r], Request(request_id="t", title="Describe"), dict(pairs=ps, run_failure=None)
    )
    assert len(topic["pool"]["evidence"]) == 1
    assert len(topic["pool"]["evidence"][0]["claims"]) == 2
    assert len(topic["answers"][0]["evidence"]["pairs"]) == 2


class SemanticBackend:
    def __init__(self, cache, offline=False):
        self.cfg = SimpleNamespace(
            cache_dir=str(cache),
            model="synthetic-model",
            base_url="EMPTY" if offline else "http://localhost",
            force_refresh=False,
        )
        self.calls = []

    async def generate(self, req):
        assert self.cfg.base_url != "EMPTY", "Replay attempted a model call"
        name = req.extra["response_format"]["json_schema"]["name"]
        self.calls.append(name)
        p = json.loads(req.messages[1]["content"])
        if name == "ClaimSupport":
            value = dict(
                claims=[
                    dict(
                        id=c["id"],
                        eligibility="claim",
                        label="supported",
                        uncertain=False,
                        contradiction=False,
                    )
                    for c in p["claims"]
                ]
            )
        elif name == "ReferenceChecklist":
            eid = p["evidence"][0]["id"]
            value = dict(
                answer=[dict(text="Blue.", evidence_ids=[eid])],
                checklist=[
                    dict(
                        id="N1",
                        question="What color?",
                        requirement="State the color.",
                        acceptable_alternatives=[],
                        evidence_ids=[eid],
                        request_basis="Requested color.",
                        kind="factual",
                        priority="core",
                    )
                ],
                gaps=[],
            )
        elif name == "ReferenceBundle":
            eid = p["evidence"][0]["id"]
            value = dict(
                answer=[dict(text="Blue.", evidence_ids=[eid])],
                rubric=[
                    dict(
                        id="R1",
                        requirement="State the color.",
                        priority="core",
                        reason="The question asks the color.",
                        evidence_ids=[eid],
                    )
                ],
                gaps=[],
            )
        elif name == "ChecklistCoverage":
            assert set(p) == {"question", "items", "answer"}
            assert all(set(c) == {"id", "text"} for c in p["answer"]["claims"])
            value = dict(
                answer_present=True,
                items=[
                    dict(
                        id=i["id"],
                        status="covered",
                        claim_ids=["C1"],
                        reason="Color supplied.",
                    )
                    for i in p["items"]
                ],
            )
        elif name == "Links":
            value = dict(
                links=[
                    dict(
                        case=p["answer"]["id"],
                        item=i["id"],
                        mentioned=True,
                        status="supported",
                        pair_ids=[p["answer"]["pairs"][0]["id"]],
                        supported_portion="Blue.",
                        missing_or_unsupported="",
                    )
                    for i in p["items"]
                ]
            )
        elif name == "Checklist":
            value = dict(
                items=[
                    dict(
                        id="N1",
                        question="What color?",
                        requirement="Blue.",
                        acceptable_alternatives=[],
                        evidence_ids=[p["evidence"][0]["id"]],
                        request_basis="Requested color.",
                    )
                ],
                gaps=[],
            )
        elif name == "PairwiseDecision":
            value = dict(winner="tie", reason="Equivalent answers.")
        elif name == "CoverageGrade":
            value = dict(
                items=[
                    dict(
                        id=c["id"],
                        status="covered",
                        answer_sentences=[0],
                        reason="Color supplied.",
                    )
                    for c in p["reference"]["rubric"]
                ],
                summary="Covered.",
            )
        else:
            raise AssertionError(name)
        return MinimaLlmResponse(
            req.request_id,
            json.dumps(value),
            raw=dict(choices=[dict(finish_reason="stop")]),
        )

    async def aclose(self):
        pass


@pytest.mark.parametrize("track", ["rag", "ragtime"])
@pytest.mark.parametrize("method", ["pairwise", "evidence", "combined"])
def test_all_variants_cold_start_and_offline_replay(tmp_path, track, method):
    from judges.generic.unified import UnifiedJudge

    rs = [
        report(
            [dict(text="Blue.", citations={"d": 100})],
            {"d": dict(id="d", text="Blue.")},
        ),
        report(
            [dict(text="Blue.", citations={"d": 100})],
            {"d": dict(id="d", text="Blue.")},
        ),
    ]
    rs[1].metadata.run_id = "second"
    q = Request(request_id="t", title="What color?", word_limit=20)
    backend = SemanticBackend(tmp_path / "cache")
    judge = UnifiedJudge(backend_factory=lambda _: backend)
    judge.judge(
        rs,
        [q],
        SimpleNamespace(),
        outdir=tmp_path / "first",
        track=track,
        method=method,
        resolve_partial_grounding=True,
    )
    if method == "pairwise":
        assert set(backend.calls) == {"PairwiseDecision"}
    elif method == "evidence":
        assert "PairwiseDecision" not in backend.calls
    else:
        assert set(backend.calls) == {
            "PairwiseDecision",
            "ClaimSupport",
            "ReferenceChecklist",
            "ChecklistCoverage",
        }
    if method != "pairwise":
        assert backend.calls.count("ReferenceChecklist") == 1
        # These two answers are byte-identical, so the second grade is cached.
        assert backend.calls.count("ChecklistCoverage") == 1
        assert "Checklist" not in backend.calls and "CoverageGrade" not in backend.calls
    offline = SemanticBackend(tmp_path / "cache", offline=True)
    UnifiedJudge(backend_factory=lambda _: offline).judge(
        rs,
        [q],
        SimpleNamespace(),
        outdir=tmp_path / "second",
        track=track,
        method=method,
        resolve_partial_grounding=True,
    )
    first = json.loads((tmp_path / "first/unified.stages.json").read_text())
    if method != "pairwise":
        for record in first["topics"][0]["evidence"]["answers"]:
            if track == "rag":
                assert record["scoring"]["values"]["RAG_REQUEST_COVERAGE_PROXY"] == 1
            assert (
                record["scoring"]["values"][
                    f"{track.upper()}_GROUNDED_COVERAGE_PARTIAL_PROXY"
                ]
                == 1
            )
    second = json.loads((tmp_path / "second/unified.stages.json").read_text())
    assert first["topics"] == second["topics"]
    if method != "pairwise":
        evidence = first["topics"][0]["evidence"]
        assert "rubric" not in evidence["reference"]
        assert evidence["reference"]["checklist"] == evidence["checklist"]["items"]
        assert all("coverage_grade" not in a for a in evidence["answers"])
    assert not offline.calls


def test_full_credit_rejects_declared_missing_minimum_but_accepts_partial_source_subset():
    from judges.generic.shared_pipeline import validated_answer
    from judges.generic.models import JudgeError
    from tests.test_shared_track_judge import sample

    a = sample()
    # A partial source sentence can fully establish a narrower item.
    assert validated_answer(a, ["n"], strict_minimum=True)
    a["links"][0]["missing_or_unsupported"] = "Required roundness is not established."
    with pytest.raises(JudgeError, match="missing requirement"):
        validated_answer(a, ["n"], strict_minimum=True)


def test_session_replays_completed_call_despite_retained_started_marker(tmp_path):
    import asyncio
    from judges.generic.completion_session import CompletionSession
    from judges.generic.pairwise import _request, judge_topic

    packet = dict(
        question=dict(title="Which color?"),
        answers=[
            dict(id="A", sentences=[dict(text="Blue.", citations=[])]),
            dict(id="B", sentences=[dict(text="Red.", citations=[])]),
        ],
    )
    online = SemanticBackend(tmp_path / "cache")
    session = CompletionSession(
        SimpleNamespace(), tmp_path / "ledger", 0.1, factory=lambda _: online
    )
    asyncio.run(session.generate(_request(packet, 0, 1, "blind")))
    asyncio.run(session.generate(_request(packet, 1, 0, "blind")))
    assert list((tmp_path / "cache/unified-completions").glob("*.started"))
    offline = SemanticBackend(tmp_path / "cache", offline=True)
    cached = CompletionSession(
        SimpleNamespace(), tmp_path / "ledger", 0.1, factory=lambda _: offline
    )
    result = asyncio.run(judge_topic(packet, cached, None))
    assert result["scores"] == {"A": 0.5, "B": 0.5}
    assert not offline.calls


def test_budget_refusal_does_not_lock_a_never_sent_completion(tmp_path):
    import asyncio
    from judges.generic.completion_session import CompletionSession
    from judges.generic.models import RunStopped
    from judges.generic.pairwise import _request

    backend = SemanticBackend(tmp_path / "cache")
    backend.cfg.base_url = "https://openrouter.ai/api/v1"
    backend.cfg.max_attempts = 1
    packet = dict(
        question=dict(title="Color?"),
        answers=[
            dict(id="A", sentences=[dict(text="Blue.", citations=[])]),
            dict(id="B", sentences=[dict(text="Red.", citations=[])]),
        ],
    )
    session = CompletionSession(
        SimpleNamespace(), tmp_path / "ledger", 0.000001, factory=lambda _: backend
    )
    with pytest.raises(RunStopped):
        asyncio.run(session.generate(_request(packet, 0, 1, "blind")))
    assert not backend.calls
    assert not list((tmp_path / "cache/unified-completions").glob("*.started"))


@pytest.mark.parametrize("track", ["rag", "ragtime"])
def test_unified_workflow_cli_and_disabled_endpoint(tmp_path, track):
    import asyncio
    import os
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    backend = SemanticBackend(tmp_path / "unused")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "synthetic-model"
            response = asyncio.run(
                backend.generate(
                    SimpleNamespace(
                        request_id="http",
                        messages=body["messages"],
                        extra={"response_format": body["response_format"]},
                    )
                )
            )
            wire = json.dumps(
                dict(
                    choices=[
                        dict(message=dict(content=response.text), finish_reason="stop")
                    ]
                )
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "dataset"
    runs = dataset / "runs/repgen"
    runs.mkdir(parents=True)
    topics = dataset / "topics"
    topics.mkdir()
    (topics / "t.jsonl").write_text(
        Request(request_id="t", title="What color?").model_dump_json() + "\n"
    )
    for i in range(2):
        r = report(
            [dict(text="Blue.", citations={"d": 100})],
            {"d": dict(id="d", text="Blue.")},
        )
        r.metadata.run_id = f"A{i}"
        (runs / f"A{i}.jsonl").write_text(r.model_dump_json(exclude_none=True) + "\n")
    env = dict(
        os.environ,
        OPENAI_MODEL="synthetic-model",
        OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
        OPENAI_API_KEY="test-only",
        CACHE_DIR=str(tmp_path / "cache"),
        MAX_ATTEMPTS="1",
        RPM="0",
        MINIMA_TRACE_FILE="",
        MINIMA_DEBUG="",
        CACHE_FORCE_REFRESH="0",
    )
    command = [
        sys.executable,
        "-m",
        "judges.generic.runner",
        "--stage",
        "unified",
        "--track",
        track,
        "--method",
        "combined",
        "--input-dataset",
        str(dataset),
    ]
    try:
        result = subprocess.run(
            command + ["--out-dir", str(tmp_path / "first")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    replay = subprocess.run(
        command + ["--out-dir", str(tmp_path / "second"), "--replay"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    exported = (tmp_path / "first" / f"{track}-combined.eval.txt").read_text()
    assert "PRECISION_DEFINED" not in exported
    assert "ELIGIBILITY_COMPLETE" not in exported
    assert "ELIGIBLE_SENTENCE_PRECISION_ESTIMATE" not in exported
    assert (
        next((tmp_path / "first").glob("*.eval.txt")).read_text()
        == next((tmp_path / "second").glob("*.eval.txt")).read_text()
    )
