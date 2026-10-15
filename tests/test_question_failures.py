"""Opt-in, question-local failure recovery using synthetic completions only."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmFailure, MinimaLlmResponse

from judges.generic.models import JudgeError, RunStopped
from judges.generic.structured_completion import StructuredOutputError
from tests.report_fixtures import report
from tests.test_unified_judge import SemanticBackend


def _three_topics():
    questions = [
        Request(request_id=topic_id, title=f"What color is {topic_id}?", word_limit=20)
        for topic_id in ("first", "middle", "last")
    ]
    reports = []
    for question in questions:
        for index, run_id in enumerate(("run-a", "run-b")):
            document_id = f"{question.request_id}-doc-{index}"
            item = report(
                [
                    dict(
                        text=f"Blue is the answer for {question.request_id}.",
                        citations={document_id: 100},
                    )
                ],
                {
                    document_id: dict(
                        id=document_id,
                        text=f"Blue is supported for {question.request_id}.",
                    )
                },
            )
            item.metadata.topic_id = question.request_id
            item.metadata.run_id = run_id
            reports.append(item)
    return reports, questions


def _append_topic(reports, questions, topic_id):
    question = Request(request_id=topic_id, title=f"What color is {topic_id}?", word_limit=20)
    questions.append(question)
    for run_id in ("run-a", "run-b"):
        document_id = f"{topic_id}-doc-{run_id}"
        item = report(
            [dict(text=f"Blue is the answer for {topic_id}.", citations={document_id: 100})],
            {document_id: dict(id=document_id, text=f"Blue is supported for {topic_id}.")},
        )
        item.metadata.topic_id = topic_id
        item.metadata.run_id = run_id
        reports.append(item)


class _OneMalformedCompletion(SemanticBackend):
    """Corrupt one completed synthetic response at a selected pipeline stage."""

    def __init__(self, cache, *, phase, offline=False):
        super().__init__(cache, offline=offline)
        self.phase = phase
        self.target_reached = []

    async def generate(self, request):
        response = await super().generate(request)
        name = request.extra["response_format"]["json_schema"]["name"]
        payload = json.loads(request.messages[1]["content"])
        if self.phase == "wrapped" and name == "ClaimSupport":
            target = "middle" in json.dumps(payload)
        else:
            target = self.phase == "direct" and name == "ReferenceChecklist" and (
                payload.get("question", {}).get("request_id") == "middle"
            )
        if target:
            self.target_reached.append(name)
            return MinimaLlmResponse(
                request.request_id,
                "{synthetic malformed response",
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        return response


@pytest.mark.parametrize("phase", ["direct", "wrapped"])
def test_opt_in_defers_completed_output_failure_and_offline_replay_uses_receipts(
    tmp_path, phase
):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    cache = tmp_path / "cache"
    backend = _OneMalformedCompletion(cache, phase=phase)
    with pytest.raises(JudgeError, match="deferred"):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "online",
            repair_reference_output=False,
            repair_citation_output=False,
            defer_question_failures=True,
        )

    assert backend.target_reached == (
        ["ClaimSupport"] if phase == "wrapped" else ["ReferenceChecklist"]
    )
    assert backend.calls.count("ReferenceChecklist") == (2 if phase == "wrapped" else 3)
    assert not (tmp_path / "online/unified.stages.json").exists()
    snapshots = list((tmp_path / "online/unified.topic-results").glob("*.json"))
    assert len(snapshots) == 2
    saved = [json.loads(path.read_text()) for path in snapshots]
    assert {item["topic_id"] for item in saved} == {"first", "last"}
    assert all(item["rows"] and item["artifact"]["topic_id"] in {"first", "last"}
               for item in saved)
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in snapshots)
    assert os.stat(tmp_path / "online/unified.topic-results").st_mode & 0o777 == 0o700

    manifest_path = tmp_path / "online/unified.deferred-questions.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest == [
        dict(ordinal=2, category="structured_output", code="schema")
    ]
    assert "synthetic malformed" not in manifest_path.read_text()
    progress = json.loads((tmp_path / "online/unified.topic-progress.json").read_text())
    assert progress == dict(total=3, completed=2, deferred=1, fatal=0)
    assert all(type(value) is int for value in progress.values())

    # A normal offline backend can replay the failed and successful native
    # receipts. Per-topic snapshots are never consulted as a cache.
    receipts_before_replay = _receipt_names(cache)
    offline = SemanticBackend(cache, offline=True)
    with pytest.raises(JudgeError, match="deferred"):
        UnifiedJudge(backend_factory=lambda _: offline).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "offline",
            repair_reference_output=False,
            repair_citation_output=False,
            defer_question_failures=True,
        )
    assert offline.calls == []
    assert not (tmp_path / "offline/unified.stages.json").exists()
    assert _receipt_names(cache) == receipts_before_replay


def _receipt_names(cache):
    return {
        path.name
        for path in (cache / "unified-completions").glob("*.json")
        if path.name != "safe-failure.json"
    }


def test_default_failure_semantics_still_stop_at_the_bad_question(tmp_path):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    backend = _OneMalformedCompletion(tmp_path / "cache", phase="direct")
    with pytest.raises(StructuredOutputError):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "out",
            repair_reference_output=False,
        )
    assert backend.calls.count("ReferenceChecklist") == 2
    assert not list((tmp_path / "out").glob("unified.topic-*"))
    assert not (tmp_path / "out/unified.stages.json").exists()


class _BudgetFailureBackend(SemanticBackend):
    def __init__(self, cache):
        super().__init__(cache)
        self.reference_topics = []

    async def generate(self, request):
        name = request.extra["response_format"]["json_schema"]["name"]
        payload = json.loads(request.messages[1]["content"])
        topic_id = payload.get("question", {}).get("request_id")
        if name == "ReferenceChecklist":
            self.reference_topics.append(topic_id)
        if name == "ReferenceChecklist" and topic_id == "middle":
            raise RunStopped("budget_exhausted")
        return await super().generate(request)


def test_shared_budget_failure_is_fatal_and_does_not_schedule_later_topics(tmp_path):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    backend = _BudgetFailureBackend(tmp_path / "cache")
    with pytest.raises(RunStopped) as caught:
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "out",
            defer_question_failures=True,
        )
    assert caught.value.code == "budget_exhausted"
    assert backend.reference_topics == ["first", "middle"]
    progress = json.loads((tmp_path / "out/unified.topic-progress.json").read_text())
    assert progress == dict(total=3, completed=1, deferred=0, fatal=1)
    assert not (tmp_path / "out/unified.deferred-questions.json").exists()
    assert not (tmp_path / "out/unified.stages.json").exists()


class _MalformedTransportBackend(SemanticBackend):
    def __init__(self, cache, failures, *, error_type="MalformedResponse", status=200, attempts=1):
        super().__init__(cache)
        self.failures = set(failures)
        self.error_type = error_type
        self.status = status
        self.attempts = attempts
        self.reference_topics = []
        self.failed_topics = []

    async def generate(self, request):
        name = request.extra["response_format"]["json_schema"]["name"]
        payload = json.loads(request.messages[1]["content"])
        topic_id = payload.get("question", {}).get("request_id")
        if name == "ReferenceChecklist":
            self.reference_topics.append(topic_id)
            if topic_id in self.failures:
                self.failed_topics.append(topic_id)
                return MinimaLlmFailure(
                    request_id=request.request_id,
                    error_type=self.error_type,
                    message="synthetic body must not be persisted",
                    attempts=self.attempts,
                    status=self.status,
                )
        return await super().generate(request)


@pytest.mark.parametrize(
    ("bad_topics", "expected_topics", "expected_completed", "expected_fatal"),
    [
        (("middle",), ["first", "middle", "last"], 2, 0),
        (("middle", "last"), ["first", "middle", "last"], 1, 1),
    ],
)
def test_only_one_exact_malformed_completion_may_be_deferred(
    tmp_path, bad_topics, expected_topics, expected_completed, expected_fatal
):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    if "last" in bad_topics:
        # The second transport ambiguity stops before any subsequent question.
        _append_topic(reports, questions, "after")

    backend = _MalformedTransportBackend(tmp_path / "cache", bad_topics)
    expected_error = JudgeError if len(bad_topics) == 1 else RunStopped
    with pytest.raises(expected_error) as caught:
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "out",
            defer_question_failures=True,
        )
    if len(bad_topics) == 1:
        assert "deferred" in str(caught.value)
    else:
        assert caught.value.code == "transport"

    assert backend.reference_topics == expected_topics
    assert backend.failed_topics == list(bad_topics)
    progress = json.loads((tmp_path / "out/unified.topic-progress.json").read_text())
    assert progress == dict(
        total=len(questions),
        completed=expected_completed,
        deferred=len(bad_topics),
        fatal=expected_fatal,
    )
    manifest = json.loads(
        (tmp_path / "out/unified.deferred-questions.json").read_text()
    )
    assert manifest == [
        dict(ordinal=2, category="transport_ambiguity", code="malformed_response"),
        *(
            [dict(ordinal=3, category="transport_ambiguity", code="malformed_response")]
            if len(bad_topics) == 2
            else []
        ),
    ]
    assert "synthetic body" not in json.dumps(manifest)
    assert not (tmp_path / "out/unified.stages.json").exists()


@pytest.mark.parametrize(
    ("error_type", "status", "attempts"),
    [("HTTPError", 429, 1), ("MalformedResponse", 200, 2)],
)
def test_near_miss_transport_diagnostics_remain_fatal(
    tmp_path, error_type, status, attempts
):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    backend = _MalformedTransportBackend(
        tmp_path / "cache",
        ("middle",),
        error_type=error_type,
        status=status,
        attempts=attempts,
    )
    with pytest.raises(RunStopped) as caught:
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            outdir=tmp_path / "out",
            defer_question_failures=True,
        )

    assert caught.value.code == "transport"
    assert backend.reference_topics == ["first", "middle"]
    assert backend.failed_topics == ["middle"]
    progress = json.loads((tmp_path / "out/unified.topic-progress.json").read_text())
    assert progress == dict(total=3, completed=1, deferred=0, fatal=1)
    assert not (tmp_path / "out/unified.deferred-questions.json").exists()
    assert not (tmp_path / "out/unified.stages.json").exists()


def test_question_failure_deferral_flag_requires_boolean(tmp_path):
    from judges.generic.unified import UnifiedJudge

    for value in (0, 1, "true", None):
        with pytest.raises(JudgeError, match="deferral must be boolean"):
            UnifiedJudge(backend_factory=lambda _: pytest.fail("backend constructed")).judge(
                [], [], SimpleNamespace(), defer_question_failures=value,
                outdir=tmp_path / str(value),
            )


class _SlowSemanticBackend(SemanticBackend):
    def __init__(self, cache, *, offline=False):
        super().__init__(cache, offline=offline)
        self.active = 0
        self.peak = 0

    async def generate(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.002)
            return await super().generate(request)
        finally:
            self.active -= 1


def test_opt_in_parallel_success_preserves_rows_artifacts_and_receipt_identity(tmp_path):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    sequential_cache = tmp_path / "sequential-cache"
    sequential = UnifiedJudge(
        backend_factory=lambda _: SemanticBackend(sequential_cache)
    ).judge(
        reports,
        questions,
        SimpleNamespace(),
        method="evidence",
        outdir=tmp_path / "sequential",
    )

    parallel_cache = tmp_path / "parallel-cache"
    parallel_backend = _SlowSemanticBackend(parallel_cache)
    parallel = UnifiedJudge(
        backend_factory=lambda _: parallel_backend
    ).judge(
        reports,
        questions,
        SimpleNamespace(),
        method="evidence",
        question_concurrency=2,
        defer_question_failures=True,
        outdir=tmp_path / "parallel",
    )

    assert parallel_backend.peak >= 2
    assert _score_rows(parallel) == _score_rows(sequential)
    first_stages = json.loads((tmp_path / "sequential/unified.stages.json").read_text())
    parallel_stages = json.loads((tmp_path / "parallel/unified.stages.json").read_text())
    assert parallel_stages["topics"] == first_stages["topics"]
    assert _receipt_names(parallel_cache) == _receipt_names(sequential_cache)
    assert len(list((tmp_path / "parallel/unified.topic-results").glob("*.json"))) == 3

    offline_backend = SemanticBackend(parallel_cache, offline=True)
    replay = UnifiedJudge(backend_factory=lambda _: offline_backend).judge(
        reports,
        questions,
        SimpleNamespace(),
        method="evidence",
        question_concurrency=2,
        defer_question_failures=True,
        outdir=tmp_path / "replay",
    )
    replay_stages = json.loads((tmp_path / "replay/unified.stages.json").read_text())
    assert _score_rows(replay) == _score_rows(parallel)
    assert replay_stages["topics"] == parallel_stages["topics"]
    assert offline_backend.calls == []


def _score_rows(leaderboard):
    return [
        (entry.run_id, entry.topic_id, tuple(entry.values.items()))
        for entry in leaderboard.entries
    ]


class _SlowFirstOutputFailureBackend(_OneMalformedCompletion):
    def __init__(self, cache):
        super().__init__(cache, phase="direct")
        self.active = 0
        self.peak = 0

    async def generate(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            response = await super().generate(request)
            if "first" in request.messages[1]["content"]:
                await asyncio.sleep(0.01)
            return response
        finally:
            self.active -= 1


def test_question_concurrency_two_defers_local_failure_and_runs_later_topic(tmp_path):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    backend = _SlowFirstOutputFailureBackend(tmp_path / "cache")
    with pytest.raises(JudgeError, match="deferred"):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            question_concurrency=2,
            repair_reference_output=False,
            outdir=tmp_path / "out",
            defer_question_failures=True,
        )

    assert backend.peak >= 2
    assert backend.target_reached == ["ReferenceChecklist"]
    snapshots = [
        json.loads(path.read_text())
        for path in (tmp_path / "out/unified.topic-results").glob("*.json")
    ]
    assert {snapshot["topic_id"] for snapshot in snapshots} == {"first", "last"}
    assert json.loads((tmp_path / "out/unified.topic-progress.json").read_text()) == {
        "total": 3, "completed": 2, "deferred": 1, "fatal": 0
    }


class _SlowSiblingTransportBackend(_MalformedTransportBackend):
    async def generate(self, request):
        response = await super().generate(request)
        if "first" in request.messages[1]["content"]:
            await asyncio.sleep(0.01)
        return response


def test_concurrent_second_transport_stops_scheduling_but_drains_sibling(tmp_path):
    from judges.generic.unified import UnifiedJudge

    reports, questions = _three_topics()
    _append_topic(reports, questions, "after")
    backend = _SlowSiblingTransportBackend(tmp_path / "cache", ("middle", "last"))
    with pytest.raises(RunStopped) as caught:
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="evidence",
            question_concurrency=3,
            outdir=tmp_path / "out",
            defer_question_failures=True,
        )

    assert caught.value.code == "transport"
    assert set(backend.reference_topics) == {"first", "middle", "last"}
    assert "after" not in backend.reference_topics
    assert set(backend.failed_topics) == {"middle", "last"}
    snapshots = [
        json.loads(path.read_text())
        for path in (tmp_path / "out/unified.topic-results").glob("*.json")
    ]
    assert [snapshot["topic_id"] for snapshot in snapshots] == ["first"]
    assert json.loads((tmp_path / "out/unified.topic-progress.json").read_text()) == {
        "total": 4, "completed": 1, "deferred": 2, "fatal": 1
    }
