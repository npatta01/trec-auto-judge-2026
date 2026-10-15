"""Synthetic integration tests for question-level scheduling."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmRequest, MinimaLlmResponse

from judges.generic.models import JudgeError
from tests.report_fixtures import report


class PairwiseBackend:
    def __init__(self, cache, *, offline=False):
        self.cfg = SimpleNamespace(
            cache_dir=str(cache),
            model="synthetic-model",
            base_url="EMPTY" if offline else "http://localhost",
            force_refresh=False,
        )
        self.offline = offline
        self.active = 0
        self.peak = 0
        self.calls = []

    async def generate(self, request):
        if self.offline:
            raise AssertionError("offline replay reached the model backend")
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(request)
        try:
            await asyncio.sleep(0.01)
            return MinimaLlmResponse(
                request.request_id,
                json.dumps(dict(winner="tie", reason="Equivalent answers.")),
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        finally:
            self.active -= 1

    async def aclose(self):
        pass


def _inputs(question_titles):
    questions = [
        Request(request_id=topic_id, title=title, word_limit=20)
        for topic_id, title in question_titles
    ]
    reports = []
    for question in questions:
        for run_id, text in (("a", "A blue answer."), ("b", "A green answer.")):
            item = report([dict(text=text, citations={})], {})
            item.metadata.topic_id = question.request_id
            item.metadata.run_id = run_id
            reports.append(item)
    return reports, questions


def _run(
    reports, questions, outdir, backend, *, method="pairwise", track="ragtime", **settings
):
    from judges.generic.unified import UnifiedJudge

    factories = 0

    def factory(_):
        nonlocal factories
        factories += 1
        return backend

    result = UnifiedJudge(backend_factory=factory).judge(
        reports,
        questions,
        SimpleNamespace(),
        method=method,
        track=track,
        outdir=outdir,
        **settings,
    )
    return result, factories


def _semantic_inputs():
    questions = [
        Request(request_id="topic-z", title="What color is Z?", word_limit=20),
        Request(request_id="topic-a", title="What color is A?", word_limit=20),
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


def _receipt_names(cache):
    return {
        path.name
        for path in (cache / "unified-completions").glob("*.json")
        if path.name != "safe-failure.json"
    }


def _score_rows(leaderboard):
    return [
        (entry.run_id, entry.topic_id, tuple(entry.values.items()))
        for entry in leaderboard.entries
    ]


def test_questions_overlap_but_rows_artifacts_and_receipt_ids_are_stable(tmp_path):
    reports, questions = _inputs(
        [("topic-z", "Which color?"), ("topic-a", "Which shade?")]
    )
    sequential_backend = PairwiseBackend(tmp_path / "sequential-cache")
    sequential, sequential_factories = _run(
        reports, questions, tmp_path / "sequential", sequential_backend
    )

    parallel_backend = PairwiseBackend(tmp_path / "parallel-cache")
    parallel, parallel_factories = _run(
        reports,
        questions,
        tmp_path / "parallel",
        parallel_backend,
        question_concurrency=2,
    )

    assert sequential_factories == parallel_factories == 1
    assert sequential_backend.peak == 1  # The default remains serial.
    assert parallel_backend.peak == 2
    assert sequential == parallel

    sequential_stages = json.loads(
        (tmp_path / "sequential/unified.stages.json").read_text()
    )
    parallel_stages = json.loads(
        (tmp_path / "parallel/unified.stages.json").read_text()
    )
    assert [row["topic_id"] for row in parallel_stages["topics"]] == [
        "topic-z",
        "topic-a",
    ]
    assert sequential_stages["question_concurrency"] == 1
    assert parallel_stages["question_concurrency"] == 2
    assert sequential_stages["topics"] == parallel_stages["topics"]

    sequential_receipts = {
        path.name
        for path in (tmp_path / "sequential-cache/unified-completions").glob("*.json")
        if path.name != "safe-failure.json"
    }
    parallel_receipts = {
        path.name
        for path in (tmp_path / "parallel-cache/unified-completions").glob("*.json")
        if path.name != "safe-failure.json"
    }
    assert len(sequential_receipts) == 4
    assert parallel_receipts == sequential_receipts


def test_identical_question_prompts_share_one_session_and_replay_offline(tmp_path):
    reports, questions = _inputs(
        [
            ("topic-one", "Same semantic question?"),
            ("topic-two", "Same semantic question?"),
        ]
    )
    live_backend = PairwiseBackend(tmp_path / "cache")
    live, factory_count = _run(
        reports,
        questions,
        tmp_path / "live",
        live_backend,
        question_concurrency=2,
    )

    # The two topic IDs do not affect the prompt fingerprint; one shared session
    # deduplicates both orientations rather than issuing a second pair of calls.
    assert factory_count == 1
    assert len(live_backend.calls) == 2

    offline_backend = PairwiseBackend(tmp_path / "cache", offline=True)
    replay, replay_factory_count = _run(
        reports,
        questions,
        tmp_path / "replay",
        offline_backend,
        question_concurrency=2,
    )
    assert replay_factory_count == 1
    assert offline_backend.calls == []
    assert replay == live
    assert json.loads((tmp_path / "live/unified.stages.json").read_text())[
        "topics"
    ] == json.loads((tmp_path / "replay/unified.stages.json").read_text())["topics"]


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5, "2"])
def test_question_concurrency_rejects_non_positive_or_non_integer_values(
    tmp_path, concurrency
):
    from judges.generic.unified import UnifiedJudge

    with pytest.raises(JudgeError, match="Question concurrency"):
        UnifiedJudge(
            backend_factory=lambda _: pytest.fail("backend was constructed")
        ).judge(
            [], [], SimpleNamespace(), outdir=tmp_path / "out",
            question_concurrency=concurrency,
        )
    assert not (tmp_path / "out").exists()


def test_question_failure_drains_already_started_sibling_topic(tmp_path):
    class DrainBackend(PairwiseBackend):
        def __init__(self, cache):
            super().__init__(cache)
            self.completed_sibling = False

        async def generate(self, request):
            payload = json.loads(request.messages[1]["content"])
            title = payload["question"]["title"]
            self.calls.append(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if title == "Fail this question":
                    await asyncio.sleep(0.001)
                    raise RuntimeError("synthetic backend failure")
                await asyncio.sleep(0.02)
                self.completed_sibling = True
                return MinimaLlmResponse(
                    request.request_id,
                    json.dumps(dict(winner="tie", reason="Equivalent answers.")),
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            finally:
                self.active -= 1

    reports, questions = _inputs(
        [
            ("topic-fail", "Fail this question"),
            ("topic-slow", "Finish this question"),
        ]
    )
    backend = DrainBackend(tmp_path / "cache")
    from judges.generic.unified import UnifiedJudge

    with pytest.raises(JudgeError):
        UnifiedJudge(backend_factory=lambda _: backend).judge(
            reports,
            questions,
            SimpleNamespace(),
            method="pairwise",
            outdir=tmp_path / "out",
            question_concurrency=2,
        )

    assert backend.peak == 2
    assert backend.active == 0
    assert backend.completed_sibling
    receipts = list((tmp_path / "cache/unified-completions").glob("*.json"))
    assert len([path for path in receipts if path.name != "safe-failure.json"]) == 2


@pytest.mark.parametrize("track", ["rag", "ragtime"])
@pytest.mark.parametrize("method", ["evidence", "combined"])
def test_semantic_question_parallelism_matches_serial_rows_and_replays(
    tmp_path, track, method
):
    from tests.test_evidence_concurrency import SlowBackend

    reports, questions = _semantic_inputs()
    sequential_cache = tmp_path / f"{track}-{method}-sequential-cache"
    parallel_cache = tmp_path / f"{track}-{method}-parallel-cache"

    sequential_backend = SlowBackend(sequential_cache)
    sequential, sequential_factories = _run(
        reports,
        questions,
        tmp_path / f"{track}-{method}-sequential",
        sequential_backend,
        method=method,
        track=track,
    )

    parallel_backend = SlowBackend(parallel_cache)
    parallel, parallel_factories = _run(
        reports,
        questions,
        tmp_path / f"{track}-{method}-parallel",
        parallel_backend,
        method=method,
        track=track,
        question_concurrency=2,
    )

    assert sequential_factories == parallel_factories == 1
    assert sequential_backend.peak == 1
    assert parallel_backend.peak == 2
    assert _score_rows(parallel) == _score_rows(sequential)

    sequential_stages = json.loads(
        (tmp_path / f"{track}-{method}-sequential/unified.stages.json").read_text()
    )
    parallel_stages = json.loads(
        (tmp_path / f"{track}-{method}-parallel/unified.stages.json").read_text()
    )
    assert parallel_stages["topics"] == sequential_stages["topics"]
    assert _receipt_names(sequential_cache) == _receipt_names(parallel_cache)

    offline_backend = SlowBackend(parallel_cache, offline=True)
    offline, offline_factories = _run(
        reports,
        questions,
        tmp_path / f"{track}-{method}-offline",
        offline_backend,
        method=method,
        track=track,
        question_concurrency=2,
    )
    assert offline_factories == 1
    assert offline_backend.requests == []
    assert _score_rows(offline) == _score_rows(parallel)
    offline_stages = json.loads(
        (tmp_path / f"{track}-{method}-offline/unified.stages.json").read_text()
    )
    assert offline_stages["topics"] == parallel_stages["topics"]


def test_many_questions_share_one_transport_limiter_of_32(tmp_path):
    class SharedTransportBackend(PairwiseBackend):
        def __init__(self, cache):
            super().__init__(cache)
            self.transport = asyncio.Semaphore(32)
            self.transport_active = 0
            self.transport_peak = 0

        async def generate(self, request):
            async with self.transport:
                self.transport_active += 1
                self.transport_peak = max(
                    self.transport_peak, self.transport_active
                )
                try:
                    await asyncio.sleep(0.01)
                    self.calls.append(request)
                    return MinimaLlmResponse(
                        request.request_id,
                        json.dumps(dict(winner="tie", reason="Equivalent answers.")),
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                finally:
                    self.transport_active -= 1

    reports, questions = _inputs(
        [(f"topic-{index:02d}", f"Question {index}?") for index in range(40)]
    )
    backend = SharedTransportBackend(tmp_path / "cache")
    _leaderboard, factories = _run(
        reports,
        questions,
        tmp_path / "out",
        backend,
        question_concurrency=40,
    )

    assert factories == 1
    assert backend.transport_peak == 32
    assert backend.transport_active == 0
    assert len(backend.calls) == 80


def test_completion_admission_limits_budget_reservations_before_transport(
    tmp_path, monkeypatch
):
    from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

    state = {"active": 0, "peak": 0}
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def synthetic_post(client, url, payload):
        assert client.cfg.max_outstanding == 2
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        if state["active"] == 2:
            saturated.set()
        try:
            await release.wait()
            wire = {
                "choices": [
                    {"message": {"content": "synthetic result"}, "finish_reason": "stop"}
                ],
                "usage": {"cost": 0.001},
            }
            return 200, {}, json.dumps(wire).encode()
        finally:
            state["active"] -= 1

    monkeypatch.setattr(OpenAIMinimaLlm, "_post_json", synthetic_post)

    async def exercise():
        from judges.generic.completion_session import CompletionSession

        backend = OpenAIMinimaLlm(
            MinimaLlmConfig(
                base_url="https://openrouter.ai/api/v1",
                model="synthetic-openrouter-model",
                api_key="synthetic-only",
                cache_dir=str(tmp_path / "cache"),
                max_attempts=1,
                max_outstanding=2,
                rpm=0,
            )
        )
        ledger = tmp_path / "budget.sqlite3"
        session = CompletionSession(
            SimpleNamespace(), ledger, 1.0, factory=lambda _: backend
        )
        requests = [
            MinimaLlmRequest(
                f"request-{index}",
                [{"role": "user", "content": f"distinct prompt {index}"}],
                max_tokens=8,
            )
            for index in range(8)
        ]
        duplicate = MinimaLlmRequest(
            "duplicate-request",
            requests[0].messages,
            max_tokens=requests[0].max_tokens,
        )
        tasks = [
            asyncio.create_task(session.generate(request))
            for request in [*requests, duplicate]
        ]
        try:
            await asyncio.wait_for(saturated.wait(), timeout=1)
            # Give all queued tasks a chance to reach the admission boundary.
            await asyncio.sleep(0.02)
            with sqlite3.connect(ledger) as db:
                outstanding = db.execute(
                    "SELECT COUNT(*) FROM calls WHERE reported IS NULL"
                ).fetchone()[0]
            assert outstanding == 2
            assert state["peak"] == 2

            release.set()
            results = await asyncio.gather(*tasks)
            assert len(results) == len(requests) + 1
            assert results[-1].cached
            with sqlite3.connect(ledger) as db:
                assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 8
                assert db.execute(
                    "SELECT COUNT(*) FROM calls WHERE reported IS NULL"
                ).fetchone()[0] == 0
            assert state["peak"] <= backend.cfg.max_outstanding

            # A cached replay should remain available even if all provider slots
            # are occupied by unrelated work.
            for _ in range(2):
                await session._admission.acquire()
            try:
                replay = await asyncio.wait_for(
                    session.generate(duplicate), timeout=0.1
                )
                assert replay.cached
            finally:
                for _ in range(2):
                    session._admission.release()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await session.aclose()

    asyncio.run(exercise())


def test_cached_replay_still_dispatches_subclass_generate_hook(tmp_path):
    async def exercise():
        from judges.generic.completion_session import CompletionSession

        class HookSession(CompletionSession):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.hook_calls = 0

            async def _generate(self, request):
                self.hook_calls += 1
                return await super()._generate(request)

        backend = PairwiseBackend(tmp_path / "cache")
        backend.cfg.max_outstanding = 1
        session = HookSession(
            SimpleNamespace(),
            tmp_path / "budget.sqlite3",
            1.0,
            factory=lambda _: backend,
        )
        request = MinimaLlmRequest(
            "request",
            [{"role": "user", "content": "seed one cached completion"}],
            max_tokens=8,
        )
        try:
            first = await session.generate(request)
            assert not first.cached
            assert session.hook_calls == 1
            assert len(backend.calls) == 1

            await session._admission.acquire()
            try:
                replay = await asyncio.wait_for(session.generate(request), timeout=0.1)
            finally:
                session._admission.release()

            assert replay.cached
            assert session.hook_calls == 2
            assert len(backend.calls) == 1
        finally:
            await session.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("max_outstanding", [0, -1, True, 1.5, "2"])
def test_completion_admission_rejects_invalid_width(tmp_path, max_outstanding):
    from judges.generic.completion_session import CompletionSession

    cache = tmp_path / "cache"
    backend = PairwiseBackend(cache)
    backend.cfg.base_url = "http://localhost"
    backend.cfg.max_outstanding = max_outstanding
    with pytest.raises(JudgeError, match="outstanding completion limit"):
        CompletionSession(
            SimpleNamespace(), tmp_path / "budget.sqlite3", 1.0,
            factory=lambda _: backend,
        )
    assert not cache.exists()
