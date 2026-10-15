"""Synthetic regressions for overlapping async judge output suppression."""

import asyncio
import io
import json
import sys
import threading
from types import SimpleNamespace

from minima_llm import MinimaLlmResponse


def _pairwise_packet():
    return {
        "question": {"title": "Synthetic question"},
        "answers": [
            {"id": "answer-a", "sentences": [{"text": "A", "citations": []}]},
            {"id": "answer-b", "sentences": [{"text": "B", "citations": []}]},
        ],
    }


def test_overlapping_pairwise_topics_restore_streams_after_first_exit(monkeypatch):
    from judges.generic import pairwise

    original_stdout, original_stderr = sys.stdout, sys.stderr
    expected_stdout, expected_stderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = expected_stdout, expected_stderr

    async def exercise():
        entered = {name: asyncio.Event() for name in ("first", "second")}
        release = {name: asyncio.Event() for name in ("first", "second")}
        calls = {name: 0 for name in ("first", "second")}

        async def synthetic_call(_request, backend, _checkpoints):
            calls[backend.name] += 1
            if calls[backend.name] == 1:
                entered[backend.name].set()
                await release[backend.name].wait()
            return {"winner": "tie", "reason": "synthetic"}

        async def judge_and_print(name):
            result = await pairwise.judge_topic(
                _pairwise_packet(), SimpleNamespace(name=name), None, mode="blind"
            )
            print(f"synthetic_after:{name}")
            print(f"synthetic_after:{name}", file=sys.stderr)
            return result

        monkeypatch.setattr(pairwise, "_call", synthetic_call)
        first = asyncio.create_task(judge_and_print("first"))
        await entered["first"].wait()
        second = asyncio.create_task(judge_and_print("second"))
        await entered["second"].wait()

        # The first-entered context exits while the later one is still active.
        release["first"].set()
        first_result = await first
        output_after_first_exit = expected_stdout.getvalue()
        errors_after_first_exit = expected_stderr.getvalue()

        release["second"].set()
        second_result = await asyncio.gather(second, return_exceptions=True)
        return (
            first_result,
            second_result[0],
            output_after_first_exit,
            errors_after_first_exit,
        )

    try:
        first_result, second_result, leaked_stdout, leaked_stderr = asyncio.run(exercise())
        assert leaked_stdout == ""
        assert leaked_stderr == ""
        assert not isinstance(second_result, BaseException)
        assert first_result["scores"] == second_result["scores"]
        assert sys.stdout is expected_stdout
        assert sys.stderr is expected_stderr
        assert expected_stdout.getvalue() == "synthetic_after:second\n"
        assert expected_stderr.getvalue() == "synthetic_after:second\n"
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_overlapping_document_judges_suppress_until_last_checker_exits(tmp_path):
    from autojudge_base import Request
    from judges.generic.document_judge import DocumentJudge
    from tests.report_fixtures import report

    original_stdout, original_stderr = sys.stdout, sys.stderr
    expected_stdout, expected_stderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = expected_stdout, expected_stderr
    entered = {name: threading.Event() for name in ("first", "second")}
    release = {name: threading.Event() for name in ("first", "second")}
    outcomes = {}
    stdout_suppressed_after_first = False
    stderr_suppressed_after_first = False
    stdout_restored = False
    stderr_restored = False

    class Backend:
        cfg = SimpleNamespace(model="synthetic", base_url="EMPTY")

        def __init__(self, name):
            self.name = name

        async def generate(self, request):
            entered[self.name].set()
            if not await asyncio.to_thread(release[self.name].wait, 5):
                raise TimeoutError("synthetic release timed out")
            payload = json.loads(request.messages[1]["content"])
            return MinimaLlmResponse(
                request_id=request.request_id,
                text=json.dumps(
                    {
                        "claims": [
                            {
                                "id": claim["id"],
                                "eligibility": "claim",
                                "label": "supported",
                                "uncertain": False,
                                "contradiction": False,
                            }
                            for claim in payload["claims"]
                        ]
                    }
                ),
            )

        async def aclose(self):
            pass

    def judge_and_print(name):
        try:
            DocumentJudge(backend_factory=lambda _config: Backend(name)).judge(
                [
                    report(
                        [{"text": "Synthetic claim.", "citations": {"d": 100.0}}],
                        {"d": {"id": "d", "text": "Synthetic source."}},
                    )
                ],
                [Request(request_id="t", title="Synthetic question")],
                SimpleNamespace(),
                filebase=f"document-{name}",
                outdir=tmp_path / name,
            )
            print(f"synthetic_after:{name}")
            print(f"synthetic_after:{name}", file=sys.stderr)
        except BaseException as error:
            outcomes[name] = error
        else:
            outcomes[name] = "complete"

    threads = [
        threading.Thread(target=judge_and_print, args=(name,))
        for name in ("first", "second")
    ]
    try:
        threads[0].start()
        assert entered["first"].wait(5)
        threads[1].start()
        assert entered["second"].wait(5)

        # The first-entered context exits while the later one is still active.
        release["first"].set()
        threads[0].join(5)
        assert not threads[0].is_alive()
        leaked_stdout = expected_stdout.getvalue()
        leaked_stderr = expected_stderr.getvalue()
        stdout_suppressed_after_first = sys.stdout is not expected_stdout
        stderr_suppressed_after_first = sys.stderr is not expected_stderr

        release["second"].set()
        threads[1].join(5)
        assert not threads[1].is_alive()
        stdout_restored = sys.stdout is expected_stdout
        stderr_restored = sys.stderr is expected_stderr
    finally:
        release["first"].set()
        release["second"].set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
        sys.stdout, sys.stderr = original_stdout, original_stderr

    assert leaked_stdout == ""
    assert leaked_stderr == ""
    assert outcomes == {"first": "complete", "second": "complete"}
    assert stdout_suppressed_after_first
    assert stderr_suppressed_after_first
    assert stdout_restored
    assert stderr_restored
    assert expected_stdout.getvalue() == "synthetic_after:second\n"
    assert expected_stderr.getvalue() == "synthetic_after:second\n"
