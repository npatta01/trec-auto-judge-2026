"""Synthetic failure paths: current safe diagnostics, no ambiguous resend."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmRequest, MinimaLlmFailure

from judges.generic.completion_session import CompletionSession
from judges.generic.document_pipeline import evaluate
from judges.generic.evidence_bridge import build_topic
from judges.generic.models import JudgeError, RunStopped
from tests.report_fixtures import report


def cited_report():
    return report([dict(text="Blue.", citations={"d": 100})],
                  {"d": dict(id="d", text="Blue.")})


@pytest.mark.parametrize("code", ["transport", "budget_exhausted", "budget_locked", "configuration"])
def test_bridge_retains_current_fixed_stop_category(code):
    async def checker(_):
        raise RunStopped(code)

    r = cited_report()
    support = asyncio.run(evaluate([r], checker))
    with pytest.raises(JudgeError, match="Citation stage incomplete") as failure:
        build_topic([r], Request(request_id="t", title="Color?"), support)
    assert getattr(failure.value, "code", None) == code


def test_internal_failure_keeps_type_and_source_location_not_exception_value():
    async def checker(_):
        raise JudgeError("synthetic-sensitive-value")

    r = cited_report()
    support = asyncio.run(evaluate([r], checker))
    with pytest.raises(JudgeError) as failure:
        build_topic([r], Request(request_id="t", title="Color?"), support)
    assert getattr(failure.value, "code", None) == "internal"
    diagnostic = getattr(failure.value, "diagnostics", None)
    assert diagnostic["exception_type"] == "JudgeError"
    assert any(p.startswith("test_citation_failure_diagnostics.py:")
               for p in diagnostic["source_locations"])
    assert "synthetic-sensitive-value" not in json.dumps(diagnostic) + str(failure.value)


def test_transport_failure_keeps_marker_and_reservation_and_never_resends(tmp_path):
    class BrokenProvider:
        cfg = SimpleNamespace(model="synthetic", cache_dir=tmp_path / "cache",
                              force_refresh=False, max_attempts=1,
                              base_url="https://openrouter.ai/api/v1")
        calls = 0

        async def generate(self, request):
            self.calls += 1
            raise ConnectionResetError("synthetic-sensitive-provider-body")

        async def aclose(self):
            pass

    backend = BrokenProvider()
    ledger = tmp_path / "budget.sqlite3"
    session = CompletionSession(SimpleNamespace(), ledger, 1, factory=lambda _: backend)
    req = MinimaLlmRequest("synthetic", [], max_tokens=256)

    async def checker(_):
        return await session.generate(req)

    r = cited_report()
    support = asyncio.run(evaluate([r], checker))
    with pytest.raises(JudgeError) as failure:
        build_topic([r], Request(request_id="t", title="Color?"), support)
    assert getattr(failure.value, "code", None) == "transport"
    assert failure.value.diagnostics["exception_type"] == "ConnectionResetError"
    assert "synthetic-sensitive-provider-body" not in json.dumps(failure.value.diagnostics)
    markers = list(session.directory.glob("*.started"))
    assert len(markers) == 1 and not markers[0].with_suffix(".json").exists()
    marker_bytes = markers[0].read_bytes()
    with sqlite3.connect(ledger) as db:
        before = db.execute("SELECT * FROM calls").fetchall()
    assert len(before) == 1 and before[0][2] is None
    with pytest.raises(JudgeError):
        asyncio.run(session.generate(req))
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT * FROM calls").fetchall() == before
    assert markers[0].read_bytes() == marker_bytes
    assert backend.calls == 1


@pytest.mark.parametrize("category,status", [
    ("HTTPError", 400), ("TimeoutError", 408), ("HTTPError", 429),
    ("HTTPError", 502), ("MalformedResponse", 200), ("JSONDecodeError", 200),
    ("synthetic-sensitive-category", None),
])
def test_returned_failure_reports_only_safe_fields_and_keeps_unknown_reservation(
    tmp_path, category, status, capsys,
):
    class FailedProvider:
        cfg = SimpleNamespace(model="synthetic", cache_dir=tmp_path / "cache",
                              force_refresh=False, max_attempts=1,
                              base_url="https://openrouter.ai/api/v1")
        calls = 0

        async def generate(self, request):
            self.calls += 1
            return MinimaLlmFailure(
                request_id="synthetic-sensitive-id", error_type=category,
                message="synthetic-sensitive-message", attempts=1, status=status,
                body_snippet="synthetic-sensitive-body", timeout_s=600.0,
                attempt_timestamps=(12345.6,),
            )

        async def aclose(self):
            pass

    backend = FailedProvider()
    ledger = tmp_path / "budget.sqlite3"
    session = CompletionSession(SimpleNamespace(), ledger, 1, factory=lambda _: backend)
    request = MinimaLlmRequest("synthetic", [], max_tokens=256)
    with pytest.raises(RunStopped) as failure:
        asyncio.run(session.generate(request))
    expected = {
        "stage": "completion", "exception_type": "MinimaLlmFailure",
        "failure_type": category if category != "synthetic-sensitive-category" else "other",
        "http_status": status, "attempts": 1, "timeout_s": 600.0,
    }
    assert failure.value.code == "transport"
    assert failure.value.diagnostics == expected
    files = list(session.directory.glob("*.failure-*.json"))
    assert len(files) == 1 and files[0].stat().st_mode & 0o777 == 0o600
    assert json.loads(files[0].read_text()) == expected
    diagnostic_bytes = files[0].read_bytes()
    marker = next(session.directory.glob("*.started"))
    marker_bytes = marker.read_bytes()
    with sqlite3.connect(ledger) as db:
        before = db.execute("SELECT * FROM calls").fetchall()
    assert len(before) == 1 and before[0][2] is None
    with pytest.raises(JudgeError):
        asyncio.run(session.generate(request))
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT * FROM calls").fetchall() == before
    assert backend.calls == 1
    assert marker.read_bytes() == marker_bytes
    assert files[0].read_bytes() == diagnostic_bytes
    exposed = capsys.readouterr()
    assert "synthetic-sensitive" not in (
        exposed.out + exposed.err + str(failure.value) + repr(failure.value)
        + json.dumps(failure.value.diagnostics) + files[0].read_text()
    )


@pytest.mark.parametrize("status,attempts,timeout", [
    ("synthetic-sensitive-status", "synthetic-sensitive-attempts", "synthetic-sensitive-timeout"),
    (True, True, True), (999, -1, float("nan")), (0, 0, float("inf")),
])
def test_failure_diagnostics_reject_non_numeric_or_invalid_resource_fields(
    tmp_path, status, attempts, timeout,
):
    class FailedProvider:
        cfg = SimpleNamespace(model="synthetic", cache_dir=tmp_path / "cache",
                              force_refresh=False, base_url="http://localhost")

        async def generate(self, request):
            return MinimaLlmFailure("secret", "HTTPError", "secret", attempts,
                                    status=status, timeout_s=timeout)

    session = CompletionSession(None, tmp_path / "ledger", 1,
                                factory=lambda _: FailedProvider())
    with pytest.raises(RunStopped) as failure:
        asyncio.run(session.generate(MinimaLlmRequest("test", [], max_tokens=256)))
    assert failure.value.diagnostics == {
        "stage": "completion", "exception_type": "MinimaLlmFailure",
        "failure_type": "HTTPError", "http_status": None,
        "attempts": None, "timeout_s": None,
    }


def test_returned_failure_diagnostics_reach_the_citation_stage(tmp_path):
    class FailedProvider:
        cfg = SimpleNamespace(model="synthetic", cache_dir=tmp_path / "cache",
                              force_refresh=False, base_url="http://localhost")

        async def generate(self, request):
            return MinimaLlmFailure("secret", "HTTPError", "secret", 1, status=429)

    session = CompletionSession(None, tmp_path / "ledger", 1,
                                factory=lambda _: FailedProvider())

    async def checker(_):
        return await session.generate(MinimaLlmRequest("test", [], max_tokens=256))

    r = cited_report()
    support = asyncio.run(evaluate([r], checker))
    with pytest.raises(JudgeError) as failure:
        build_topic([r], Request(request_id="t", title="Color?"), support)
    assert failure.value.code == "transport"
    assert failure.value.diagnostics["http_status"] == 429
    assert failure.value.diagnostics["failure_type"] == "HTTPError"


def test_unexpected_result_is_not_formatted_or_serialized(tmp_path):
    class SensitiveObject:
        def __repr__(self):
            raise AssertionError("Never format provider objects")

    class FailedProvider:
        cfg = SimpleNamespace(model="synthetic", cache_dir=tmp_path / "cache",
                              force_refresh=False, base_url="http://localhost")

        async def generate(self, request):
            return SensitiveObject()

    session = CompletionSession(None, tmp_path / "ledger", 1,
                                factory=lambda _: FailedProvider())
    with pytest.raises(RunStopped) as failure:
        asyncio.run(session.generate(MinimaLlmRequest("test", [], max_tokens=256)))
    assert failure.value.diagnostics == {
        "stage": "completion", "exception_type": "UnexpectedResult",
    }
