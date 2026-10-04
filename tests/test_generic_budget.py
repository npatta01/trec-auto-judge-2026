import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from minima_llm import MinimaLlmRequest, MinimaLlmResponse


def test_other_run_settlement_cannot_increase_own_allowance(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
        hits = 0

        async def generate(self, req):
            self.hits += 1
            return MinimaLlmResponse(
                req.request_id, "{}", raw={"usage": {"cost": 0.008}}
            )

    ledger = tmp_path / "budget.db"
    # Legacy schema and an outstanding reservation from an earlier invocation.
    with sqlite3.connect(ledger) as db:
        db.execute(
            "CREATE TABLE calls (id INTEGER PRIMARY KEY, reserved REAL, reported REAL, cached INTEGER)"
        )
        db.execute("INSERT INTO calls VALUES (1, .1, NULL, 0)")
    raw = Backend()
    guarded = BudgetBackend(raw, ledger, incremental_cap=0.015)
    with sqlite3.connect(ledger) as db:
        db.execute("UPDATE calls SET reported=.001 WHERE id=1")
    req = MinimaLlmRequest("test", [{"role": "user", "content": "hi"}], max_tokens=100)
    asyncio.run(guarded.generate(req))
    with pytest.raises(JudgeError, match="budget"):
        asyncio.run(guarded.generate(req))
    assert raw.hits == 1
    assert guarded.summary()["reported_usd"] == pytest.approx(0.009)


def test_budget_persists_and_refuses_before_network(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
        hits = 0

        async def generate(self, req):
            self.hits += 1
            assert req.extra["provider"]["max_price"]["completion"] == 3
            return MinimaLlmResponse(req.request_id, "{}", raw={})

        async def aclose(self):
            pass

    raw = Backend()
    path = tmp_path / "budget.db"
    req = MinimaLlmRequest("test", [{"role": "user", "content": "hi"}], max_tokens=100)
    first = BudgetBackend(raw, path, cap=0.015)
    assert path.stat().st_mode & 0o777 == 0o600
    asyncio.run(first.generate(req))
    assert first.summary()["unknown_cost_calls"] == 1
    second = BudgetBackend(raw, path, cap=0.015)
    with pytest.raises(JudgeError, match="budget"):
        asyncio.run(second.generate(req))
    assert raw.hits == 1


def test_budget_rejects_hidden_retries(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    with pytest.raises(JudgeError):
        BudgetBackend(
            SimpleNamespace(
                cfg=SimpleNamespace(
                    max_attempts=2, base_url="https://openrouter.ai/api/v1"
                )
            ),
            tmp_path / "budget.db",
        )


def test_confirmed_cost_releases_only_unused_reservation(tmp_path):
    from judges.generic.budget import BudgetBackend

    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")

        async def generate(self, req):
            return MinimaLlmResponse(
                req.request_id, "{}", raw={"usage": {"cost": 0.001}}
            )

    backend = BudgetBackend(Backend(), tmp_path / "budget.db", cap=0.015)
    req = MinimaLlmRequest("test", [{"role": "user", "content": "hi"}], max_tokens=100)
    asyncio.run(backend.generate(req))
    asyncio.run(backend.generate(req))
    assert backend.summary()["reported_usd"] == 0.002


def test_schema_bytes_are_included_in_preflight_cost(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")

        async def generate(self, req):
            pytest.fail("Oversized schema was sent without a reservation")

    backend = BudgetBackend(Backend(), tmp_path / "budget.db", cap=0.015)
    req = MinimaLlmRequest(
        "test",
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        extra={"response_format": {"schema": "x" * 10000}},
    )
    with pytest.raises(JudgeError, match="budget"):
        asyncio.run(backend.generate(req))


@pytest.mark.parametrize("topic_id", ["", "  "])
def test_blank_topic_ids_are_rejected(topic_id):
    from autojudge_base import Request
    from judges.generic.judge import GenericJudge
    from judges.generic.models import JudgeError
    from tests.test_generic_judge import report

    r = report(sentences=[])
    r.metadata.topic_id = topic_id
    with pytest.raises(JudgeError):
        GenericJudge().judge(
            [r], [Request(request_id=topic_id, title="")], SimpleNamespace(raw={})
        )


def test_budget_preserves_high_reasoning_and_stops_at_incremental_cap(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
        hits = 0

        async def generate(self, req):
            self.hits += 1
            assert req.extra["reasoning"] == {"effort": "high"}
            assert req.extra["provider"]["max_price"]["completion"] == 4
            return MinimaLlmResponse(req.request_id, "{}", raw={})

    raw = Backend()
    guarded = BudgetBackend(
        raw, tmp_path / "budget.db", cap=16, incremental_cap=0.015, completion_price=4
    )
    req = MinimaLlmRequest(
        "test",
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        extra={"reasoning": {"effort": "high"}},
    )
    asyncio.run(guarded.generate(req))
    with pytest.raises(JudgeError, match="budget"):
        asyncio.run(guarded.generate(req))
    assert raw.hits == 1
