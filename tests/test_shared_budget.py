import asyncio
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest
from minima_llm import MinimaLlmRequest, MinimaLlmResponse


def test_explicit_budget_increase_preserves_outstanding_reservations(tmp_path):
    from judges.generic.budget import BudgetBackend, set_budget_cap
    from judges.generic.models import JudgeError

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    path = tmp_path / "budget.db"
    BudgetBackend(backend, path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO calls(reserved,reported) VALUES(2,NULL)")
    set_budget_cap(path, expected_cap=16, new_cap=25)
    assert BudgetBackend(backend, path).summary()["committed_usd"] == 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT cap FROM policy").fetchone()[0] == 25
    with pytest.raises(JudgeError):
        set_budget_cap(path, expected_cap=16, new_cap=30)
    with pytest.raises(JudgeError):
        set_budget_cap(path, expected_cap=25, new_cap=501)
    with pytest.raises(JudgeError):
        BudgetBackend(backend, path, cap=30)
    set_budget_cap(path, expected_cap=25, new_cap=150)
    with sqlite3.connect(path) as db:
        before = db.execute("SELECT * FROM calls ORDER BY id").fetchall()
    set_budget_cap(path, expected_cap=150, new_cap=300)
    assert BudgetBackend(backend, path, cap=300).summary()['committed_usd'] == 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM calls ORDER BY id").fetchall() == before
        assert db.execute("SELECT cap FROM policy").fetchone()[0] == 300
        assert db.execute(
            "SELECT old_cap, new_cap FROM budget_changes ORDER BY rowid DESC LIMIT 1"
        ).fetchone() == (150, 300)


def test_new_budget_accepts_authorized_300_ceiling(tmp_path):
    from judges.generic.budget import BudgetBackend

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    guarded = BudgetBackend(backend, tmp_path / "budget.db", cap=300)
    assert guarded.summary()["calls"] == 0


def test_new_budget_rejects_above_authorized_500_ceiling(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    path = tmp_path / "budget.db"
    with pytest.raises(JudgeError):
        BudgetBackend(backend, path, cap=501)
    assert not path.exists()


def test_authorized_500_update_preserves_calls_and_records_audit(tmp_path):
    from judges.generic.budget import BudgetBackend, set_budget_cap

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    ledger = tmp_path / "budget.db"
    BudgetBackend(backend, ledger, cap=300)
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO calls(reserved,reported,run_id) VALUES(2,NULL,'old')")
        db.execute("INSERT INTO calls(reserved,reported,run_id) VALUES(3,1,'saved')")
        before = db.execute("SELECT * FROM calls ORDER BY id").fetchall()

    set_budget_cap(ledger, expected_cap=300, new_cap=500)

    assert BudgetBackend(backend, ledger, cap=500).summary()["committed_usd"] == 3
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT * FROM calls ORDER BY id").fetchall() == before
        assert db.execute("SELECT cap FROM policy WHERE id=1").fetchone() == (500,)
        assert db.execute("SELECT old_cap,new_cap FROM budget_changes").fetchall() == [(300, 500)]


def test_new_budget_accepts_authorized_500_ceiling(tmp_path):
    from judges.generic.budget import BudgetBackend

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    assert BudgetBackend(backend, tmp_path / "budget.db", cap=500).summary()["calls"] == 0


@pytest.mark.parametrize('drift', [False, True])
def test_cap_update_checks_approved_calls_in_the_write_transaction(tmp_path, drift):
    from judges.generic.budget import BudgetBackend, set_budget_cap
    from judges.generic.models import JudgeError

    backend = SimpleNamespace(
        cfg=SimpleNamespace(max_attempts=1, base_url="https://openrouter.ai/api/v1")
    )
    ledger = tmp_path / 'budget.db'
    BudgetBackend(backend, ledger, cap=300)
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO calls(reserved,reported,run_id) VALUES(2,NULL,'old')")
        rows = db.execute('SELECT id,reserved,reported,run_id FROM calls ORDER BY id').fetchall()
    approved = hashlib.sha256(json.dumps(rows, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    # A competing writer commits after admission's read and before policy mutation.
    if drift:
        with sqlite3.connect(ledger) as db:
            db.execute("INSERT INTO calls(reserved,reported,run_id) VALUES(3,1,'other')")
        with pytest.raises(JudgeError):
            set_budget_cap(ledger, expected_cap=300, new_cap=500,
                           expected_calls_sha256=approved)
    else:
        set_budget_cap(ledger, expected_cap=300, new_cap=500,
                       expected_calls_sha256=approved)
    with sqlite3.connect(ledger) as db:
        assert db.execute('SELECT cap FROM policy WHERE id=1').fetchone() == (300 if drift else 500,)
        assert db.execute("SELECT reserved,reported FROM calls WHERE run_id='old'").fetchall() == [(2, None)]
        if drift:
            assert db.execute("SELECT name FROM sqlite_master WHERE name='budget_changes'").fetchall() == []
        else:
            assert db.execute('SELECT old_cap,new_cap FROM budget_changes').fetchall() == [(300, 500)]


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
