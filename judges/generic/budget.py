"""Conservative, restart-safe OpenRouter experiment budget; no prompt logging."""

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from minima_llm import MinimaLlmResponse

from .models import JudgeError, RunStopped
from .private_io import private_directory, prepare_private_file


def with_openrouter_options(req, *, completion_price):
    """Keep paid and cache-only requests identical without coupling replay to a ledger."""
    extra = dict(req.extra or {})
    extra.update(
        provider={
            "max_price": {"prompt": 1, "completion": completion_price, "request": 0},
            "require_parameters": True,
            "allow_fallbacks": False,
            "data_collection": "deny",
        },
        reasoning=extra.get("reasoning", {"enabled": False}),
    )
    return replace(req, extra=extra)


def set_budget_cap(ledger, *, expected_cap, new_cap):
    """Explicit operator action after user authorization; never clears reservations."""
    if not math.isfinite(new_cap) or not 0 < new_cap <= 100:
        raise JudgeError("Authorized budget ceiling must be positive and at most $100.")
    with closing(sqlite3.connect(ledger)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT cap FROM policy WHERE id=1").fetchone()[0]
        if current <= 0 or current != expected_cap:
            raise JudgeError("Budget policy changed or locked; no automatic override.")
        db.execute(
            "CREATE TABLE IF NOT EXISTS budget_changes (changed_at TEXT, old_cap REAL, new_cap REAL)"
        )
        db.execute(
            "INSERT INTO budget_changes VALUES (datetime('now'),?,?)",
            (current, new_cap),
        )
        db.execute("UPDATE policy SET cap=? WHERE id=1", (new_cap,))


class BudgetBackend:
    def __init__(
        self,
        backend,
        ledger,
        cap=None,
        *,
        incremental_cap=None,
        completion_price=3,
        prompt_price=1,
        max_output_tokens=8192,
        data_collection="deny",
    ):
        if (
            backend.cfg.max_attempts != 1
            or urlparse(backend.cfg.base_url).hostname != "openrouter.ai"
        ):
            raise JudgeError(
                "Budgeted experiments require OpenRouter and exactly one transport attempt."
            )
        if cap is not None and (not math.isfinite(cap) or not 0 < cap <= 100):
            raise JudgeError("Experiment budget cap must be positive and at most $100.")
        if completion_price not in (3, 4, 10) or prompt_price not in (1, 2):
            raise JudgeError("Unsupported completion price ceiling.")
        if max_output_tokens not in (8192, 16384):
            raise JudgeError("Unsupported output token ceiling.")
        if data_collection not in ("deny", "allow"):
            raise JudgeError("Unsupported provider data policy.")
        if incremental_cap is not None and (
            not math.isfinite(incremental_cap) or incremental_cap <= 0
        ):
            raise JudgeError("Incremental budget must be finite and positive.")
        self.cfg = backend.cfg
        self.run_id = uuid4().hex
        self.incremental_cap = incremental_cap
        self.completion_price = completion_price
        self.prompt_price = prompt_price
        self.max_output_tokens = max_output_tokens
        self.data_collection = data_collection
        self.backend, self.ledger = backend, Path(ledger)
        private_directory(self.ledger.parent)
        prepare_private_file(self.ledger)
        with closing(sqlite3.connect(self.ledger)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, cap REAL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO policy VALUES (1, ?)",
                (16.0 if cap is None else cap,),
            )
            stored_cap = db.execute("SELECT cap FROM policy WHERE id=1").fetchone()[0]
            if stored_cap <= 0 or (cap is not None and stored_cap != cap):
                raise JudgeError("Existing budget cap cannot be changed.")
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, reserved REAL, reported REAL, cached INTEGER)"
            )
            # Preserve earlier receipts/reservations; only new calls get an owner.
            if "run_id" not in {
                row[1] for row in db.execute("PRAGMA table_info(calls)")
            }:
                db.execute("ALTER TABLE calls ADD COLUMN run_id TEXT")

    async def generate(self, req):
        if (
            type(req.max_tokens) is not int
            or not 0 < req.max_tokens <= self.max_output_tokens
        ):
            raise JudgeError("Budgeted call exceeds the configured output token limit.")
        if any(not isinstance(m.get("content"), str) for m in req.messages):
            raise JudgeError("Budgeted experiments support text messages only.")
        if req.extra and set(req.extra) - {"response_format", "reasoning"}:
            raise JudgeError("Unbudgeted request extras are prohibited.")
        reasoning = (req.extra or {}).get("reasoning", {"enabled": False})
        if reasoning not in ({"enabled": False}, {"effort": "high"}):
            raise JudgeError("Unsupported budgeted reasoning settings.")
        # Bytes upper-estimate ordinary text tokens; overhead and 2x margin cover
        # template overhead. Unknown/failed calls remain fully reserved; confirmed
        # cost receipts reconcile unused capacity without raising the original cap.
        prompt_bound = (
            len(
                json.dumps(
                    {"messages": req.messages, "extra": req.extra}, ensure_ascii=False
                ).encode()
            )
            + 4096
        )
        reserved = (
            2
            * (
                prompt_bound * self.prompt_price
                + req.max_tokens * self.completion_price
            )
            / 1_000_000
        )
        with closing(sqlite3.connect(self.ledger)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute(
                "SELECT COALESCE(SUM(COALESCE(reported,reserved)),0) FROM calls"
            ).fetchone()[0]
            cap = db.execute("SELECT cap FROM policy WHERE id=1").fetchone()[0]
            own_used = db.execute(
                "SELECT COALESCE(SUM(COALESCE(reported,reserved)),0) FROM calls WHERE run_id=?",
                (self.run_id,),
            ).fetchone()[0]
            if used + reserved > cap or (
                self.incremental_cap is not None
                and own_used + reserved > self.incremental_cap
            ):
                raise RunStopped("budget_exhausted")
            call_id = db.execute(
                "INSERT INTO calls(reserved,run_id) VALUES (?,?)",
                (reserved, self.run_id),
            ).lastrowid
        extra = dict(req.extra or {})
        extra.update(
            provider={
                "max_price": {
                    "prompt": self.prompt_price,
                    "completion": self.completion_price,
                    "request": 0,
                },
                "require_parameters": True,
                "allow_fallbacks": False,
                "data_collection": self.data_collection,
            },
            reasoning=reasoning,
        )
        result = await self.backend.generate(replace(req, extra=extra))
        cost, cached = None, 0
        if isinstance(result, MinimaLlmResponse):
            cached = int(result.cached)
            candidate = 0 if cached else (result.raw or {}).get("usage", {}).get("cost")
            if (
                type(candidate) in (int, float)
                and math.isfinite(candidate)
                and candidate >= 0
            ):
                cost = candidate
        with closing(sqlite3.connect(self.ledger)) as db, db:
            db.execute(
                "UPDATE calls SET reported=?, cached=? WHERE id=?",
                (cost, cached, call_id),
            )
        if cost is not None and cost > reserved:
            # Persistently lock further spending if provider billing breaks bound.
            with closing(sqlite3.connect(self.ledger)) as db, db:
                db.execute("UPDATE policy SET cap=0 WHERE id=1")
            raise RunStopped("budget_locked")
        return result

    def summary(self):
        with closing(sqlite3.connect(self.ledger)) as db:
            n, reserved, reported, unknown, cached, committed = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved),0), COALESCE(SUM(reported),0), "
                "COALESCE(SUM(reported IS NULL),0), COALESCE(SUM(cached),0), "
                "COALESCE(SUM(COALESCE(reported,reserved)),0) FROM calls"
            ).fetchone()
        return dict(
            calls=n,
            reserved_usd=reserved,
            reported_usd=reported,
            unknown_cost_calls=unknown,
            cache_hits=cached,
            committed_usd=committed,
        )

    async def aclose(self):
        await self.backend.aclose()
