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


class BudgetBackend:
    def __init__(
        self, backend, ledger, cap=16.0, *, incremental_cap=None, completion_price=3
    ):
        if (
            backend.cfg.max_attempts != 1
            or urlparse(backend.cfg.base_url).hostname != "openrouter.ai"
        ):
            raise JudgeError(
                "Budgeted experiments require OpenRouter and exactly one transport attempt."
            )
        if not math.isfinite(cap) or not 0 < cap <= 16:
            raise JudgeError("Experiment budget cap must be positive and at most $16.")
        if completion_price not in (3, 4):
            raise JudgeError("Unsupported completion price ceiling.")
        if incremental_cap is not None and (
            not math.isfinite(incremental_cap) or incremental_cap <= 0
        ):
            raise JudgeError("Incremental budget must be finite and positive.")
        self.cfg = backend.cfg
        self.run_id = uuid4().hex
        self.incremental_cap = incremental_cap
        self.completion_price = completion_price
        self.backend, self.ledger = backend, Path(ledger)
        private_directory(self.ledger.parent)
        prepare_private_file(self.ledger)
        with closing(sqlite3.connect(self.ledger)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, cap REAL)"
            )
            db.execute("INSERT OR IGNORE INTO policy VALUES (1, ?)", (cap,))
            if db.execute("SELECT cap FROM policy WHERE id=1").fetchone()[0] != cap:
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
        if type(req.max_tokens) is not int or not 0 < req.max_tokens <= 8192:
            raise JudgeError(
                "Budgeted call requires an output limit of 1..8192 tokens."
            )
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
            2 * (prompt_bound + req.max_tokens * self.completion_price) / 1_000_000
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
        result = await self.backend.generate(
            with_openrouter_options(req, completion_price=self.completion_price)
        )
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
