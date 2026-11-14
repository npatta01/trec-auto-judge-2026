"""One portable cache and incremental budget identity for dependent stages."""

import hashlib
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from minima_llm import MinimaLlmResponse

from .budget import BudgetBackend
from .client import make_backend
from .models import JudgeError, RunStopped
from .private_io import private_directory, write_private_text
from .shared_semantics import claim_checkpoint


class CompletionSession:
    def __init__(self, config, ledger, run_budget, factory=make_backend):
        self.backend = factory(config)
        self.cfg = self.backend.cfg
        if not self.cfg.cache_dir or getattr(self.cfg, "force_refresh", False):
            raise JudgeError("Unified judging requires CACHE_DIR and refresh disabled.")
        self.directory = Path(self.cfg.cache_dir) / "unified-completions"
        private_directory(self.directory)
        self.ledger, self.run_budget = ledger, run_budget
        self.run_id = "unified-" + uuid4().hex

    async def generate(self, request):
        fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    model=self.cfg.model,
                    messages=request.messages,
                    extra=request.extra,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ),
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = self.directory / (fingerprint + ".json")
        if path.is_symlink():
            raise JudgeError("Symlink completion cache refused.")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["fingerprint"] != fingerprint:
                raise JudgeError("Completion cache mismatch.")
            return MinimaLlmResponse(
                request.request_id, saved["text"], raw=saved["raw"], cached=True
            )
        if self.cfg.base_url == "EMPTY":
            raise JudgeError("Offline completion cache miss.")
        if urlparse(self.cfg.base_url).hostname == "openrouter.ai" and not isinstance(
            self.backend, BudgetBackend
        ):
            self.backend = BudgetBackend(
                self.backend,
                self.ledger,
                incremental_cap=self.run_budget,
                prompt_price=2,
                completion_price=10,
                max_output_tokens=16384,
            )
            self.backend.run_id = self.run_id
        started = path.with_suffix(".started")
        claim_checkpoint(started, fingerprint)
        try:
            with (
                open(os.devnull, "w") as sink,
                redirect_stdout(sink),
                redirect_stderr(sink),
            ):
                result = await self.backend.generate(request)
        except RunStopped as error:
            if error.code == "budget_exhausted":
                started.unlink()  # This call was refused before reservation/network.
            raise
        except Exception:
            raise JudgeError("Completion failed; no automatic resend.") from None
        if not isinstance(result, MinimaLlmResponse):
            raise JudgeError("Completion failed; no automatic resend.")
        write_private_text(
            path,
            json.dumps(
                dict(fingerprint=fingerprint, text=result.text, raw=result.raw),
                ensure_ascii=False,
            ),
        )
        return result

    def summary(self):
        return (
            self.backend.summary() if isinstance(self.backend, BudgetBackend) else None
        )

    async def aclose(self):
        await self.backend.aclose()
