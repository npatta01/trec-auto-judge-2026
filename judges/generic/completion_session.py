"""One portable cache and incremental budget identity for dependent stages."""

import hashlib
import asyncio
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from minima_llm import MinimaLlmResponse, MinimaLlmFailure

from .budget import BudgetBackend
from .client import make_backend
from .models import JudgeError, RunStopped
from .private_io import private_directory, quiet_output, write_private_text
from .shared_semantics import claim_checkpoint


def _failure_diagnostics(result):
    """Allowlist operational fields; failure messages/bodies may echo inputs."""
    diagnostic = dict(stage="completion", exception_type="UnexpectedResult")
    if isinstance(result, MinimaLlmFailure):
        known = {"HTTPError", "TimeoutError", "JSONDecodeError", "MalformedResponse"}
        diagnostic.update(
            exception_type="MinimaLlmFailure",
            failure_type=(result.error_type
                          if isinstance(result.error_type, str) and result.error_type in known
                          else "other"),
            http_status=(result.status
                         if type(result.status) is int and 100 <= result.status <= 599
                         else None),
            attempts=(result.attempts
                      if type(result.attempts) is int and result.attempts > 0 else None),
            timeout_s=(result.timeout_s
                       if type(result.timeout_s) in (int, float)
                       and math.isfinite(result.timeout_s) and result.timeout_s > 0
                       else None),
        )
    return diagnostic


class CompletionSession:
    def __init__(self, config, ledger, run_budget, factory=make_backend):
        self.backend = factory(config)
        self.cfg = self.backend.cfg
        max_outstanding = getattr(self.cfg, "max_outstanding", 32)
        if type(max_outstanding) is not int or max_outstanding < 1:
            raise JudgeError(
                "The outstanding completion limit must be a positive integer."
            )
        if not self.cfg.cache_dir or getattr(self.cfg, "force_refresh", False):
            raise JudgeError("Unified judging requires CACHE_DIR and refresh disabled.")
        self.directory = Path(self.cfg.cache_dir) / "unified-completions"
        private_directory(self.directory)
        self.ledger, self.run_budget = ledger, run_budget
        self.run_id = "unified-" + uuid4().hex
        self._prompt_locks = {}
        self._admission = asyncio.Semaphore(max_outstanding)

    async def generate(self, request):
        # Duplicate prompts share one receipt even when scheduled concurrently.
        fingerprint_key = hashlib.sha256(json.dumps([request.messages, request.extra, request.max_tokens, request.temperature], sort_keys=True).encode()).hexdigest()
        lock = self._prompt_locks.setdefault(fingerprint_key, asyncio.Lock())
        async with lock:
            return await self._generate(request)

    def _cached_receipt(self, request):
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
            return fingerprint, path, MinimaLlmResponse(
                request.request_id, saved["text"], raw=saved["raw"], cached=True)
        return fingerprint, path, None

    def cached_response(self, request):
        """Read an existing receipt without admitting a new request."""
        _, path, result = self._cached_receipt(request)
        if result is None and path.with_suffix('.started').exists():
            raise JudgeError('Unresolved completion marker; no automatic resend.')
        return result

    async def _generate(self, request):
        fingerprint, path, result = self._cached_receipt(request)
        if result is not None:
            return result
        async with self._admission:
            # Another session may have written the receipt while this request
            # waited for capacity; avoid reserving budget or contacting provider.
            fingerprint, path, result = self._cached_receipt(request)
            if result is not None:
                return result
            return await self._generate_uncached(request, fingerprint, path)

    async def _generate_uncached(self, request, fingerprint, path):
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
                max_output_tokens=65536,
            )
            self.backend.run_id = self.run_id
        started = path.with_suffix(".started")
        claim_checkpoint(started, fingerprint)
        try:
            with quiet_output():
                result = await self.backend.generate(request)
        except RunStopped as error:
            if error.code == "budget_exhausted":
                started.unlink()  # This call was refused before reservation/network.
            raise
        except Exception as error:
            write_private_text(
                self.directory / 'safe-failure.json',
                json.dumps(dict(stage='completion', exception_type=type(error).__name__)),
            )
            raise RunStopped("transport", cause=error) from None
        if not isinstance(result, MinimaLlmResponse):
            error = RunStopped("transport")
            error.args = ("Completion failed; no automatic resend.",)
            error.diagnostics = _failure_diagnostics(result)
            # Append per-attempt diagnostics rather than overwriting another call.
            diagnostic_path = path.with_suffix(f".failure-{uuid4().hex}.json")
            fd = os.open(diagnostic_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(error.diagnostics, stream, allow_nan=False)
            raise error from None
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
