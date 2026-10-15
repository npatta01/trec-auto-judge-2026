"""Strict model-output contracts. Validation errors must never echo report text."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class JudgeError(RuntimeError):
    """Sanitized operational error; never contains inputs or provider bodies."""


class RunStopped(JudgeError):
    """Safe fixed category for a failure that should stop further model calls."""

    def __init__(
        self,
        code: Literal[
            "budget_exhausted",
            "budget_locked",
            "configuration",
            "transport",
            "internal",
        ],
        *,
        cause: Exception | None = None,
    ):
        if code not in {
            "budget_exhausted",
            "budget_locked",
            "configuration",
            "transport",
            "internal",
        }:
            raise ValueError("Unknown failure category.")
        self.code = code
        self.diagnostics = None
        if cause is not None:
            locations = []
            trace = cause.__traceback__
            while trace:
                locations.append(f"{Path(trace.tb_frame.f_code.co_filename).name}:{trace.tb_lineno}")
                trace = trace.tb_next
            self.diagnostics = dict(exception_type=type(cause).__name__, source_locations=locations)
        super().__init__(code)


class CitationStageError(RunStopped):
    """Keep the fixed stop category when citation checking aborts synthesis."""

    def __init__(self, code, diagnostics=None):
        super().__init__(code)
        self.diagnostics = diagnostics
        self.args = ("Citation stage incomplete; cannot build evidence.",)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
