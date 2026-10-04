"""Strict model-output contracts. Validation errors must never echo report text."""

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
        super().__init__(code)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
