"""Strict model-output contracts. Validation errors must never echo report text."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


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


Grade = Annotated[int, Field(ge=0, le=4)]


class Brief(StrictModel):
    explicit_needs: list[str]
    inferred_needs: list[str]
    ambiguities: list[str]
    questionable_premises: list[str]


class SentenceNeed(StrictModel):
    sentence_id: Annotated[int, Field(ge=0)]
    needs_citation: bool


class Assessment(StrictModel):
    usefulness: Grade
    request_coverage: Grade
    ambiguity_handling: Grade
    sentences: list[SentenceNeed]


class Evidence(StrictModel):
    sentence_id: Annotated[int, Field(ge=0)]
    document_id: str
    status: Literal["supported", "partial", "unsupported", "contradicted", "unverified"]
    quote: str


class Audit(StrictModel):
    evidence: list[Evidence]


class DirectResult(StrictModel):
    assessment: Assessment
    evidence: list[Evidence]


class SpanEvidence(StrictModel):
    sentence_id: Annotated[int, Field(ge=0)]
    document_id: str
    status: Literal["supported", "partial", "unsupported", "contradicted"]
    source_ids: list[Annotated[int, Field(ge=0)]]


class SpanAudit(StrictModel):
    evidence: list[SpanEvidence]
