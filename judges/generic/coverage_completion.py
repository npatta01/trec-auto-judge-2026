"""Exact attribution for one completed coverage-contract correction.

The original question, checklist and answer are regraded, never the bad draft.
Only output identity is constrained; ordinary requests and scoring are unchanged.
"""

import json
from enum import Enum
from typing import Literal, Union

from pydantic import ConfigDict, Field, create_model

from .coverage_join import ChecklistCoverage
from .models import StrictModel


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate coverage object key.")
        value[key] = item
    return value


class KeyedOutput(StrictModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @classmethod
    def model_validate_json(cls, data, **kwargs):
        # Reject duplicates at every depth before a parser can discard them.
        json.loads(data, object_pairs_hook=_unique_object)
        return super().model_validate_json(data, **kwargs)


FORMAT = """\nOutput format only: items is an object with every supplied item ID
as a required key, not a list. Values contain status, claim_ids and reason;
omit id inside values. Select only supplied claim IDs, without repeats.
missing requires []; partial requires a nonempty selection. covered may use []
when omission satisfies the requirement. All other grading rules are unchanged."""


def keyed_coverage(payload):
    """Return deterministic request-local aliases, schema and restoration map."""
    mapping = {f"I{i}": item["id"] for i, item in enumerate(payload["items"], 1)}
    packet = dict(payload, items=[dict(item, id=alias)
        for alias, item in zip(mapping, payload["items"])])
    claim_ids = [c["id"] for c in payload["answer"]["claims"]]
    # A shared enum avoids duplicating the claim vocabulary for every item.
    claim_type = Enum("CoverageClaimId", {cid: cid for cid in claim_ids}, type=str) if claim_ids else str
    missing = create_model("MissingCoverage", __base__=KeyedOutput,
        status=(Literal["missing"], ...),
        claim_ids=(list[claim_type], Field(..., max_length=0)), reason=(str, ...))
    covered = create_model("CoveredCoverage", __base__=KeyedOutput,
        status=(Literal["covered"], ...),
        claim_ids=(list[claim_type], Field(..., **({} if claim_ids else {"max_length": 0}))),
        reason=(str, ...))
    variants = Union[missing, covered]
    if claim_ids:
        partial = create_model("PartialCoverage", __base__=KeyedOutput,
            status=(Literal["partial"], ...),
            claim_ids=(list[claim_type], Field(..., min_length=1)), reason=(str, ...))
        variants = Union[missing, covered, partial]
    items = create_model("KeyedCoverageItems", __base__=KeyedOutput,
        **{alias: (variants, ...) for alias in mapping})
    schema = create_model("KeyedChecklistCoverage", __base__=KeyedOutput,
        answer_present=(bool, ...), items=(items, ...))
    return packet, schema, mapping


def restore_coverage(result, mapping):
    values = result.model_dump(mode="json")["items"]
    return ChecklistCoverage.model_validate(dict(answer_present=result.answer_present,
        items=[dict(id=original, **values[alias]) for alias, original in mapping.items()]))
