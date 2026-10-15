"""Citation output validation using the shared completed-response boundary."""

import json
from dataclasses import replace

from .document_pipeline import Verdicts, token_count
from .models import JudgeError
from .structured_completion import StructuredOutputError, complete_structured, parse_structured


def citation_request(backend, request, *, context_tokens, max_output_tokens):
    """Select the original or larger request without admitting network calls."""
    if (type(context_tokens) is not int or context_tokens <= 0
            or type(max_output_tokens) is not int
            or not 8192 <= max_output_tokens <= 65536):
        raise JudgeError('Invalid citation completion budget or repair setting.')

    input_tokens = token_count(json.dumps(
        dict(messages=request.messages, schema=Verdicts.model_json_schema()),
        ensure_ascii=False))
    ceiling = min(max_output_tokens, context_tokens - input_tokens - 2048)
    cached = getattr(backend, 'cached_response', None)
    if cached is not None and cached(request) is None:
        if ceiling < request.max_tokens:
            raise StructuredOutputError('input_overflow')
        request = replace(request, max_tokens=min(32768, ceiling))
    return request, ceiling


async def complete_citation(backend, request, *, context_tokens,
                            max_output_tokens, repair_output=True, expected_ids=None):
    """Reuse receipts, with at most one larger call after completed truncation.

    Transport and ambiguous envelopes are not retried. The original completion
    remains in the cache, and the larger request has its own deterministic key.
    """
    if type(repair_output) is not bool:
        raise JudgeError('Invalid citation completion budget or repair setting.')
    request, ceiling = citation_request(backend, request,
        context_tokens=context_tokens, max_output_tokens=max_output_tokens)

    def validate(result):
        return parse_structured(result, Verdicts)

    def validate_repair(result):
        value = validate(result)
        ids = [claim.id for claim in value.claims]
        if expected_ids is not None and (len(ids) != len(set(ids)) or set(ids) != expected_ids):
            raise StructuredOutputError('claim_ids')
        return value

    def repair(error, _result):
        if error.code != 'truncated' or not repair_output:
            raise error
        if ceiling <= request.max_tokens:
            raise error
        repaired = replace(request,
            request_id=request.request_id + '-output-repair-v1',
            max_tokens=ceiling)
        cached = getattr(backend, 'cached_response', None)
        # The earlier release repaired at 32K. A larger allowed ceiling must
        # not bypass that settled receipt (or its unresolved-call marker).
        if (cached is not None and cached(repaired) is None
                and request.max_tokens < 32768 < ceiling):
            previous = replace(repaired, max_tokens=32768)
            if cached(previous) is not None:
                repaired = previous
        return repaired, validate_repair

    value, _ = await complete_structured(backend, request, validate, repair=repair)
    return value
