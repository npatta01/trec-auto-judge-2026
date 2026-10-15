"""Validate completed model output; allow one distinct, stage-defined repair.

Transport, caching and budget ownership remain with CompletionSession. This
boundary never catches those failures or loops after a failed repair.
"""

from minima_llm import MinimaLlmResponse
from pydantic import ValidationError

from .models import JudgeError


MESSAGES = {
    'missing': 'Missing model completion.',
    'incomplete': 'Incomplete model completion; no answer accepted.',
    'truncated': 'Model completion reached its output token limit.',
    'schema': 'Invalid structured answer; provider text suppressed.',
    'foreign_evidence': 'Invalid evidence references: foreign evidence.',
    'duplicate_evidence': 'Invalid evidence references: duplicate evidence.',
    'invalid_items': 'Invalid rubric requirements.',
    'answer_length': 'Reference exceeds answer length limit; no silent trimming.',
    'revision_changed': 'Reference revision changed the frozen rubric or gaps.',
    'input_overflow': 'Reference revision input exceeds configured limit.',
    'claim_ids': 'Invalid repaired citation claim identifiers.',
}
REPAIRABLE = frozenset({
    'truncated', 'schema', 'foreign_evidence', 'duplicate_evidence',
    'invalid_items', 'answer_length',
})


class StructuredOutputError(JudgeError):
    """Closed, value-free categories suitable for safe status diagnostics."""

    def __init__(self, code):
        if code not in MESSAGES:
            raise ValueError('Unknown structured-output category.')
        self.code = code
        super().__init__(MESSAGES[code])


def parse_structured(result, schema):
    if not isinstance(result, MinimaLlmResponse):
        raise StructuredOutputError('missing')
    raw = result.raw if isinstance(result.raw, dict) else {}
    choices = raw.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise StructuredOutputError('incomplete')
    choice = choices[0]
    message = choice.get('message')
    if isinstance(message, dict) and message.get('refusal'):
        raise StructuredOutputError('incomplete')
    if choice.get('finish_reason') == 'length':
        raise StructuredOutputError('truncated')
    if choice.get('finish_reason') != 'stop':
        raise StructuredOutputError('incomplete')
    try:
        return schema.model_validate_json(result.text)
    except (ValidationError, ValueError, TypeError):
        raise StructuredOutputError('schema') from None


async def complete_structured(backend, request, validate, *, repair=None, audit=None):
    """Return validated output and its receipt, using at most one repair call.

    A repair builder supplies both its request and validator. It can refuse a
    repair by raising the original error; only completed output errors arrive
    here. Existing receipts are never modified.
    """
    result = await backend.generate(request)
    accepted_request, failure_code = request, None
    try:
        value = validate(result)
    except StructuredOutputError as error:
        if repair is None or error.code not in REPAIRABLE:
            raise
        failure_code = error.code
        accepted_request, repair_validate = repair(error, result)
        result = await backend.generate(accepted_request)
        value = repair_validate(result)
    if audit is not None:
        audit.update(initial_request_id=request.request_id,
            accepted_request_id=accepted_request.request_id, repair_failure_code=failure_code,
            repaired=failure_code is not None)
    return value, result
