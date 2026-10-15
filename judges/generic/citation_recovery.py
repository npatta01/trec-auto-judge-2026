"""One exact-key correction after a completed citation ID-contract failure."""

import json
from dataclasses import replace

from pydantic import create_model

from .document_pipeline import Verdict, Verdicts, token_count
from .models import StrictModel
from .structured_completion import StructuredOutputError, parse_structured


Assessment = create_model('ClaimAssessment', __base__=StrictModel, **{
    name: (field.annotation, ...) for name, field in Verdict.model_fields.items()
    if name != 'id'})


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate object key.')
        value[key] = item
    return value


async def recover_citation_ids(backend, request, *, expected_ids,
                               context_tokens, max_output_tokens):
    """Rejudge the whole failed batch once; never salvage ambiguous rows.

    Legacy requests/receipts stay unchanged. A bad correction or unknown outcome
    is terminal, including across resumes through the normal completion cache.
    """
    aliases = list(expected_ids)
    schema_model = create_model('KeyedClaimSupport', __base__=StrictModel,
        **{alias: (Assessment, ...) for alias in aliases})
    schema = schema_model.model_json_schema()
    corrected = replace(request, request_id='document-support-keyed-ids-v1',
        messages=request.messages + [dict(role='user', content=
            'Return a JSON object keyed by the supplied claim IDs: '
            + json.dumps(aliases) + '. Each value contains eligibility, label, '
            'uncertain and contradiction only; do not put an id field in values. '
            'Use the same judging rules. All keys are required; no extra keys.')],
        extra=dict(request.extra or {}, response_format=dict(type='json_schema',
            json_schema=dict(name='KeyedClaimSupport', strict=True, schema=schema))))
    input_tokens = token_count(json.dumps(dict(messages=corrected.messages,
        schema=schema), ensure_ascii=False))
    output_tokens = min(32768, max_output_tokens, context_tokens - input_tokens - 2048)
    if output_tokens < 8192:
        raise StructuredOutputError('input_overflow')
    corrected = replace(corrected, max_tokens=output_tokens)
    result = await backend.generate(corrected)
    value = parse_structured(result, schema_model)
    # A JSON parser otherwise keeps the last duplicate key. Validate before
    # accepting any attribution, including duplicate fields inside values.
    try:
        json.loads(result.text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        raise StructuredOutputError('schema') from None
    return Verdicts(claims=[Verdict(id=alias, **getattr(value, alias).model_dump())
                           for alias in aliases])
