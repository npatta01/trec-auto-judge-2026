import pytest


def test_source_spans_are_reconstructed_not_generated():
    from judges.generic.spans import source_units, materialize
    from judges.generic.models import SpanAudit
    text = 'Alpha is red. Beta is blue. Gamma is green.'
    units = source_units(text, 15)
    assert ''.join(u['text'] for u in units) == text
    audit = SpanAudit.model_validate({'evidence': [{'sentence_id': 0, 'document_id': 'd',
        'status': 'supported', 'source_ids': [0, 2]}]})
    evidence = materialize(audit, text, 'd', {0}, width=15)
    assert evidence[0].quote == text


@pytest.mark.parametrize('ids', [[], [999], [-1], [0, 0]])
def test_invalid_positive_source_references_fail(ids):
    from judges.generic.spans import materialize
    from judges.generic.models import SpanAudit, JudgeError
    from pydantic import ValidationError
    with pytest.raises((JudgeError, ValidationError)):
        audit = SpanAudit.model_validate({'evidence': [{'sentence_id': 0, 'document_id': 'd',
            'status': 'supported', 'source_ids': ids}]})
        materialize(audit, 'blue', 'd', {0})


def test_unsupported_needs_no_source_span():
    from judges.generic.spans import materialize
    from judges.generic.models import SpanAudit
    audit = SpanAudit.model_validate({'evidence': [{'sentence_id': 0, 'document_id': 'd',
        'status': 'unsupported', 'source_ids': []}]})
    assert materialize(audit, 'blue', 'd', {0})[0].quote == ''
