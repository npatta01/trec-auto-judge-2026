"""Numbered text units avoid asking the model to reproduce exact source quotes."""
from .models import Evidence, JudgeError
from .scoring import validate_evidence


def source_units(text, width=400):
    if type(width) is not int or width <= 0:
        raise JudgeError('Source-unit width must be positive.')
    return [{'source_id': i // width, 'text': text[i:i + width]}
            for i in range(0, len(text), width)]


def materialize(audit, text, document_id, sentence_ids, width=400):
    count = len(source_units(text, width))
    evidence = []
    for row in audit.evidence:
        ids = row.source_ids
        if (len(ids) != len(set(ids)) or any(not 0 <= i < count for i in ids)
                or (row.status != 'unsupported' and not ids)):
            raise JudgeError('Source references are missing, duplicated or outside supplied units.')
        # Preserve the contiguous source envelope, including intervening context.
        # Never concatenate separated fragments and call that a verbatim quote.
        quote = text[min(ids) * width:(max(ids) + 1) * width] if ids else ''
        evidence.append(Evidence(sentence_id=row.sentence_id, document_id=row.document_id,
                                 status=row.status, quote=quote))
    validate_evidence(evidence, {(i, document_id) for i in sentence_ids}, {document_id: text})
    return evidence
