"""Normalize report citation formats without judging or scoring them."""

import math

from autojudge_base import Report
from autojudge_base.report import Rag24ReportSentence

from .models import JudgeError


def normalize_report(report: Report) -> dict:
    sentences = []
    for sentence in report.responses:
        if isinstance(sentence, Rag24ReportSentence) and any(
            not 0 <= i < len(report.references or []) for i in sentence.citations or []
        ):
            raise JudgeError("Citation index is outside the supplied reference list.")
    # This accessor retains priority values; the list-only accessor loses ties.
    for i, s in enumerate(report.get_sentences_with_citation_confidences()):
        citations = s.citations or {}
        if any(not math.isfinite(v) for v in citations.values()):
            raise JudgeError("Nonfinite citation priority in input.")
        ordered = sorted(citations, key=citations.get, reverse=True)
        top = (
            [d for d in ordered if citations[d] == citations[ordered[0]]]
            if ordered
            else []
        )
        sentences.append(
            {
                "sentence_id": i,
                "text": s.text,
                "citations": ordered,
                "top_citations": top,
            }
        )
    cited = {d for s in sentences for d in s["citations"]}
    documents = {}
    for d in sorted(cited):
        doc = (report.documents or {}).get(d)
        if doc is not None and doc.text and doc.text.strip():
            documents[d] = doc.get_text()
    return {"sentences": sentences, "documents": documents}
