"""Input normalization, evidence provenance and explicit proxy aggregation."""
import math
from statistics import mean

from autojudge_base import LeaderboardSpec, MeasureSpec, Report, Request
from autojudge_base.report import Rag24ReportSentence

from .models import Assessment, Evidence, JudgeError


DESCRIPTIONS = {
    'USEFULNESS': 'LLM estimate of task usefulness, grade 0..4 / 4; not human preference probability.',
    'REQUEST_COVERAGE': 'LLM estimate of request coverage, grade 0..4 / 4; NOT official nugget recall.',
    'AMBIGUITY_HANDLING': 'LLM estimate of justified assumptions and premise/uncertainty handling, 0..4 / 4.',
    'CITE_PRECISION_PROXY': 'Mean verified support per sentence/citation link; 1 full, .5 partial, 0 otherwise.',
    'CITE_RECALL_PROXY': 'Mean best cited support per citation-required sentence; uncited=0.',
    'TOP1_SUPPORT_PROXY': 'Mean first-priority support per citation-required sentence; ties use submitted order. No hidden-nugget exclusions.',
    'TOP_TIE_MEAN_PROXY': 'Mean support across tied top citations per citation-required sentence. Sensitivity proxy, not official tie rule.',
    'EVIDENCE_AVAILABILITY': 'Fraction of citation links with inspectable document text (diagnostic, not answer quality).',
}
SPEC = LeaderboardSpec(measures=tuple(
    MeasureSpec(k, description=v + ' Range 0..1; empty denominator/report=0; aggregate=topic mean.')
    for k, v in DESCRIPTIONS.items()
))
SUPPORT = {'supported': 1., 'partial': .5, 'unsupported': 0., 'contradicted': 0., 'unverified': 0.}


def request_payload(request: Request) -> dict:
    return request.model_dump(include={
        'title', 'background', 'original_background', 'problem_statement',
        'limit', 'word_limit', 'collection_ids',
    }, exclude_none=True)


def normalize_report(report: Report) -> dict:
    sentences = []
    for sentence in report.responses:
        if isinstance(sentence, Rag24ReportSentence) and any(
            not 0 <= i < len(report.references or []) for i in sentence.citations or []
        ):
            raise JudgeError('Citation index is outside the supplied reference list.')
    # This accessor retains priority values; the list-only accessor loses ties.
    for i, s in enumerate(report.get_sentences_with_citation_confidences()):
        citations = s.citations or {}
        if any(not math.isfinite(v) for v in citations.values()):
            raise JudgeError('Nonfinite citation priority in input.')
        ordered = sorted(citations, key=citations.get, reverse=True)
        top = [d for d in ordered if citations[d] == citations[ordered[0]]] if ordered else []
        sentences.append({'sentence_id': i, 'text': s.text,
                          'citations': ordered, 'top_citations': top})
    cited = {d for s in sentences for d in s['citations']}
    documents = {}
    for d in sorted(cited):
        doc = (report.documents or {}).get(d)
        if doc is not None and doc.text and doc.text.strip():
            documents[d] = doc.get_text()
    return {'sentences': sentences, 'documents': documents}


def validate_evidence(evidence: list[Evidence], expected: set[tuple], documents: dict) -> None:
    keys = [(e.sentence_id, e.document_id) for e in evidence]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise JudgeError('Evidence labels are missing, duplicated or unexpected.')
    for e in evidence:
        source = documents.get(e.document_id)
        if e.status == 'unverified':
            if source is not None or e.quote:
                raise JudgeError('Unverified label is only valid for missing evidence.')
        elif source is None:
            raise JudgeError('Evidence label refers to unavailable document text.')
        elif e.status in ('supported', 'partial', 'contradicted'):
            if not e.quote.strip() or e.quote not in source:
                raise JudgeError('Evidence quote is absent from the inspected source.')
        elif e.quote and e.quote not in source:
            raise JudgeError('Evidence quote is absent from the inspected source.')


def scores(report: dict, assessment: Assessment, evidence: list[Evidence]) -> dict[str, float]:
    sentences = report['sentences']
    ids = [s.sentence_id for s in assessment.sentences]
    if len(ids) != len(set(ids)) or set(ids) != {s['sentence_id'] for s in sentences}:
        raise JudgeError('Citation-needed labels do not cover every sentence exactly once.')
    expected = {(s['sentence_id'], d) for s in sentences for d in s['citations']}
    validate_evidence(evidence, expected, report['documents'])
    need = {s.sentence_id: s.needs_citation for s in assessment.sentences}
    support = {(e.sentence_id, e.document_id): SUPPORT[e.status] for e in evidence}
    recall, top1, tied = [], [], []
    for s in sentences:
        i = s['sentence_id']
        if not need[i]:
            continue
        vals = [support[i, d] for d in s['citations']]
        tops = [support[i, d] for d in s['top_citations']]
        recall.append(max(vals, default=0.))
        top1.append(vals[0] if vals else 0.)
        tied.append(mean(tops) if tops else 0.)
    return {
        'USEFULNESS': assessment.usefulness / 4,
        'REQUEST_COVERAGE': assessment.request_coverage / 4,
        'AMBIGUITY_HANDLING': assessment.ambiguity_handling / 4,
        'CITE_PRECISION_PROXY': mean(support.values()) if support else 0.,
        'CITE_RECALL_PROXY': mean(recall) if recall else 0.,
        'TOP1_SUPPORT_PROXY': mean(top1) if top1 else 0.,
        'TOP_TIE_MEAN_PROXY': mean(tied) if tied else 0.,
        'EVIDENCE_AVAILABILITY': mean(e.status != 'unverified' for e in evidence) if evidence else 0.,
    }
