"""Synthetic tests, with no evaluation-run contents or provider credentials."""
import json

import pytest
from autojudge_base import Report, Request


def report(sentences=None, documents=None):
    return Report.model_validate({
        'metadata': {'run_id': 'synthetic', 'team_id': 'hidden-team', 'topic_id': 't'},
        'responses': sentences if sentences is not None else [
            {'text': 'The sample is blue.', 'citations': {'d1': 100., 'd2': 100.}},
            {'text': 'A second factual claim.', 'citations': {}},
            {'text': 'Conclusion.', 'citations': {}},
        ],
        'documents': documents if documents is not None else {
            'd1': {'id': 'd1', 'text': 'The sample is blue.'},
            'd2': {'id': 'd2', 'text': 'The sample is red.'},
        },
    })


def test_citation_proxies_keep_ties_and_uncited_claims():
    from judges.generic.models import Assessment, Evidence
    from judges.generic.scoring import normalize_report, scores
    normalized = normalize_report(report())
    assert normalized['sentences'][0]['citations'] == ['d1', 'd2']
    assert normalized['sentences'][0]['top_citations'] == ['d1', 'd2']
    assessment = Assessment.model_validate({
        'usefulness': 3, 'request_coverage': 2, 'ambiguity_handling': 4,
        'sentences': [{'sentence_id': i, 'needs_citation': n}
                      for i, n in enumerate([True, True, False])],
    })
    evidence = [Evidence(sentence_id=0, document_id='d1', status='supported', quote='The sample is blue.'),
                Evidence(sentence_id=0, document_id='d2', status='contradicted', quote='The sample is red.')]
    got = scores(normalized, assessment, evidence)
    assert got['USEFULNESS'] == .75
    assert got['REQUEST_COVERAGE'] == .5
    assert got['CITE_PRECISION_PROXY'] == .5
    assert got['CITE_RECALL_PROXY'] == .5
    assert got['TOP1_SUPPORT_PROXY'] == .5
    assert got['TOP_TIE_MEAN_PROXY'] == .25
    assert got['EVIDENCE_AVAILABILITY'] == 1.


def test_missing_document_is_unverified_not_supported():
    from judges.generic.scoring import normalize_report
    got = normalize_report(report(documents={}))
    assert got['documents'] == {}
    assert got['sentences'][0]['citations'] == ['d1', 'd2']


def test_full_request_includes_background_but_not_run_identity():
    from judges.generic.scoring import request_payload, normalize_report
    q = Request(request_id='t', title='Compare options', background='For a school',
                problem_statement='Include costs', limit=100)
    assert request_payload(q)['background'] == 'For a school'
    assert request_payload(q)['limit'] == 100
    text = json.dumps(normalize_report(report()))
    assert 'hidden-team' not in text
    assert 'run_id' not in text


def test_fabricated_quote_and_missing_sentence_labels_fail_safely():
    from judges.generic.models import Assessment, Evidence, JudgeError
    from judges.generic.scoring import validate_evidence, scores, normalize_report
    e = Evidence(sentence_id=0, document_id='d1', status='supported', quote='SECRET invented span')
    with pytest.raises(JudgeError) as error:
        validate_evidence([e], {(0, 'd1')}, {'d1': 'The sample is blue.'})
    assert 'SECRET' not in str(error.value)
    a = Assessment(usefulness=4, request_coverage=4, ambiguity_handling=4, sentences=[])
    with pytest.raises(JudgeError):
        scores(normalize_report(report()), a, [])


@pytest.mark.parametrize('value', [True, 5, -1, 2.5, '4', float('nan')])
def test_invalid_grades_are_not_coerced(value):
    from judges.generic.models import Assessment
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Assessment(usefulness=value, request_coverage=2, ambiguity_handling=2, sentences=[])


def test_out_of_range_index_is_not_silently_dropped():
    from judges.generic.models import JudgeError
    from judges.generic.scoring import normalize_report
    r = report(sentences=[{'text': 'Claim.', 'citations': [99]}])
    with pytest.raises(JudgeError, match='index'):
        normalize_report(r)
