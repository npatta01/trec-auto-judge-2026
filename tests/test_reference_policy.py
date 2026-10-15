"""Prompt wiring, not a claim that offline tests establish semantic accuracy."""
import pytest

from judges.generic.coverage_join import COVERAGE_PROMPT, coverage_packet
from judges.generic.reference_answer import JOINT_REQUIREMENTS, make_request


@pytest.mark.parametrize('compact', [False, True])
def test_same_decision_and_local_evidence_policy_in_both_reference_paths(compact):
    request = make_request(dict(question={}, evidence=[]), joint_checklist=True,
                           validated_claims=compact)
    prompt = request.messages[0]['content']
    assert JOINT_REQUIREMENTS in prompt
    assert 'reasoned deferral' in prompt
    assert 'mandatory actions or factual requirements' in prompt
    assert 'attached to that statement' in prompt
    assert 'recommendations, not proven optima' in prompt


def test_coverage_honors_frozen_routes_without_adding_deferral_alternatives():
    items = [dict(id='N1', kind='request', priority='core',
        question='What should happen in year one?',
        requirement='Launch the required partnership in year one.',
        acceptable_alternatives=['A co-funded launch in year one.'],
        request_basis='The request mandates the first-year launch.', evidence_ids=['E1']),
        dict(id='N2', kind='factual', priority='core', question='Price?',
        requirement='State the price.', acceptable_alternatives=[],
        request_basis='A fact was requested.', evidence_ids=['E1'])]
    packet = coverage_packet({}, items, dict(sentences=[dict(text='Defer it.')]))
    assert packet['items'] == [{k: v for k, v in item.items() if k != 'evidence_ids'}
                               for item in items]
    assert 'frozen requirement or an acceptable alternative' in COVERAGE_PROMPT
    assert 'not a substantive part of a route requiring reasons' in COVERAGE_PROMPT
    assert 'mandatory actions or factual requirements' in COVERAGE_PROMPT


def test_complete_alternative_does_not_inherit_main_requirement_facets():
    assert 'must not inherit extra facets from' in COVERAGE_PROMPT
