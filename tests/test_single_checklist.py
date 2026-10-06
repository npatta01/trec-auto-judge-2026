import asyncio
import json

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.models import JudgeError


def item(id, kind='factual', priority='core'):
    return dict(id=id, question='What is needed?', requirement='State the color.',
                acceptable_alternatives=[], evidence_ids=['e'], request_basis='Requested.',
                kind=kind, priority=priority)


def test_reference_generates_one_checklist_with_no_second_rubric():
    from judges.generic.reference_answer import generate_answer
    calls = []
    value = dict(answer=[dict(text='Blue.', evidence_ids=['e'])],
                 checklist=[item('F'), item('R', 'request')], gaps=[])

    class Backend:
        async def generate(self, req):
            calls.append(req)
            return MinimaLlmResponse(req.request_id, json.dumps(value),
                raw={'choices': [{'finish_reason': 'stop'}]})

    result, _ = asyncio.run(generate_answer(Backend(),
        dict(question={'title': 'Color and recommendation?'}, evidence=[{'id': 'e'}]),
        joint_checklist=True))
    assert result.model_dump() == value
    assert len(calls) == 1


def test_core_coverage_and_factual_grounding_use_different_subsets_of_same_list():
    from judges.generic.coverage_join import checklist_scores

    items = [item('F'), item('R', 'request'), item('O', priority='optional')]
    grades = dict(items=[dict(id=id, status=status, claim_ids=['C1'], reason='Reason')
                        for id, status in [('F', 'covered'), ('R', 'partial'), ('O', 'missing')]])
    links = [dict(item='F', status='supported'), dict(item='R', status='supported'),
             dict(item='O', status='not_established')]
    assert checklist_scores(items, grades, links) == dict(
        request_coverage=0.75, grounded_recall=1.0, grounded_partial_recall=1.0,
        factual_core_items=1)
    items[0]['kind'] = 'request'
    assert checklist_scores(items, grades, links)['grounded_recall'] == 0
    assert checklist_scores(items, grades, links)['factual_core_items'] == 0


@pytest.mark.parametrize('change', ['duplicate', 'no_core', 'unknown_evidence'])
def test_joint_checklist_rejects_invalid_structure(change):
    from judges.generic.reference_answer import parse_answer

    value = dict(answer=[dict(text='Blue.', evidence_ids=['e'])], checklist=[item('F')], gaps=[])
    if change == 'duplicate':
        value['checklist'] *= 2
    elif change == 'no_core':
        value['checklist'][0]['priority'] = 'optional'
    else:
        value['checklist'][0]['evidence_ids'] = ['foreign']
    result = MinimaLlmResponse('r', json.dumps(value), raw={'choices': [{'finish_reason': 'stop'}]})
    with pytest.raises(JudgeError):
        parse_answer(result, {'e'}, joint_checklist=True)


def test_nonfactual_requirement_never_becomes_a_grounded_fact():
    from judges.generic.coverage_join import join_coverage
    from tests.test_shared_track_judge import sample

    answer = sample()
    answer['pairs'][0]['label'] = 'supported'
    grade = dict(answer_present=True, items=[dict(id='R', status='covered', claim_ids=['C1'], reason='Recommendation.')])
    link = join_coverage(answer, grade, ['R'], factual_ids=set())[0]
    assert link['mentioned'] and link['status'] == 'not_established'
    assert link['pair_ids'] == []


def test_joint_reference_length_revision_preserves_single_checklist():
    from judges.generic.reference_answer import generate_answer
    calls = []
    items = [item('F')]

    class Backend:
        async def generate(self, req):
            calls.append(req)
            value = dict(answer=[dict(text='Blue and very long.' if len(calls) == 1 else 'Blue.',
                                     evidence_ids=['e'])], checklist=items, gaps=[])
            return MinimaLlmResponse(req.request_id, json.dumps(value),
                                    raw={'choices': [{'finish_reason': 'stop'}]})

    reference, _ = asyncio.run(generate_answer(Backend(), dict(question={'limit': 5},
        evidence=[dict(id='e', text='Blue.')]), joint_checklist=True, repair_length=True))
    assert reference.answer[0].text == 'Blue.'
    assert [i.model_dump() for i in reference.checklist] == items
    assert len(calls) == 2
