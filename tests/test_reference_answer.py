"""Synthetic checks: unknown citations and truncated answers must never pass."""
import asyncio
import json

import pytest
from minima_llm import MinimaLlmResponse


def bundle():
    return dict(answer=[dict(text='Elm costs $2.', evidence_ids=['e1'])],
        rubric=[dict(id='r1', requirement='Explain price.', priority='core',
            reason='The question explicitly asks for price.', evidence_ids=['e1'])],
        gaps=['No duration evidence.'])


def test_reference_uses_full_question_and_keeps_partial_labels():
    from judges.generic.reference_answer import make_request
    packet = dict(question=dict(title='Route?', background='Budget $3',
        problem_statement='Compare duration.'), claims=[dict(text='Elm costs $2.',
        evidence=[dict(id='e1', status='partial')])],
        evidence=[dict(id='e1', document_id='d1', text='Elm costs $2.')])
    req = make_request(packet)
    assert json.loads(req.messages[1]['content']) == packet


@pytest.mark.parametrize('finish', ['length', 'error', 'content_filter', None])
def test_incomplete_provider_answer_rejected_even_with_valid_json(finish):
    from judges.generic.reference_answer import parse_answer
    from judges.generic.models import JudgeError
    result = MinimaLlmResponse('r', json.dumps(bundle()),
        raw={'choices': [{'finish_reason': finish}]})
    with pytest.raises(JudgeError, match='completion'):
        parse_answer(result, {'e1'})


@pytest.mark.parametrize('ids', [['unknown'], [], ['e1', 'e1']])
def test_invalid_citations_are_rejected(ids):
    from judges.generic.reference_answer import parse_answer
    from judges.generic.models import JudgeError
    value = bundle()
    value['answer'][0]['evidence_ids'] = ids
    result = MinimaLlmResponse('r', json.dumps(value),
        raw={'choices': [{'finish_reason': 'stop'}]})
    with pytest.raises(JudgeError):
        parse_answer(result, {'e1'})


def test_one_call_produces_traceable_answer_and_preserves_gaps():
    from judges.generic.reference_answer import generate_answer
    class Backend:
        async def generate(self, req):
            payload = json.loads(req.messages[1]['content'])
            assert payload['evidence'][0]['text'] == 'Elm costs $2.'
            return MinimaLlmResponse(req.request_id, json.dumps(bundle()),
                raw={'choices': [{'finish_reason': 'stop'}]})
    packet = dict(question={'title': 'Price and duration?'}, claims=[],
        evidence=[dict(id='e1', document_id='d1', text='Elm costs $2.')])
    answer, _ = asyncio.run(generate_answer(Backend(), packet))
    assert answer.model_dump() == bundle()


def test_character_limit_is_checked_without_silent_trimming():
    from judges.generic.reference_answer import parse_answer
    from judges.generic.models import JudgeError
    result = MinimaLlmResponse('r', json.dumps(bundle()),
        raw={'choices': [{'finish_reason': 'stop'}]})
    with pytest.raises(JudgeError, match='length'):
        parse_answer(result, {'e1'}, character_limit=5)


@pytest.mark.parametrize('repair_text,passes', [('Elm $2.', True), ('Still much too long.', False)])
def test_overlength_reference_gets_one_bounded_revision(repair_text, passes):
    from judges.generic.reference_answer import generate_answer
    from judges.generic.models import JudgeError
    calls = []

    class Backend:
        async def generate(self, req):
            calls.append(req)
            value = bundle()
            if len(calls) > 1:
                value['answer'][0]['text'] = repair_text
            return MinimaLlmResponse(req.request_id, json.dumps(value),
                raw={'choices': [{'finish_reason': 'stop'}]})

    packet = dict(question={'title': 'Price?', 'limit': 8},
                  evidence=[dict(id='e1', text='Elm costs $2.')])
    if passes:
        answer, _ = asyncio.run(generate_answer(Backend(), packet, repair_length=True))
        assert answer.answer[0].text == 'Elm $2.'
        assert answer.rubric[0].model_dump() == bundle()['rubric'][0]
    else:
        with pytest.raises(JudgeError, match='length'):
            asyncio.run(generate_answer(Backend(), packet, repair_length=True))
    assert len(calls) == 2


def test_length_revision_cannot_change_rubric():
    from judges.generic.reference_answer import generate_answer
    from judges.generic.models import JudgeError
    calls = []

    class Backend:
        async def generate(self, req):
            calls.append(req)
            value = bundle()
            if len(calls) > 1:
                value['answer'][0]['text'] = '$2'
                value['rubric'][0]['requirement'] = 'Different requirement'
            return MinimaLlmResponse(req.request_id, json.dumps(value),
                raw={'choices': [{'finish_reason': 'stop'}]})

    with pytest.raises(JudgeError, match='frozen rubric'):
        asyncio.run(generate_answer(Backend(), dict(question={'limit': 8},
            evidence=[dict(id='e1', text='Elm costs $2.')]), repair_length=True))


def test_optional_details_do_not_lower_core_coverage():
    from judges.generic.reference_answer import parse_grade, ReferenceBundle
    reference = bundle()
    reference['rubric'].append(dict(id='r2', requirement='Mention color.',
        priority='optional', reason='Not requested.', evidence_ids=[]))
    result = MinimaLlmResponse('r', json.dumps(dict(items=[
        dict(id='r1', status='covered', answer_sentences=[0], reason='Price stated.'),
        dict(id='r2', status='missing', answer_sentences=[], reason='No color.')],
        summary='Core need covered.')), raw={'choices': [{'finish_reason': 'stop'}]})
    _, score = parse_grade(result, ReferenceBundle.model_validate(reference), 1)
    assert score == 1.0


def test_grader_cannot_silently_omit_a_core_need():
    from judges.generic.reference_answer import parse_grade, ReferenceBundle
    from judges.generic.models import JudgeError
    result = MinimaLlmResponse('r', json.dumps(dict(items=[], summary='Good.')),
        raw={'choices': [{'finish_reason': 'stop'}]})
    with pytest.raises(JudgeError):
        parse_grade(result, ReferenceBundle.model_validate(bundle()), 1)


def test_grader_gets_measured_normalized_length_not_a_guess():
    from judges.generic.reference_answer import grade_request, ReferenceBundle
    request = grade_request({'title': 'Price?', 'limit': 4},
        ReferenceBundle.model_validate(bundle()), ['ＡB', 'C'])
    measured = json.loads(request.messages[1]['content'])['answer_length']
    assert measured == dict(nfkc_characters=4, words=2, character_limit=4,
        word_limit=None, within_limit=True)


def test_conflicting_evidence_and_qualified_forecast_survive_into_grading():
    from judges.generic.reference_answer import parse_answer, grade_request
    value = bundle()
    value['answer'] = [dict(text='Sources disagree on the April fare; a July cut is possible.',
        evidence_ids=['e1', 'e2', 'e3'])]
    value['gaps'] = ['e1 reports $2 and e2 $3 for the same April service; unresolved.',
        'e3 forecasts a possible July cut, not an enacted change.']
    result = MinimaLlmResponse('r', json.dumps(value),
        raw={'choices': [{'finish_reason': 'stop'}]})
    reference = parse_answer(result, {'e1', 'e2', 'e3'})
    request = grade_request({'title': 'Explain the fare and outlook.'}, reference,
        ['The April fare is disputed; July fares might fall.'])
    payload = json.loads(request.messages[1]['content'])
    assert payload['reference']['answer'][0]['evidence_ids'] == ['e1', 'e2', 'e3']
    assert payload['reference']['answer'][0]['text'] == (
        'Sources disagree on the April fare; a July cut is possible.')
    assert payload['reference']['gaps'] == value['gaps']


def test_uncertainty_does_not_override_partial_or_missing_coverage():
    from judges.generic.reference_answer import parse_grade, ReferenceBundle
    reference = bundle()
    reference['rubric'].append(dict(id='r2', requirement='Explain duration.',
        priority='core', reason='Explicitly requested.', evidence_ids=[]))
    result = MinimaLlmResponse('r', json.dumps(dict(items=[
        dict(id='r1', status='partial', answer_sentences=[0],
            reason='Notes price uncertainty but does not describe the available options.'),
        dict(id='r2', status='missing', answer_sentences=[], reason='No duration discussion.')],
        summary='Uncertainty alone does not answer both needs.')),
        raw={'choices': [{'finish_reason': 'stop'}]})
    grade, score = parse_grade(result, ReferenceBundle.model_validate(reference), 1)
    assert [item.status for item in grade.items] == ['partial', 'missing']
    assert score == .25


def test_sol_price_and_output_allowance_are_reserved_before_network(tmp_path):
    from types import SimpleNamespace
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import JudgeError
    from minima_llm import MinimaLlmRequest
    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url='https://openrouter.ai/api/v1')
        async def generate(self, req):
            assert req.extra['provider']['max_price'] == dict(prompt=2, completion=10, request=0)
            return MinimaLlmResponse(req.request_id, '{}', raw={})
    req = MinimaLlmRequest('r', [{'role': 'user', 'content': 'hello'}], max_tokens=16384)
    backend = BudgetBackend(Backend(), tmp_path/'budget.db', incremental_cap=.36,
        prompt_price=2, completion_price=10, max_output_tokens=16384)
    asyncio.run(backend.generate(req))
    assert .34 < backend.summary()['committed_usd'] < .36
    with pytest.raises(JudgeError, match='budget'):
        asyncio.run(backend.generate(req))


def test_contributor_data_policy_is_explicit_and_default_stays_private(tmp_path):
    from types import SimpleNamespace
    from judges.generic.budget import BudgetBackend
    from minima_llm import MinimaLlmRequest
    class Backend:
        cfg = SimpleNamespace(max_attempts=1, base_url='https://openrouter.ai/api/v1')
        async def generate(self, req):
            return MinimaLlmResponse(req.request_id, req.extra['provider']['data_collection'],
                raw={'usage': {'cost': 0}})
    req = MinimaLlmRequest('r', [{'role': 'user', 'content': 'public fixture'}], max_tokens=100)
    private = BudgetBackend(Backend(), tmp_path/'private.db')
    contributor = BudgetBackend(Backend(), tmp_path/'contributor.db', data_collection='allow')
    assert asyncio.run(private.generate(req)).text == 'deny'
    assert asyncio.run(contributor.generate(req)).text == 'allow'
