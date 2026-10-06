import pytest

from judges.generic.models import JudgeError
from tests.test_shared_track_judge import sample


def grade(ids=None, status='covered'):
    return dict(answer_present=True, items=[dict(id='n', status=status,
                           claim_ids=['C1'] if ids is None else ids,
                           reason='States the requested fact.')])


def test_coverage_packet_has_no_documents_or_support_labels():
    from judges.generic.coverage_join import coverage_packet

    a = sample()
    packet = coverage_packet({'title': 'Fact?'}, [{'id': 'n', 'evidence_ids': ['e'],
        'requirement': 'Fact', 'acceptable_alternatives': []}], a)
    assert packet['answer']['claims'] == [dict(id='C1', text='Fact.'), dict(id='C2', text='Uncited.')]
    assert 'pairs' not in packet and 'evidence_ids' not in packet['items'][0]
    assert a['pairs'][0]['excerpts'] == ['The supported portion.']


@pytest.mark.parametrize('label,uncertain,want', [
    ('supported', False, 'supported'),
    ('partially_supported', False, 'uncertain'),
    ('unsupported', False, 'not_established'),
    ('supported', True, 'uncertain'),
])
def test_join_does_not_promote_partial_or_uncertain_claim(label, uncertain, want):
    from judges.generic.coverage_join import join_coverage

    a = sample()
    a['pairs'][0].update(label=label, uncertain=uncertain)
    links = join_coverage(a, grade(), ['n'])
    assert links[0]['status'] == want
    assert links[0]['mentioned']
    assert a['pairs'][0]['label'] == label


def test_partial_coverage_of_fully_supported_claim_gets_partial_grounding():
    from judges.generic.coverage_join import join_coverage

    a = sample()
    a['pairs'][0]['label'] = 'supported'
    assert join_coverage(a, grade(status='partial'), ['n'])[0]['status'] == 'partially_supported'
    # One supported sentence must not license another uncited sentence.
    assert join_coverage(a, grade(['C1', 'C2']), ['n'])[0]['status'] == 'not_established'


@pytest.mark.parametrize('bad', [grade(['C99']), grade(['C1', 'C1']),
    grade([], 'partial'), grade(['C1'], 'missing'), dict(items=[]),
    dict(items=grade()['items'] * 2)])
def test_invalid_coverage_links_fail_closed(bad):
    from judges.generic.coverage_join import join_coverage

    with pytest.raises(JudgeError):
        join_coverage(sample(), bad, ['n'])


def test_requirement_satisfied_by_omission_never_earns_grounded_credit():
    from judges.generic.coverage_join import join_coverage

    link = join_coverage(sample(), grade([], 'covered'), ['n'])[0]
    assert link['status'] == 'not_established'
    assert not link['mentioned']
    assert link['pair_ids'] == [] and link['supported_portion'] == ''


def test_false_empty_answer_grade_is_rejected():
    from judges.generic.coverage_join import join_coverage

    bad = grade([], 'missing')
    bad['answer_present'] = False
    with pytest.raises(JudgeError, match='answer presence'):
        join_coverage(sample(), bad, ['n'])


def test_unrelated_nonempty_answer_can_still_score_zero():
    from judges.generic.coverage_join import join_coverage

    assert join_coverage(sample(), grade([], 'missing'), ['n'])[0]['status'] == 'not_established'


def test_partial_grounding_score_keeps_unknown_at_zero_and_strict_score_separate():
    from judges.generic.coverage_join import checklist_scores

    items = [dict(id=x, kind='factual', priority='core') for x in ['a', 'b', 'c']]
    coverage = dict(items=[dict(id=x, status='covered') for x in ['a', 'b', 'c']])
    links = [dict(item=x, status=s) for x,s in zip(['a','b','c'],
             ['supported','partially_supported','uncertain'])]
    scores = checklist_scores(items, coverage, links)
    assert scores['grounded_recall'] == pytest.approx(1/3)
    assert scores['grounded_partial_recall'] == 0.5


def test_partial_packet_uses_only_selected_saved_evidence_once():
    from judges.generic.coverage_join import partial_packet

    a = sample()
    a['pairs'][0]['label'] = 'partially_supported'
    packet, allowed = partial_packet({}, [dict(id='n', kind='factual')], a,
                                    grade(), [dict(item='n', status='uncertain')])
    assert packet['excerpts'] == {'E1': 'The supported portion.'}
    assert packet['answer']['pairs'][0]['excerpt_ids'] == ['E1']
    assert allowed == {'n': {a['pairs'][0]['id']}}
    assert packet['items'][0]['pair_ids'] == [a['pairs'][0]['id']]
    assert 'Uncited.' not in str(packet)
    a['pairs'][0]['uncertain'] = True
    packet, allowed = partial_packet({}, [dict(id='n', kind='factual')], a,
                                    grade(), [dict(item='n', status='uncertain')])
    assert not packet['items'] and not allowed


def test_resolved_links_cannot_borrow_another_route_or_exceed_coverage():
    from judges.generic.coverage_join import merge_partial_links

    a = sample()
    link = dict(a['links'][0], status='supported', missing_or_unsupported='')
    merged = merge_partial_links(a, grade(status='partial'), [link], [link],
                                 {'n': {a['pairs'][0]['id']}})
    assert merged[0]['status'] == 'partially_supported'
    with pytest.raises(JudgeError, match='route'):
        merge_partial_links(a, grade(), [link], [link], {'n': set()})
    with pytest.raises(JudgeError):
        merge_partial_links(a, grade(), [link], [], {'n': {a['pairs'][0]['id']}})
