from collections import Counter
import pytest
from judges.generic import pairwise

def test_bounded_parallel_judgments_match_serial():
    import asyncio
    import json
    from types import SimpleNamespace
    from minima_llm import MinimaLlmResponse
    packet = dict(question=dict(title='Test'), answers=[
        dict(id=str(i), sentences=[dict(text=str(i), citations=[])]) for i in range(5)])
    class Backend:
        cfg = SimpleNamespace(model='test')
        active = peak = 0
        async def generate(self, req):
            self.active += 1
            self.peak = max(self.peak,self.active)
            await asyncio.sleep(.001)
            self.active -= 1
            return MinimaLlmResponse(req.request_id,json.dumps(dict(winner='tie',reason='same')),
                raw=dict(choices=[dict(finish_reason='stop')]))
    backend = Backend()
    result = asyncio.run(pairwise.judge_topic(packet,backend,None,concurrency=3))
    assert backend.peak == 3
    assert backend.active == 0
    assert set(result['scores'].values()) == {.5}


def test_schedule_is_balanced_nested_and_connected():
    small = pairwise.comparison_schedule(83, 10, 'topic-1')
    large = pairwise.comparison_schedule(83, 16, 'topic-1')
    assert len(small) == 415
    assert len(set(small)) == 415
    assert set(Counter(x for edge in small for x in edge).values()) == {10}
    assert all(a < b for a, b in small)
    assert set(small) <= set(large)
    assert small == pairwise.comparison_schedule(83, 10, 'topic-1')
    assert small != pairwise.comparison_schedule(83, 10, 'topic-2')
    reached = {0}
    while True:
        expanded = reached | {b for a,b in small if a in reached} | {a for a,b in small if b in reached}
        if expanded == reached:
            break
        reached = expanded
    assert len(reached) == 83


def test_small_cohort_uses_all_pairs():
    assert pairwise.comparison_schedule(3, 10, 'x') == [(0,1),(0,2),(1,2)]


def test_odd_sparse_degree_rejected():
    with pytest.raises(pairwise.JudgeError):
        pairwise.comparison_schedule(83, 15, 'x')


def test_sparse_aggregation_validates_schedule_and_orders_strengths():
    ids = ['a','b','c','d','e']
    pairs = pairwise.comparison_schedule(5, 2, 'x')
    rows = [dict(first=ids[i], second=ids[j],
                 forward=dict(winner='A', reason='better'),
                 reverse=dict(winner='B', reason='better')) for i,j in pairs]
    result = pairwise.aggregate_decisions(ids, rows, 'blind', opponents=2, schedule_seed='x')
    assert result['aggregation'] == 'regularized_bradley_terry'
    assert result['scores']['a'] > result['scores']['e']
    assert all(0 <= v <= 1 for v in result['scores'].values())
    with pytest.raises(pairwise.JudgeError):
        pairwise.aggregate_decisions(ids, rows[:-1], 'blind', opponents=2, schedule_seed='x')
