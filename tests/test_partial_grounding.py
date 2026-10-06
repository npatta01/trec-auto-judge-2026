import asyncio
import json
import copy
import pytest
from types import SimpleNamespace

from minima_llm import MinimaLlmResponse

from judges.generic.shared_semantics import prepare_topic
from tests.test_shared_track_judge import sample


def test_resolves_partial_route_without_rejudging_citation_or_replacing_baseline():
    a = sample()
    a.pop("links")
    a["pairs"][0]["label"] = "partially_supported"
    item = dict(
        id="n",
        kind="factual",
        priority="core",
        question="What fact?",
        requirement="Fact.",
        acceptable_alternatives=[],
        evidence_ids=["e"],
        request_basis="Asked.",
    )
    topic = dict(
        pool=dict(question={"title": "What fact?"}, evidence=[dict(id="e")]),
        checklist=dict(items=[item], gaps=[]),
        answers=[dict(run_id="run", evidence=a)],
    )

    class Backend:
        cfg = SimpleNamespace(model="test")
        phases = []

        async def generate(self, req):
            self.phases.append(req.request_id)
            if req.request_id == "checklist-coverage-v2":
                value = dict(
                    answer_present=True,
                    items=[
                        dict(
                            id="n",
                            status="covered",
                            claim_ids=["C1"],
                            reason="Fact stated.",
                        )
                    ],
                )
            else:
                assert req.request_id == "partial-grounding-v1"
                packet = json.loads(req.messages[1]["content"])
                assert packet["excerpts"] == {"E1": "The supported portion."}
                assert packet["items"][0]["pair_ids"] == ["P1"]
                value = dict(
                    links=[
                        dict(
                            case="A",
                            item="n",
                            mentioned=True,
                            status="partially_supported",
                            pair_ids=["P1"],
                            supported_portion="Fact.",
                            missing_or_unsupported="One detail unresolved.",
                        )
                    ]
                )
            return MinimaLlmResponse(
                req.request_id,
                json.dumps(value),
                raw={"choices": [{"finish_reason": "stop"}]},
            )

    backend = Backend()
    result = asyncio.run(
        prepare_topic(
            topic,
            backend,
            None,
            coverage_mode="separate",
            resolve_partial_grounding=True,
        )
    )
    answer = result["answers"][0]
    assert answer["evidence"]["links"][0]["status"] == "uncertain"
    assert answer["resolved_links"][0]["status"] == "partially_supported"
    assert answer["resolved_links"][0]["pair_ids"] == [a["pairs"][0]["id"]]
    assert a["pairs"] == answer["evidence"]["pairs"]
    assert backend.phases == ["checklist-coverage-v2", "partial-grounding-v1"]

    class NoCalls:
        cfg = SimpleNamespace(model="test")

        async def generate(self, request):
            raise AssertionError("Saved resolutions must not call the model.")

    replay = asyncio.run(
        prepare_topic(
            result,
            NoCalls(),
            None,
            coverage_mode="separate",
            resolve_partial_grounding=True,
        )
    )
    assert replay == result
    legacy = copy.deepcopy(result)
    legacy["answers"][0].pop("checklist_coverage")
    from judges.generic.models import JudgeError

    with pytest.raises(JudgeError, match="Saved separate coverage is missing"):
        asyncio.run(
            prepare_topic(
                legacy,
                NoCalls(),
                None,
                coverage_mode="separate",
                resolve_partial_grounding=True,
            )
        )
