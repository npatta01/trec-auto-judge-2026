import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.models import JudgeError


def packet():
    return dict(
        question={"title": "Explain the result", "limit": 2000},
        checklist={
            "items": [
                {"requirement": "Explain the result", "evidence_ids": ["private"]}
            ]
        },
        reference={"answer": [{"text": "Reference fact", "evidence_ids": ["private"]}]},
        answers=[
            dict(
                id="private-one",
                sentences=[dict(text="Fact", citations=["doc"])],
                pairs=[
                    dict(
                        sentence_index=0,
                        document_id="doc",
                        text="Fact",
                        label="supported",
                        uncertain=False,
                        excerpts=["Fact"],
                    )
                ],
            ),
            dict(
                id="private-two", sentences=[dict(text="Other", citations=[])], pairs=[]
            ),
        ],
    )


class Backend:
    cfg = SimpleNamespace(model="test", base_url="http://localhost")

    def __init__(self, winners=("A", "B"), finish="stop"):
        self.winners = iter(winners)
        self.requests = []
        self.finish = finish

    async def generate(self, request):
        self.requests.append(request)
        return MinimaLlmResponse(
            request_id=request.request_id,
            text=json.dumps(
                dict(winner=next(self.winners), reason="Material difference.")
            ),
            raw={"choices": [{"finish_reason": self.finish}]},
        )


def test_blind_has_no_auxiliary_data_and_hints_follow_swap():
    from judges.generic.pairwise import make_payload

    p = packet()
    p["question"].update(
        organizer_score=99, request_id="private-topic", background=None
    )
    blind = make_payload(p, 0, 1, "blind")
    assert set(blind) == {"question", "A", "B"}
    assert "private" not in json.dumps(blind)
    assert "excerpts" not in json.dumps(blind)
    assert blind["question"] == {"title": "Explain the result", "limit": 2000}
    hint = make_payload(p, 1, 0, "hints")
    assert hint["B"]["evidence"][0]["excerpts"] == ["Fact"]
    assert hint["A"]["evidence"] == []
    assert hint["B"]["evidence"][0]["citation"] == "B-D1"
    assert "private" not in json.dumps(hint)
    assert "reference" in make_payload(p, 0, 1, "reference")


@pytest.mark.parametrize(
    "winners,score,disagreement,ties",
    [
        (("A", "B"), 1, False, 0),
        (("A", "A"), 0.5, True, 0),
        (("tie", "tie"), 0.5, False, 2),
        (("A", "tie"), 0.75, True, 1),
    ],
)
def test_swapped_scores_and_ties_are_distinct(
    tmp_path, winners, score, disagreement, ties
):
    from judges.generic.pairwise import judge_topic

    backend = Backend(winners)
    result = asyncio.run(judge_topic(packet(), backend, tmp_path, mode="blind"))
    assert result["scores"] == {"private-one": score, "private-two": 1 - score}
    assert result["pairs"][0]["order_disagreement"] is disagreement
    assert result["pairs"][0]["declared_ties"] == ties
    # Completed receipts replay without touching the endpoint.
    assert (
        asyncio.run(judge_topic(packet(), Backend(()), tmp_path, mode="blind"))
        == result
    )


@pytest.mark.parametrize(
    "change", ["duplicate", "too_many", "context", "stale_hint", "missing_reference"]
)
def test_invalid_input_fails_before_spending(tmp_path, change):
    from judges.generic.pairwise import judge_topic

    p, backend = packet(), Backend()
    kwargs = dict(mode="hints")
    if change == "duplicate":
        p["answers"][1]["id"] = p["answers"][0]["id"]
    if change == "too_many":
        kwargs["max_pairs"] = 0
    if change == "context":
        kwargs["max_input_tokens"] = 1
    if change == "stale_hint":
        p["answers"][0]["pairs"][0]["text"] = "Wrong"
    if change == "missing_reference":
        kwargs["mode"] = "reference"
        del p["reference"]
    with pytest.raises(JudgeError):
        asyncio.run(judge_topic(p, backend, tmp_path, **kwargs))
    assert not backend.requests


def test_truncated_result_rejected_and_not_resent(tmp_path):
    from judges.generic.pairwise import judge_topic

    with pytest.raises(JudgeError):
        asyncio.run(judge_topic(packet(), Backend(finish="length"), tmp_path))
    with pytest.raises(JudgeError):
        asyncio.run(judge_topic(packet(), Backend(()), tmp_path))


def test_round_robin_complete_and_input_order_invariant(tmp_path):
    from judges.generic.pairwise import judge_topic

    p = packet()
    third = copy.deepcopy(p["answers"][1])
    third["id"] = "private-three"
    third["sentences"][0]["text"] = "Third distinct answer"
    p["answers"].append(third)
    result = asyncio.run(judge_topic(p, Backend(("A", "B") * 3), tmp_path))
    assert len(result["pairs"]) == 3
    assert result["scores"] == {
        "private-one": 1,
        "private-three": 0.5,
        "private-two": 0,
    }
    p["answers"].reverse()
    assert asyncio.run(judge_topic(p, Backend(()), tmp_path)) == result


def test_framework_replay_rejects_stale_or_partial_input(tmp_path):
    from autojudge_base import Request
    from judges.generic.pairwise_judge import PairwiseJudge
    from judges.generic.shared_judge import input_digest, sentences
    from judges.generic.pairwise import prompt_input_digest
    from tests.report_fixtures import report

    q = Request(request_id="t", title="Test")
    a = report([dict(text="First", citations={})], {})
    b = report([dict(text="Second", citations={})], {})
    b.metadata.run_id = "second"
    artifact = dict(
        topics=[
            dict(
                topic_id="t",
                inputs=[
                    dict(
                        run_id=r.metadata.run_id,
                        id=str(i),
                        input_sha256=input_digest(r, q),
                        prompt_input_sha256=prompt_input_digest(
                            q.model_dump(mode="json"), sentences(r)
                        ),
                    )
                    for i, r in enumerate((a, b))
                ],
                result=dict(
                    mode="blind",
                    scores={"0": 1.0, "1": 0.0},
                    pairs=[
                        dict(
                            first="0",
                            second="1",
                            forward=dict(winner="A", reason="Better"),
                            reverse=dict(winner="B", reason="Better"),
                            first_points=1.0,
                            order_disagreement=False,
                            declared_ties=0,
                        )
                    ],
                ),
            )
        ]
    )
    path = tmp_path / "pairwise.json"
    path.write_text(json.dumps(artifact))
    judge = PairwiseJudge()
    board = judge.judge([a, b], [q], None, pairwise_bundle=path)
    assert board is not None
    bad = copy.deepcopy(artifact)
    bad["topics"][0]["result"]["scores"]["0"] = 0.3
    path.write_text(json.dumps(bad))
    with pytest.raises(JudgeError):
        judge.judge([a, b], [q], None, pairwise_bundle=path)
    bad = copy.deepcopy(artifact)
    bad["topics"][0]["result"]["pairs"] = []
    path.write_text(json.dumps(bad))
    with pytest.raises(JudgeError):
        judge.judge([a, b], [q], None, pairwise_bundle=path)
    path.write_text(json.dumps(artifact))
    bad = copy.deepcopy(artifact)
    bad["topics"][0]["inputs"][0]["prompt_input_sha256"] = prompt_input_digest(
        q.model_dump(mode="json"), [dict(text="Changed in saved bundle", citations=[])]
    )
    path.write_text(json.dumps(bad))
    with pytest.raises(JudgeError):
        judge.judge([a, b], [q], None, pairwise_bundle=path)
    path.write_text(json.dumps(artifact))
    with pytest.raises(JudgeError):
        judge.judge([a], [q], None, pairwise_bundle=path)
    q.title = "Different question"
    with pytest.raises(JudgeError):
        judge.judge([a, b], [q], None, pairwise_bundle=path)


def test_budget_refusal_can_resume_but_transport_failure_cannot(tmp_path):
    from judges.generic.budget import BudgetBackend
    from judges.generic.models import RunStopped
    from judges.generic.pairwise import judge_topic

    raw = Backend()
    raw.cfg = SimpleNamespace(
        model="test", base_url="https://openrouter.ai/api/v1", max_attempts=1
    )
    ledger = tmp_path / "budget.db"
    first = BudgetBackend(raw, ledger, incremental_cap=0.001)
    checkpoints = tmp_path / "receipts"
    with pytest.raises(RunStopped, match="budget_exhausted"):
        asyncio.run(judge_topic(packet(), first, checkpoints))
    assert not list(checkpoints.glob("*.started"))
    assert not raw.requests
    resumed = BudgetBackend(raw, ledger, incremental_cap=2)
    result = asyncio.run(judge_topic(packet(), resumed, checkpoints))
    assert result["scores"]["private-one"] == 1

    class Failed(Backend):
        async def generate(self, request):
            raise TimeoutError("potentially billed")

    failed = tmp_path / "failed"
    with pytest.raises(JudgeError):
        asyncio.run(judge_topic(packet(), Failed(), failed))
    assert len(list(failed.glob("*.started"))) == 1
    with pytest.raises(JudgeError, match="no automatic resend"):
        asyncio.run(judge_topic(packet(), Backend(), failed))


def test_prompt_digest_accepts_framework_and_compact_question_shapes():
    from autojudge_base import Request
    from judges.generic.pairwise import prompt_input_digest
    def request_payload(q):
        return q.model_dump(exclude_none=True, exclude={'request_id'})

    q = Request(
        request_id="topic", title="Question", background="Important context", limit=1000
    )
    assert prompt_input_digest(q.model_dump(mode="json"), []) == prompt_input_digest(
        request_payload(q), []
    )
    assert prompt_input_digest({"title": "Question"}, []) != prompt_input_digest(
        request_payload(q), []
    )


def test_prepare_cli_to_framework_cli_offline(tmp_path, monkeypatch):
    """Exercise both real CLIs; replace only the external model transport."""
    import subprocess
    import sys
    from autojudge_base import Request
    from judges.generic import pairwise_run
    def request_payload(q):
        return q.model_dump(exclude_none=True, exclude={'request_id'})
    from judges.generic.shared_judge import input_digest, sentences
    from tests.report_fixtures import report

    q = Request(request_id="t", title="Test")
    a = report([dict(text="First", citations={})], {})
    b = report([], {})
    b.metadata.run_id = "second"
    reports = tmp_path / "reports"
    reports.mkdir()
    for i, r in enumerate((a, b)):
        (reports / f"{i}.jsonl").write_text(r.model_dump_json(exclude_none=True) + "\n")
    topics = tmp_path / "topics.jsonl"
    topics.write_text(q.model_dump_json() + "\n")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            dict(
                topics=[
                    dict(
                        topic_id="t",
                        pool=dict(question=request_payload(q)),
                        answers=[
                            dict(
                                run_id=r.metadata.run_id,
                                input_sha256=input_digest(r, q),
                                evidence=dict(
                                    id=f"E{i}", sentences=sentences(r), pairs=[]
                                ),
                            )
                            for i, r in enumerate((a, b))
                        ],
                    )
                ]
            )
        )
    )

    class FakeTransport(Backend):
        cfg = SimpleNamespace(
            model="injected-model",
            base_url="https://openrouter.ai/api/v1",
            max_attempts=1,
        )

        async def aclose(self):
            pass

    monkeypatch.setattr(pairwise_run, "make_backend", lambda _: FakeTransport())
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pairwise_run",
            "--bundle",
            str(bundle),
            "--topic",
            "t",
            "--answers",
            "E0",
            "E1",
            "--mode",
            "blind",
            "--output",
            str(output),
            "--checkpoints",
            str(tmp_path / "receipts"),
            "--ledger",
            str(tmp_path / "budget.db"),
        ],
    )
    pairwise_run.main()
    assert json.loads(output.read_text())["configured_model"] == "injected-model"
    replay = tmp_path / "replay"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "autojudge_base.cli",
            "run",
            "--workflow",
            "judges/generic/pairwise-workflow.yml",
            "--variant",
            "ragtime",
            "--rag-responses",
            str(reports),
            "--rag-topics",
            str(topics),
            "-J",
            f"pairwise_bundle={output}",
            "--out-dir",
            str(replay),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    files = list(replay.glob("*.eval.txt"))
    assert len(files) == 1
    assert "RAGTIME_PAIRWISE_BLIND_WIN_POINTS_PROXY" in files[0].read_text()
