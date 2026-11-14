import pytest

from judges.generic.models import JudgeError


def test_short_pair_ids_round_trip_without_mutating_provenance():
    from judges.generic.shared_semantics import alias_pairs, restore_pair_ids

    original = sample()
    original.pop('links')
    original['pairs'][0]['id'] = 'a' * 64
    aliased, mapping = alias_pairs(original)
    assert aliased['pairs'][0]['id'] == 'P1'
    assert original['pairs'][0]['id'] == 'a' * 64
    links = [dict(pair_ids=['P1'])]
    assert restore_pair_ids(links, mapping)[0]['pair_ids'] == ['a' * 64]
    assert links[0]['pair_ids'] == ['P1']
    with pytest.raises(JudgeError):
        restore_pair_ids([dict(pair_ids=['P01'])], mapping)
    with pytest.raises(JudgeError):
        restore_pair_ids([dict(pair_ids=['a' * 64])], mapping)


def sample():
    return dict(
        id="A",
        sentences=[
            dict(text="Fact.", citations=["d"]),
            dict(text="Uncited.", citations=[]),
        ],
        pairs=[
            dict(
                id="p",
                sentence_index=0,
                document_id="d",
                text="Fact.",
                label="partially_supported",
                uncertain=False,
                excerpts=["The supported portion."],
            )
        ],
        links=[
            dict(
                case="A",
                item="n",
                mentioned=True,
                status="supported",
                pair_ids=["p"],
                supported_portion="Portion",
                missing_or_unsupported="",
            )
        ],
    )


def test_partial_credit_excludes_only_confirmed_nonclaims():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a['sentences'][1]['non_claim'] = True
    r = aggregate(a, ['n'], 'ragtime')
    assert r['values']['RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY'] == 0.5
    assert r['values']['RAGTIME_SENTENCE_SUPPORT_PROXY'] == 0
    a['sentences'][1]['non_claim'] = False
    assert aggregate(a, ['n'], 'ragtime')['values']['RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY'] == 0.25
    a['pairs'][0]['uncertain'] = True
    a['links'][0].update(status='uncertain', pair_ids=[])
    assert aggregate(a, ['n'], 'ragtime')['values']['RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY'] == 0


def test_partial_pair_can_ground_item_but_not_whole_sentence():
    from judges.generic.shared_pipeline import aggregate

    r = aggregate(sample(), ["n"], "ragtime")
    assert r["values"]["RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE"] == 1
    assert r["values"]["RAGTIME_SENTENCE_SUPPORT_PROXY"] == 0
    assert r["audit"]["sentence_precision"] is None
    assert "sentence eligibility" in " ".join(r["unknown_rules"])


def test_rag_uncited_sentence_counts_for_recall_not_precision():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["pairs"][0]["label"] = "supported"
    r = aggregate(a, ["n"], "rag")
    assert r["values"]["RAG_CITATION_PRECISION_UNWEIGHTED_PROXY"] == 1
    assert r["values"]["RAG_CITATION_RECALL_UNWEIGHTED_PROXY"] == 0.5
    assert not any("RAGTIME" in k for k in r["values"])
    assert "weights" in " ".join(r["unknown_rules"])


@pytest.mark.parametrize(
    "change", ["foreign", "duplicate", "missing", "unmentioned", "wrong_text"]
)
def test_invalid_links_and_stale_pair_rejected(change):
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    if change == "foreign":
        a["links"][0]["pair_ids"] = ["elsewhere"]
    if change == "duplicate":
        a["links"] *= 2
    if change == "missing":
        a["links"] = []
    if change == "unmentioned":
        a["links"][0]["mentioned"] = False
    if change == "wrong_text":
        a["pairs"][0]["text"] = "Different claim."
    with pytest.raises(JudgeError):
        aggregate(a, ["n"], "ragtime")


def test_verified_sentence_eligibility_enables_precision():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["pairs"][0]["label"] = "supported"
    a["sentences"][0]["eligible"] = True
    a["sentences"][1]["eligible"] = False
    r = aggregate(a, ["n"], "ragtime")
    assert r["audit"]["sentence_precision"] == 1
    assert r["values"]["RAGTIME_ELIGIBILITY_COMPLETE"] == 1


def test_missing_citation_judgment_is_not_unsupported():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["pairs"] = []
    a["links"][0].update(status="uncertain", pair_ids=[])
    with pytest.raises(JudgeError):
        aggregate(a, ["n"], "ragtime")


def test_supported_pair_without_excerpt_cannot_earn_citation_credit():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["pairs"][0].update(label="supported", excerpts=[])
    a["links"][0].update(status="not_established", pair_ids=[])
    with pytest.raises(JudgeError):
        aggregate(a, ["n"], "ragtime")


def test_partial_pair_blank_excerpt_cannot_ground_item():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["pairs"][0]["excerpts"] = [" "]
    with pytest.raises(JudgeError):
        aggregate(a, ["n"], "ragtime")


def test_tied_citations_keep_submitted_order():
    from judges.generic.shared_judge import sentences
    from tests.report_fixtures import report

    r = report(
        [{"text": "Fact.", "citations": {"z": 100, "a": 100}}],
        {"z": {"id": "z", "text": "Z"}, "a": {"id": "a", "text": "A"}},
    )
    assert sentences(r)[0]["citations"] == ["z", "a"]


def test_empty_answer_remains_zero_and_undefined_precision():
    from judges.generic.shared_pipeline import aggregate

    a = dict(
        id="E",
        sentences=[],
        pairs=[],
        links=[
            dict(
                case="E",
                item="n",
                mentioned=False,
                status="not_established",
                pair_ids=[],
                supported_portion="",
                missing_or_unsupported="empty",
            )
        ],
    )
    r = aggregate(a, ["n"], "ragtime")
    assert r["values"]["RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE"] == 0
    assert r["audit"]["sentence_precision"] is None


def test_cross_answer_link_rejected():
    from judges.generic.shared_pipeline import aggregate

    a = sample()
    a["links"][0]["case"] = "B"
    with pytest.raises(JudgeError):
        aggregate(a, ["n"], "ragtime")


def test_framework_replay_and_stale_input(tmp_path):
    import json
    from types import SimpleNamespace
    from autojudge_base import Request
    from tests.report_fixtures import report
    from judges.generic.shared_judge import SharedJudge, input_digest

    q = Request(request_id="t", title="Test")
    r = report(
        [
            {"text": "Fact.", "citations": {"d": 100}},
            {"text": "Uncited.", "citations": {}},
        ],
        {"d": {"id": "d", "text": "The supported portion."}},
    )
    a = sample()
    bundle = dict(
        topics=[
            dict(
                topic_id="t",
                checklist={"items": [dict(id="n", requirement="Portion")]},
                answers=[
                    dict(
                        run_id=r.metadata.run_id,
                        input_sha256=input_digest(r, q),
                        evidence=a,
                    )
                ],
            )
        ]
    )
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))

    def forbidden(_):
        raise AssertionError("Replay must not call a model")

    judge = SharedJudge(backend_factory=forbidden)
    board = judge.judge(
        [r], [q], SimpleNamespace(), evidence_bundle=str(path), outdir=tmp_path
    )
    assert board is not None
    saved = json.loads((tmp_path / "shared.audit.json").read_text())
    assert saved[0]["values"]["RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE"] == 1
    assert (tmp_path / "shared.audit.json").stat().st_mode & 0o777 == 0o600
    q.title = "Different"
    with pytest.raises(JudgeError):
        judge.judge(
            [r], [q], SimpleNamespace(), evidence_bundle=str(path), outdir=tmp_path
        )


def test_source_mismatch_is_rejected_even_with_updated_digest(tmp_path):
    import json
    from types import SimpleNamespace
    from autojudge_base import Request
    from tests.report_fixtures import report
    from judges.generic.shared_judge import SharedJudge, input_digest

    q = Request(request_id="t", title="Test")
    r = report(
        [
            {"text": "Fact.", "citations": {"d": 100}},
            {"text": "Uncited.", "citations": {}},
        ],
        {"d": {"id": "d", "text": "Different evidence"}},
    )
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            dict(
                topics=[
                    dict(
                        topic_id="t",
                        checklist={"items": [{"id": "n"}]},
                        answers=[
                            dict(
                                run_id=r.metadata.run_id,
                                input_sha256=input_digest(r, q),
                                evidence=sample(),
                            )
                        ],
                    )
                ]
            )
        )
    )
    with pytest.raises(JudgeError):
        SharedJudge().judge(
            [r], [q], SimpleNamespace(), evidence_bundle=str(path), outdir=tmp_path
        )


def test_semantic_pipeline_generates_and_replays_without_rejudging(tmp_path):
    import asyncio
    import json
    from minima_llm import MinimaLlmResponse
    from judges.generic.shared_semantics import prepare_topic

    calls = []

    class Backend:
        async def generate(self, req):
            calls.append(req.request_id)
            if "checklist" in req.request_id:
                payload = dict(
                    items=[
                        dict(
                            id="n",
                            question="What happened?",
                            requirement="Portion",
                            acceptable_alternatives=[],
                            evidence_ids=["e"],
                            request_basis="Request",
                        )
                    ],
                    gaps=[],
                )
            else:
                payload = dict(links=sample()["links"])
                request = json.loads(req.messages[1]['content'])
                payload['links'][0]['pair_ids'] = [request['answer']['pairs'][0]['id']]
            return MinimaLlmResponse(
                req.request_id,
                json.dumps(payload),
                raw={"choices": [{"finish_reason": "stop"}]},
            )

    a = sample()
    a.pop("links")
    topic = dict(
        topic_id="t",
        pool=dict(
            question={"title": "Question"},
            claims=[],
            evidence=[{"id": "e", "text": "The supported portion."}],
        ),
        answers=[dict(run_id="r", input_sha256="digest", evidence=a)],
    )
    out = asyncio.run(prepare_topic(topic, Backend(), tmp_path))
    assert out["answers"][0]["evidence"]["links"][0]["status"] == "supported"
    assert len(calls) == 2
    asyncio.run(prepare_topic(topic, Backend(), tmp_path))
    assert len(calls) == 2


def test_semantic_pipeline_rejects_truncation(tmp_path):
    import asyncio
    from minima_llm import MinimaLlmResponse
    from judges.generic.shared_semantics import prepare_topic

    class Backend:
        async def generate(self, req):
            return MinimaLlmResponse(
                req.request_id, "{}", raw={"choices": [{"finish_reason": "length"}]}
            )

    topic = dict(
        topic_id="t", pool=dict(question={}, claims=[], evidence=[]), answers=[]
    )
    with pytest.raises(JudgeError):
        asyncio.run(prepare_topic(topic, Backend(), tmp_path))


def test_invalid_saved_pairs_fail_before_any_paid_call(tmp_path):
    import asyncio
    from judges.generic.shared_semantics import prepare_topic

    calls = []

    class Backend:
        async def generate(self, req):
            calls.append(req)
            raise AssertionError("Invalid input must not spend")

    a = sample()
    a.pop("links")
    a["pairs"] = []
    topic = dict(
        pool=dict(question={}, claims=[], evidence=[]),
        answers=[dict(run_id="r", evidence=a)],
    )
    with pytest.raises(JudgeError):
        asyncio.run(prepare_topic(topic, Backend(), tmp_path))
    assert not calls


def test_checkpoint_claim_is_exclusive(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from judges.generic.shared_semantics import claim_checkpoint

    with ProcessPoolExecutor(max_workers=2) as pool:
        attempts = [
            pool.submit(claim_checkpoint, tmp_path / "call.started", "digest")
            for _ in range(2)
        ]
        results = []
        for attempt in attempts:
            try:
                attempt.result()
                results.append("claimed")
            except JudgeError:
                results.append("blocked")
    assert sorted(results) == ["blocked", "claimed"]
