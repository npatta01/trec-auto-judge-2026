import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from minima_llm import MinimaLlmResponse

from judges.generic.models import JudgeError
from judges.generic.document_pipeline import token_count
from judges.generic.shared_semantics import (
    PARTIAL_PROMPT,
    alias_pairs,
    prepare_topic,
)
from judges.generic.coverage_join import partial_packet


def _topic(item_count=2):
    question = {
        "title": "How should the submitted answer address the requested facts?",
        "background": "Preserve this complete context. " * 36,
    }
    items = []
    sentences = []
    pairs = []
    original_links = []
    coverage_items = []
    evidence = []
    for index in range(item_count):
        item_id = f"checklist-item-{index + 1}"
        sentence = f"Synthetic factual statement {index + 1}."
        evidence_id = f"evidence-{index + 1}"
        pair_id = f"original-pair-id-{index + 1}-" + "x" * 28
        excerpt = f"Synthetic retained evidence for item {index + 1}. " * 5
        items.append(
            dict(
                id=item_id,
                kind="factual",
                priority="core",
                question=f"What is synthetic fact {index + 1}?",
                requirement=f"Synthetic fact {index + 1} is stated accurately.",
                acceptable_alternatives=[f"Equivalent synthetic route {index + 1}."]
                * 2,
                evidence_ids=[evidence_id],
                request_basis=f"The synthetic question asks for fact {index + 1}.",
            )
        )
        sentences.append(dict(text=sentence, citations=[evidence_id]))
        pairs.append(
            dict(
                id=pair_id,
                sentence_index=index,
                document_id=evidence_id,
                text=sentence,
                label="partially_supported",
                uncertain=False,
                excerpts=[excerpt, f"Second retained excerpt {index + 1}."],
            )
        )
        original_links.append(
            dict(
                case="answer-A",
                item=item_id,
                mentioned=True,
                status="uncertain",
                pair_ids=[],
                supported_portion="",
                missing_or_unsupported="The original partial support is unresolved.",
            )
        )
        coverage_items.append(
            dict(
                id=item_id,
                status="partial",
                claim_ids=[f"C{index + 1}"],
                reason=f"A required synthetic detail is unresolved for {index + 1}.",
            )
        )
        evidence.append(dict(id=evidence_id))

    answer = dict(
        id="answer-A",
        sentences=sentences,
        pairs=pairs,
        links=original_links,
    )
    return dict(
        pool=dict(question=question, evidence=evidence),
        checklist=dict(items=items, gaps=[]),
        answers=[
            dict(
                run_id="synthetic-run",
                evidence=answer,
                checklist_coverage=dict(answer_present=True, items=coverage_items),
            )
        ],
    )


def _aliased_payload(topic):
    answer_record = topic["answers"][0]
    payload, _allowed = partial_packet(
        topic["pool"]["question"],
        topic["checklist"]["items"],
        answer_record["evidence"],
        answer_record["checklist_coverage"],
        answer_record["evidence"]["links"],
    )
    payload["answer"], mapping = alias_pairs(payload["answer"])
    aliases = {original: alias for alias, original in mapping.items()}
    for item in payload["items"]:
        item["pair_ids"] = [aliases[pair_id] for pair_id in item["pair_ids"]]
    return payload, mapping


def _project_items(payload, items):
    projected = copy.deepcopy(payload)
    projected["items"] = copy.deepcopy(items)
    pair_ids = {pair_id for item in items for pair_id in item["pair_ids"]}
    projected["answer"]["pairs"] = [
        pair for pair in projected["answer"]["pairs"] if pair["id"] in pair_ids
    ]
    excerpt_ids = {
        excerpt_id
        for pair in projected["answer"]["pairs"]
        for excerpt_id in pair["excerpt_ids"]
    }
    projected["excerpts"] = {
        excerpt_id: excerpt
        for excerpt_id, excerpt in projected["excerpts"].items()
        if excerpt_id in excerpt_ids
    }
    return projected


def _request_token_count(payload):
    messages = [
        dict(role="system", content=PARTIAL_PROMPT),
        dict(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ),
    ]
    return token_count(json.dumps(messages))


def _response(packet, *, ids=None):
    items = packet["items"] if ids is None else ids
    return dict(
        links=[
            dict(
                case=packet["answer"]["id"],
                item=item["id"],
                mentioned=True,
                status="partially_supported",
                pair_ids=item["pair_ids"],
                supported_portion=item["requirement"],
                missing_or_unsupported="A synthetic detail remains unresolved.",
            )
            for item in items
        ]
    )


class RecordingBackend:
    cfg = SimpleNamespace(model="synthetic-model")

    def __init__(self, response_factory=None):
        self.calls = []
        self.response_factory = response_factory or _response

    async def generate(self, request):
        packet = json.loads(request.messages[1]["content"])
        self.calls.append((request, packet))
        return MinimaLlmResponse(
            request.request_id,
            json.dumps(self.response_factory(packet)),
            raw={"choices": [{"finish_reason": "stop"}]},
        )


def _prepare(topic, backend, checkpoint_dir=None, *, max_input_tokens, **settings):
    return asyncio.run(
        prepare_topic(
            topic,
            backend,
            checkpoint_dir,
            max_input_tokens=max_input_tokens,
            coverage_mode="separate",
            resolve_partial_grounding=True,
            **settings,
        )
    )


def test_partial_grounding_splits_oversized_packet_without_losing_global_aliases():
    topic = _topic()
    original = copy.deepcopy(topic)
    payload, mapping = _aliased_payload(topic)
    single_item_payloads = [
        _project_items(payload, [item]) for item in payload["items"]
    ]
    budget = max(_request_token_count(packet) for packet in single_item_payloads)
    assert _request_token_count(payload) > budget

    backend = RecordingBackend()
    result = _prepare(topic, backend, max_input_tokens=budget)

    assert len(backend.calls) == 2
    assert all(
        request.request_id == "partial-grounding-v1"
        and token_count(json.dumps(request.messages)) <= budget
        for request, _packet in backend.calls
    )
    for (_request, packet), expected in zip(backend.calls, single_item_payloads):
        assert packet == expected
        assert packet["question"] == topic["pool"]["question"]
        assert len(packet["items"]) == 1
        pair = packet["answer"]["pairs"][0]
        assert pair["id"] in mapping
        assert len(pair["excerpt_ids"]) == 2
        assert all(excerpt_id in packet["excerpts"] for excerpt_id in pair["excerpt_ids"])

    resolved = result["answers"][0]["resolved_links"]
    assert [link["item"] for link in resolved] == [
        item["id"] for item in topic["checklist"]["items"]
    ]
    assert [link["pair_ids"] for link in resolved] == [
        [pair["id"]] for pair in original["answers"][0]["evidence"]["pairs"]
    ]
    assert topic == original
    assert result["answers"][0]["evidence"]["pairs"] == original["answers"][0]["evidence"]["pairs"]
    assert result["answers"][0]["evidence"]["links"] == original["answers"][0]["evidence"]["links"]


def test_single_packet_request_stays_byte_identical_when_it_fits():
    topic = _topic()
    payload, _mapping = _aliased_payload(topic)
    expected = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    backend = RecordingBackend()

    _prepare(topic, backend, max_input_tokens=_request_token_count(payload))

    assert len(backend.calls) == 1
    request, _packet = backend.calls[0]
    assert request.messages[1]["content"] == expected


def test_oversized_single_partial_item_fails_before_provider_call():
    topic = _topic()
    topic["checklist"]["items"][1]["requirement"] += " detail" * 3000
    payload, _mapping = _aliased_payload(topic)
    first_item = _project_items(payload, [payload["items"][0]])
    second_item = _project_items(payload, [payload["items"][1]])
    budget = _request_token_count(first_item)
    assert _request_token_count(second_item) > budget
    backend = RecordingBackend()

    with pytest.raises(JudgeError, match="context budget"):
        _prepare(topic, backend, max_input_tokens=budget)

    assert backend.calls == []


def test_single_item_evidence_overflow_reports_only_numeric_guard_diagnostics():
    topic = _topic(1)
    pair = topic["answers"][0]["evidence"]["pairs"][0]
    pair["excerpts"] = ["Synthetic private sentinel retained evidence. " * 5000]
    backend = RecordingBackend()
    with pytest.raises(JudgeError) as failed:
        _prepare(topic, backend, max_input_tokens=2000)

    assert getattr(failed.value, "code", None) == "partial_grounding_input_overflow"
    diagnostics = failed.value.diagnostics
    assert set(diagnostics) == {
        "input_tokens", "max_input_tokens", "without_excerpts_input_tokens",
        "item_count", "pair_count", "excerpt_count", "max_excerpt_tokens",
    }
    assert all(type(value) is int for value in diagnostics.values())
    assert diagnostics["input_tokens"] > 2000
    assert diagnostics["without_excerpts_input_tokens"] < 2000
    assert diagnostics["max_input_tokens"] == 2000
    assert diagnostics["item_count"] == 1
    assert diagnostics["pair_count"] == 1
    assert diagnostics["excerpt_count"] == 1
    assert diagnostics["max_excerpt_tokens"] > 2000
    assert "sentinel" not in str(failed.value) + repr(diagnostics)
    assert backend.calls == []


def test_declared_context_admits_one_oversized_item_with_all_evidence():
    topic = _topic(1)
    original = copy.deepcopy(topic)
    pair = topic["answers"][0]["evidence"]["pairs"][0]
    pair["excerpts"] = ["Synthetic retained evidence. " * 2500]
    backend = RecordingBackend()
    result = _prepare(topic, backend, max_input_tokens=2000,
                      partial_context_tokens=65536)
    assert len(backend.calls) == 1
    request, packet = backend.calls[0]
    assert token_count(json.dumps(request.messages)) > 2000
    assert token_count(json.dumps(request.messages)) + request.max_tokens < 65536
    assert list(packet["excerpts"].values()) == pair["excerpts"]
    assert result["answers"][0]["resolved_links"][0]["pair_ids"] == [pair["id"]]
    assert result["answers"][0]["resolved_links"][0]["status"] == "partially_supported"
    assert result["answers"][0]["evidence"]["pairs"] == topic["answers"][0]["evidence"]["pairs"]
    assert original["answers"][0]["evidence"]["links"] == topic["answers"][0]["evidence"]["links"]


def test_declared_context_keeps_legacy_batch_receipts_without_resend(tmp_path):
    topic = _topic()
    payload, _mapping = _aliased_payload(topic)
    budget = max(_request_token_count(_project_items(payload, [item]))
                 for item in payload["items"])
    original = RecordingBackend()
    expected = _prepare(topic, original, tmp_path, max_input_tokens=budget)
    assert len(original.calls) == 2
    replay = RecordingBackend()
    actual = _prepare(topic, replay, tmp_path, max_input_tokens=budget,
                      partial_context_tokens=65536)
    assert replay.calls == []
    assert actual == expected


def test_declared_context_reserves_output_before_any_provider_call():
    backend = RecordingBackend()
    with pytest.raises(JudgeError, match="context"):
        _prepare(_topic(1), backend, max_input_tokens=2000,
                 partial_context_tokens=20000, partial_max_output_tokens=32768)
    assert backend.calls == []


def test_oversized_item_is_isolated_without_enlarging_neighbor_batches():
    topic = _topic(3)
    middle = topic["answers"][0]["evidence"]["pairs"][1]
    middle["excerpts"] = ["Synthetic middle evidence. " * 2500]
    backend = RecordingBackend()
    result = _prepare(topic, backend, max_input_tokens=1200,
                      partial_context_tokens=65536)
    assert len(backend.calls) == 3
    assert [len(packet["items"]) for _, packet in backend.calls] == [1, 1, 1]
    assert [packet["answer"]["pairs"][0]["id"] for _, packet in backend.calls] == [
        "P1", "P2", "P3",
    ]
    assert _request_token_count(backend.calls[0][1]) <= 1200
    assert _request_token_count(backend.calls[2][1]) <= 1200
    assert _request_token_count(backend.calls[1][1]) > 1200
    assert list(backend.calls[1][1]["excerpts"].values()) == middle["excerpts"]
    assert len(result["answers"][0]["resolved_links"]) == 3


def test_item_exceeding_declared_capacity_still_fails_before_admission():
    topic = _topic(1)
    topic["answers"][0]["evidence"]["pairs"][0]["excerpts"] = [
        "Synthetic oversized source. " * 30000
    ]
    backend = RecordingBackend()
    with pytest.raises(JudgeError) as failed:
        _prepare(topic, backend, max_input_tokens=2000,
                 partial_context_tokens=65536)
    assert failed.value.code == "partial_grounding_input_overflow"
    assert failed.value.diagnostics["input_tokens"] > failed.value.diagnostics["max_input_tokens"]
    assert failed.value.diagnostics["max_input_tokens"] < 65536 - 16384
    assert backend.calls == []


@pytest.mark.parametrize("failure", ["wrong_batch", "duplicate", "missing"])
def test_batch_output_must_route_every_expected_item_exactly_once(failure):
    topic = _topic()
    payload, _mapping = _aliased_payload(topic)
    budget = max(
        _request_token_count(_project_items(payload, [item]))
        for item in payload["items"]
    ) + 64  # Leave room for the bounded repair's routing-only prompt suffix.
    assert _request_token_count(payload) > budget

    def malformed(packet):
        if failure == "wrong_batch":
            other = next(
                item
                for item in payload["items"]
                if item["id"] != packet["items"][0]["id"]
            )
            return _response(packet, ids=[other])
        if failure == "duplicate":
            response = _response(packet)
            response["links"].append(copy.deepcopy(response["links"][0]))
            return response
        return dict(links=[])

    backend = RecordingBackend(malformed)
    with pytest.raises(JudgeError, match="partial grounding batch"):
        _prepare(topic, backend, max_input_tokens=budget)
    assert len(backend.calls) == 2  # One completed bad-ID result, one bounded repair.


def test_completed_batch_is_replayed_without_resend_after_later_batch_failure(tmp_path):
    topic = _topic()
    payload, _mapping = _aliased_payload(topic)
    budget = max(
        _request_token_count(_project_items(payload, [item]))
        for item in payload["items"]
    )

    class FailsOnSecondBatch(RecordingBackend):
        async def generate(self, request):
            if self.calls:
                packet = json.loads(request.messages[1]["content"])
                self.calls.append((request, packet))
                raise RuntimeError("synthetic failure")
            return await super().generate(request)

    first = FailsOnSecondBatch()
    with pytest.raises(JudgeError, match="no automatic retry"):
        _prepare(topic, first, tmp_path, max_input_tokens=budget)
    assert len(first.calls) == 2

    replay = RecordingBackend()
    with pytest.raises(JudgeError, match="no automatic resend"):
        _prepare(topic, replay, tmp_path, max_input_tokens=budget)
    assert replay.calls == []


@pytest.mark.parametrize(
    ("finish", "category"),
    [
        ("length", "output token limit"),
        ("content_filter", "did not stop normally"),
        (None, "did not stop normally"),
    ],
)
def test_incomplete_receipt_has_safe_category_and_is_not_resent(tmp_path, finish, category):
    class IncompleteBackend(RecordingBackend):
        async def generate(self, request):
            result = await super().generate(request)
            result.raw["choices"][0]["finish_reason"] = finish
            return result

    topic = _topic()
    first = IncompleteBackend()
    with pytest.raises(JudgeError, match=category):
        _prepare(topic, first, tmp_path, max_input_tokens=180000)
    assert len(first.calls) == 1
    receipts = list(tmp_path.glob("*.json"))
    assert len(receipts) == 1
    original = receipts[0].read_bytes()

    replay = RecordingBackend()
    with pytest.raises(JudgeError, match=category):
        _prepare(topic, replay, tmp_path, max_input_tokens=180000)
    assert replay.calls == []
    assert receipts[0].read_bytes() == original


def test_receipt_fingerprint_mismatch_remains_fail_closed(tmp_path):
    topic = _topic()
    _prepare(topic, RecordingBackend(), tmp_path, max_input_tokens=180000)
    receipt_path = next(tmp_path.glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["fingerprint"] = "wrong-synthetic-fingerprint"
    receipt_path.write_text(json.dumps(receipt))
    replay = RecordingBackend()
    with pytest.raises(JudgeError, match="fingerprint mismatch"):
        _prepare(topic, replay, tmp_path, max_input_tokens=180000)
    assert replay.calls == []
