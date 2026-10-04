import asyncio
import pytest
from tests.test_generic_judge import report


class Checker:
    def __init__(self):
        self.payloads = []

    async def __call__(self, payload):
        self.payloads.append(payload)
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="supported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }


def test_groups_across_answers_preserves_provenance_and_scores():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "Blue.", "citations": {"d1": 100}}],
        {"d1": {"id": "d1", "text": "Blue."}},
    )
    b = a.model_copy(deep=True)
    b.metadata.run_id = "second"
    checker = Checker()
    result = asyncio.run(evaluate([a, b], checker))
    assert len(checker.payloads) == 1
    assert len(checker.payloads[0]["claims"]) == 2
    assert {x["run_id"] for x in result["supported_claims"]} == {"synthetic", "second"}
    assert [s["score"] for s in result["answers"]] == [1.0, 1.0]
    assert all(x["document_id"] == "d1" for x in result["pairs"])


def test_missing_sources_and_uncited_answers_are_not_perfect():
    from judges.generic.document_pipeline import evaluate

    a = report(documents={})
    b = report([], {})
    b.metadata.run_id = "empty"
    got = asyncio.run(evaluate([a, b], Checker()))
    assert all(s["score"] is None for s in got["answers"])
    assert not got["supported_claims"]
    assert any(p["status"] == "missing_document" for p in got["pairs"])


def test_same_id_different_text_not_merged():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "Blue.", "citations": {"d1": 100}}],
        {"d1": {"id": "d1", "text": "Blue."}},
    )
    b = a.model_copy(deep=True)
    b.metadata.run_id = "second"
    b.documents["d1"].text = "Red."
    checker = Checker()
    asyncio.run(evaluate([a, b], checker))
    assert len(checker.payloads) == 2


def test_real_token_chunk_budget_and_lossless_coverage():
    from judges.generic.document_pipeline import split_document, token_count

    text = ("Unicode é 漢字.\n\n" * 100) + "x" * 2000
    chunks = split_document(text, 100, 10)
    assert len(chunks) > 1
    end = 0
    for c in chunks:
        assert c["start"] <= end and c["end"] > end
        assert c["text"] == text[c["start"] : c["end"]]
        assert token_count(c["text"]) <= 100
        end = c["end"]
    assert end == len(text)


def test_bad_ids_fail_closed_and_do_not_export_claims():
    from judges.generic.document_pipeline import evaluate

    async def bad(payload):
        return {"claims": []}

    got = asyncio.run(evaluate([report()], bad))
    assert got["answers"][0]["score"] is None
    assert not got["supported_claims"]


def test_context_overflow_never_calls_checker():
    from judges.generic.document_pipeline import evaluate

    checker = Checker()
    got = asyncio.run(
        evaluate(
            [report()], checker, context_tokens=100, output_tokens=50, safety_tokens=40
        )
    )
    assert not checker.payloads and got["answers"][0]["score"] is None


def test_forced_chunks_use_best_and_keep_chunk_provenance():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "Blue.", "citations": {"d": 100}}],
        {"d": {"id": "d", "text": "Blue.\n\n" + "Noise. " * 200}},
    )

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="supported"
                    if "Blue." in payload["document"]
                    else "unsupported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([a], checker, document_tokens=50, overlap_tokens=5))
    assert got["answers"][0]["score"] == 1
    assert len(got["pairs"][0]["chunks"]) > 1
    assert got["supported_claims"][0]["winning_chunk_ids"]


def test_conflict_excludes_reference_and_partial_gets_no_half_credit():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "Blue.", "citations": {"d": 100}}],
        {
            "d": {
                "id": "d",
                "text": "Blue.\n\n" + "Noise. " * 100 + "\n\nCorrection: red.",
            }
        },
    )

    async def conflict(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="supported"
                    if "Blue." in payload["document"]
                    else "unsupported",
                    uncertain=False,
                    contradiction="Correction" in payload["document"],
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([a], conflict, document_tokens=50, overlap_tokens=5))
    assert got["answers"][0]["score"] == 0 and not got["supported_claims"]
    assert got["pairs"][0]["aggregation"]["conflict"]

    async def partial(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="partially_supported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([a], partial))
    assert got["answers"][0]["score"] == 0 and not got["supported_claims"]


@pytest.mark.parametrize(
    "eligibility,text",
    [
        ("incomplete", "A survey found that 73% of peo"),
        ("non_claim", "## Introduction"),
    ],
)
def test_ineligible_text_is_retained_but_not_scored_or_exported(eligibility, text):
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": text, "citations": {"d": 100}}],
        {"d": {"id": "d", "text": "A survey found 73% preferred blue."}},
    )

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility=eligibility,
                    label="supported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([a], checker))
    assert got["answers"][0]["complete"]
    assert got["answers"][0]["excluded_pairs"] == 1
    assert got["answers"][0]["eligible_pairs"] == 0
    assert got["answers"][0]["score"] is None
    assert got["pairs"][0]["eligibility"] == eligibility
    assert got["pairs"][0]["aggregation"]["label"] is None
    assert not got["supported_claims"]


def test_exclusion_does_not_dilute_valid_claim_score():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [
            {"text": t, "citations": {"d": 100}}
            for t in ["## Introduction", "The box is blue."]
        ],
        {"d": {"id": "d", "text": "The box is blue."}},
    )

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="non_claim" if c["text"].startswith("#") else "claim",
                    label="supported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([a], checker))
    assert got["answers"][0]["score"] == 1
    assert got["answers"][0]["expected_pairs"] == 2
    assert got["answers"][0]["eligible_pairs"] == 1
    assert len(got["supported_claims"]) == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_eligibility_disagreement_retains_claim_and_checks_support(chunked):
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "A fragment", "citations": {"d": 100, "e": 90}}],
        {d: {"id": d, "text": "Noise. " * 60} for d in ["d", "e"]},
    )
    calls = 0

    async def checker(payload):
        nonlocal calls
        calls += 1
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim" if calls == 1 else "incomplete",
                    label="supported" if calls == 1 else "unsupported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(
        evaluate(
            [a],
            checker,
            **({"document_tokens": 50, "overlap_tokens": 5} if chunked else {}),
        )
    )
    assert got["answers"][0]["complete"]
    assert got["answers"][0]["score"] == 0.5
    assert len(got["supported_claims"]) == 1
    assert all(p["eligibility"] == "claim" for p in got["pairs"])
    assert all(p["eligibility_disagreement"] for p in got["pairs"])


def test_uncertain_exclusion_cannot_silently_shrink_denominator():
    from judges.generic.document_pipeline import evaluate

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="non_claim",
                    label="supported",
                    uncertain=True,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([report()], checker))
    assert got["answers"][0]["complete"]
    assert got["answers"][0]["excluded_pairs"] == 0
    assert got["answers"][0]["eligible_pairs"] == 2
    assert got["answers"][0]["score"] == 0
    assert not got["supported_claims"]


def test_class_percentages_separate_acceptance_from_support_labels():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [
            {"text": t, "citations": {"d": 100}}
            for t in ["One.", "Two.", "Three.", "Heading"]
        ],
        {"d": {"id": "d", "text": "Source."}},
    )
    labels = ["supported", "partially_supported", "unsupported", "unsupported"]

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    label=labels[i],
                    uncertain=False,
                    eligibility="non_claim" if i == 3 else "claim",
                    contradiction=False,
                )
                for i, c in enumerate(payload["claims"])
            ]
        }

    got = asyncio.run(evaluate([a], checker))["answers"][0]
    assert got["label_counts"] == {
        "supported": 1,
        "partially_supported": 1,
        "unsupported": 1,
    }
    assert got["supported_or_partial_pct"] == pytest.approx(200 / 3)
    assert got["label_percentages"]["supported"] == pytest.approx(100 / 3)
    assert got["excluded_pairs"] == 1


def test_claim_with_null_support_fails_closed():
    from judges.generic.document_pipeline import evaluate

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label=None,
                    uncertain=False,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    got = asyncio.run(evaluate([report()], checker))
    assert not got["answers"][0]["complete"]
    assert not got["supported_claims"]


def test_repeated_claims_have_explicit_original_positions():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [
            {"text": text, "citations": {"d": 100} if i % 2 else {}}
            for i, text in enumerate(
                ["The sky.", "It is blue.", "The car.", "It is blue."]
            )
        ],
        {"d": {"id": "d", "text": "The sky and car are blue."}},
    )
    checker = Checker()
    asyncio.run(evaluate([a], checker))
    claims = checker.payloads[0]["claims"]
    assert [c["claim_index"] for c in claims] == [1, 3]
    assert claims[0]["answer_context"][0] == {"index": 0, "text": "The sky."}
    assert claims[1]["answer_context"][2] == {"index": 2, "text": "The car."}


def test_failed_pair_is_not_retried_on_later_chunks():
    from judges.generic.document_pipeline import evaluate

    a = report(
        [{"text": "Blue.", "citations": {"d": 100}}],
        {"d": {"id": "d", "text": "Noise. " * 200}},
    )
    calls = 0

    async def checker(payload):
        nonlocal calls
        calls += 1
        return {"claims": []}

    got = asyncio.run(evaluate([a], checker, document_tokens=50, overlap_tokens=5))
    assert calls == 1
    assert got["pairs"][0]["failure_reason"] == "coverage"
    assert got["answers"][0]["score"] is None


def test_unexpected_checker_failure_stops_without_leaking_details():
    from judges.generic.document_pipeline import evaluate

    calls = 0

    async def checker(payload):
        nonlocal calls
        calls += 1
        raise TypeError("PRIVATE_SENTINEL")

    got = asyncio.run(evaluate([report()], checker))
    assert calls == 1
    assert got["run_failure"] == "internal"
    assert all(p["failure_reason"] == "internal" for p in got["pairs"])
    assert "PRIVATE_SENTINEL" not in str(got)


def test_supported_but_uncertain_percentage_is_not_accepted_score():
    from judges.generic.document_pipeline import evaluate

    async def checker(payload):
        return {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="supported",
                    uncertain=True,
                    contradiction=False,
                )
                for c in payload["claims"]
            ]
        }

    answer = asyncio.run(evaluate([report()], checker))["answers"][0]
    assert answer["label_percentages"]["supported"] == 100
    assert answer["score"] == 0


def test_pair_ids_do_not_change_when_another_answer_is_added():
    from judges.generic.document_pipeline import evaluate

    a = report()
    b = a.model_copy(deep=True)
    b.metadata.run_id = "aaa-earlier"
    alone = asyncio.run(evaluate([a], Checker()))
    together = asyncio.run(evaluate([b, a], Checker()))
    assert [p["id"] for p in alone["pairs"]] == [
        p["id"] for p in together["pairs"] if p["run_id"] == a.metadata.run_id
    ]
