"""Shared semantic links, separate track aggregation over reusable evidence.

No hidden assessor rubric, weights or sentence eligibility is invented here.
Every computed approximation is named accordingly. The unified workflow may use
partial credit in its composite; this is not an official half-credit rule.
"""

from collections import Counter
from typing import Literal

from pydantic import Field, ValidationError

from .models import StrictModel, JudgeError


class Sentence(StrictModel):
    text: str
    citations: list[str]
    non_claim: bool = False
    # Caller supplies organizer eligibility semantics; None means not assessed.
    eligible: bool | None = None


class Pair(StrictModel):
    id: str
    sentence_index: int = Field(ge=0)
    document_id: str
    text: str
    label: Literal["supported", "partially_supported", "unsupported"]
    uncertain: bool
    excerpts: list[str]


class ItemLink(StrictModel):
    case: str
    item: str
    mentioned: bool
    status: Literal["supported", "partially_supported", "not_established", "uncertain"]
    pair_ids: list[str]
    supported_portion: str
    missing_or_unsupported: str


class AnswerEvidence(StrictModel):
    id: str
    sentences: list[Sentence]
    pairs: list[Pair]
    links: list[ItemLink]


def validated_answer(data, item_ids, *, strict_minimum=False):
    try:
        answer = AnswerEvidence.model_validate(data)
    except (ValidationError, TypeError, ValueError):
        raise JudgeError("Invalid shared answer artifact.") from None
    if not item_ids or len(set(item_ids)) != len(item_ids):
        raise JudgeError("Checklist IDs must be nonempty and unique.")
    if len(answer.links) != len(item_ids) or {x.item for x in answer.links} != set(
        item_ids
    ):
        raise JudgeError("Missing or duplicate checklist links.")
    pairs = {p.id: p for p in answer.pairs}
    if len(pairs) != len(answer.pairs):
        raise JudgeError("Duplicate citation pair IDs.")
    expected = {(i, d) for i, s in enumerate(answer.sentences) for d in s.citations}
    actual = {(p.sentence_index, p.document_id) for p in answer.pairs}
    if expected != actual or len(actual) != len(answer.pairs):
        raise JudgeError(
            "Missing or duplicate citation judgments; no automatic rejudging."
        )
    for s in answer.sentences:
        if len(s.citations) != len(set(s.citations)):
            raise JudgeError("Duplicate sentence citations.")
    for p in answer.pairs:
        if (
            p.label == "supported"
            and not p.uncertain
            and (not p.excerpts or any(not e.strip() for e in p.excerpts))
        ):
            raise JudgeError(
                "Supported citation judgment requires retained source excerpts."
            )
        if (
            p.sentence_index >= len(answer.sentences)
            or p.text != answer.sentences[p.sentence_index].text
        ):
            raise JudgeError(
                "Citation judgment text does not match the original answer."
            )
    for link in answer.links:
        if (
            strict_minimum
            and link.status == "supported"
            and link.missing_or_unsupported.strip()
        ):
            raise JudgeError("Full item support cannot report a missing requirement.")
        if (
            link.case != answer.id
            or len(link.pair_ids) != len(set(link.pair_ids))
            or not set(link.pair_ids) <= pairs.keys()
        ):
            raise JudgeError("Invalid within-answer support pointer.")
        if link.status in ("supported", "partially_supported"):
            if (
                not link.mentioned
                or not link.pair_ids
                or not link.supported_portion.strip()
            ):
                raise JudgeError(
                    "Positive link requires expressed content and evidence."
                )
            if not any(
                pairs[k].label in ("supported", "partially_supported")
                and not pairs[k].uncertain
                and any(e.strip() for e in pairs[k].excerpts)
                for k in link.pair_ids
            ):
                raise JudgeError("Positive link has no established saved support.")
    return answer


def aggregate(data, item_ids, track):
    if track not in ("rag", "ragtime"):
        raise JudgeError("Track must be rag or ragtime.")
    a = validated_answer(data, item_ids)
    pairs = {(p.sentence_index, p.document_id): p for p in a.pairs}

    def supported(p):
        return p.label == "supported" and not p.uncertain

    top = [
        bool(s.citations) and supported(pairs[i, s.citations[0]])
        for i, s in enumerate(a.sentences)
    ]
    any_support = [
        any(supported(pairs[i, d]) for d in s.citations)
        for i, s in enumerate(a.sentences)
    ]
    count = len(a.sentences)
    counts = Counter(x.status for x in a.links)
    recall = counts["supported"] / len(item_ids)
    unknown = [
        "Estimated checklist is not the hidden assessor nugget bank.",
        "Partial support and empty-denominator conventions are not verified official rules.",
        "Original answer text is retained; length handling must match the evaluation release.",
    ]
    audit = dict(
        link_counts=dict(counts),
        checklist_items=len(item_ids),
        partial_mentions=sum(x.mentioned for x in a.links),
        sentences=count,
        citations=len(a.pairs),
        sentence_precision=None,
    )
    if track == "ragtime":
        substantive = [i for i, s in enumerate(a.sentences) if not s.non_claim]
        def partial_credit(i):
            s = a.sentences[i]
            if not s.citations:
                return 0.0
            p = pairs[i, s.citations[0]]
            if p.uncertain:
                return 0.0
            return {'supported': 1.0, 'partially_supported': 0.5, 'unsupported': 0.0}[p.label]

        audit['non_claims_excluded'] = count - len(substantive)
        audit['partial_credit_denominator'] = len(substantive)
        complete = all(s.eligible is not None for s in a.sentences)
        eligible = [i for i, s in enumerate(a.sentences) if s.eligible is True]
        if complete and eligible:
            audit["sentence_precision"] = sum(top[i] for i in eligible) / len(eligible)
        values = dict(
            RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY=(
                sum(partial_credit(i) for i in substantive) / len(substantive)
                if substantive else 0.0
            ),
            RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE=recall,
            RAGTIME_SENTENCE_SUPPORT_PROXY=sum(top) / count if count else 0.0,
            RAGTIME_ELIGIBLE_SENTENCE_PRECISION_ESTIMATE=audit["sentence_precision"]
            or 0.0,
            RAGTIME_ELIGIBILITY_COMPLETE=float(complete),
            RAGTIME_PRECISION_DEFINED=float(audit["sentence_precision"] is not None),
        )
        if not complete:
            unknown.append(
                "Missing sentence eligibility: official exclusions cannot be reconstructed from pair labels alone."
            )
        unknown.append(
            "Citation order must be supplied in assessment-priority order; support is strict full support."
        )
    else:
        values = dict(
            RAG_NARRATIVE_NUGGET_RECALL_ESTIMATE=recall,
            RAG_CITATION_PRECISION_UNWEIGHTED_PROXY=(
                sum(supported(p) for p in a.pairs) / len(a.pairs) if a.pairs else 0.0
            ),
            RAG_CITATION_RECALL_UNWEIGHTED_PROXY=(
                sum(any_support) / count if count else 0.0
            ),
            RAG_PRECISION_DEFINED=float(bool(a.pairs)),
        )
        unknown.extend(
            [
                "Official citation weights unavailable; reported proxies use unit weights.",
                "Joint partial-citation support is not inferred by combining unrelated partial labels.",
                "Narrative rubric scoring scale and pairwise battle aggregation are not implemented by this pointwise proxy.",
            ]
        )
    return dict(values=values, audit=audit, unknown_rules=unknown)
