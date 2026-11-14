"""Shared cold-start judge: blind pairwise, grounded evidence, or fixed fusion."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from minima_llm import MinimaLlmRequest, MinimaLlmResponse

from .completion_session import CompletionSession
from .document_pipeline import evaluate, messages, Verdicts, token_count
from .evidence_bridge import build_topic
from .models import JudgeError
from .pairwise import judge_topic, preflight_topic
from .private_io import private_directory, write_private_text
from .reference_answer import generate_answer, make_request
from .coverage_join import checklist_scores
from .shared_judge import sentences, input_digest
from .shared_pipeline import aggregate
from .shared_semantics import prepare_topic


EVIDENCE_COMPONENTS = {
    "ragtime": (
        "RAGTIME_GROUNDED_COVERAGE_PARTIAL_PROXY",
        "RAGTIME_SENTENCE_SUPPORT_PROXY",
    ),
    "rag": (
        "RAG_REQUEST_COVERAGE_PROXY",
        "RAG_CITATION_PRECISION_UNWEIGHTED_PROXY",
        "RAG_CITATION_RECALL_UNWEIGHTED_PROXY",
    ),
}


def fuse(values, pairwise, track, weight):
    """Fixed track-specific mean; partial grounding is not full grounding."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Pairwise weight must be in [0,1].")
    keys = EVIDENCE_COMPONENTS[track]
    numbers = [values[k] for k in keys]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [*numbers, pairwise]):
        raise ValueError("Fusion requires finite unit-range scores.")
    evidence = sum(numbers) / len(numbers)
    return evidence, weight * pairwise + (1 - weight) * evidence


class UnifiedJudge:
    def __init__(self, backend_factory=None):
        self.backend_factory = backend_factory

    def judge(
        self,
        rag_responses,
        rag_topics,
        llm_config,
        *,
        track="ragtime",
        method="combined",
        filebase="unified",
        outdir=Path("."),
        pairwise_weight=0.5,
        max_pairs=100,
        max_input_tokens=45000,
        document_tokens=24000,
        context_tokens=64000,
        budget_ledger="output/budget20/budget.sqlite3",
        run_budget_usd=0.5,
        coverage_mode="separate",
        resolve_partial_grounding=False,
        repair_reference_length=True,
        **kwargs,
    ):
        if track not in ("rag", "ragtime") or method not in (
            "pairwise",
            "evidence",
            "combined",
        ):
            raise JudgeError("Unknown track or method.")
        if not math.isfinite(pairwise_weight) or not 0 <= pairwise_weight <= 1:
            raise JudgeError("Invalid pairwise weight.")
        if coverage_mode != "separate":
            raise JudgeError(
                "The single-checklist workflow requires separate coverage."
            )
        basename = Path(filebase).name
        if not basename or basename in (".", ".."):
            raise JudgeError("Invalid output basename.")
        reports = sorted(
            rag_responses, key=lambda r: (r.metadata.topic_id, r.metadata.run_id)
        )
        implementation_sha256 = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest()
        questions = {q.request_id: q for q in rag_topics}
        keys = [(r.metadata.topic_id, r.metadata.run_id) for r in reports]
        if (
            not questions
            or len(questions) != len(rag_topics)
            or len(keys) != len(set(keys))
            or {k[0] for k in keys} != set(questions)
        ):
            raise JudgeError("Incomplete or duplicate topic/run input.")
        groups = {
            tid: [r for r in reports if r.metadata.topic_id == tid] for tid in questions
        }
        if method != "evidence" and any(
            len(rs) < 2 or len(rs) * (len(rs) - 1) // 2 > max_pairs
            for rs in groups.values()
        ):
            raise JudgeError("Pairwise cohort outside configured pair limit.")
        outdir = Path(outdir)
        private_directory(outdir)

        async def run():
            options = {"factory": self.backend_factory} if self.backend_factory else {}
            session = CompletionSession(
                llm_config, budget_ledger, run_budget_usd, **options
            )
            rows, artifacts = [], []
            try:
                for tid, rs in groups.items():
                    q = questions[tid]
                    aliases = {
                        r.metadata.run_id: f"A{i + 1:03d}" for i, r in enumerate(rs)
                    }
                    packet = dict(
                        question=q.model_dump(mode="json"),
                        answers=[
                            dict(id=aliases[r.metadata.run_id], sentences=sentences(r))
                            for r in rs
                        ],
                    )
                    if method != "evidence":
                        preflight_topic(
                            packet,
                            mode="blind",
                            max_pairs=max_pairs,
                            max_input_tokens=max_input_tokens,
                        )
                    topic = None
                    if method != "pairwise":

                        async def checker(payload):
                            req = MinimaLlmRequest(
                                "document-support",
                                messages(payload),
                                max_tokens=8192,
                                extra=dict(
                                    reasoning=dict(effort="high"),
                                    response_format=dict(
                                        type="json_schema",
                                        json_schema=dict(
                                            name="ClaimSupport",
                                            strict=True,
                                            schema=Verdicts.model_json_schema(),
                                        ),
                                    ),
                                ),
                            )
                            result = await session.generate(req)
                            if not isinstance(result, MinimaLlmResponse) or [
                                c.get("finish_reason")
                                for c in (result.raw or {}).get("choices", [])
                            ] != ["stop"]:
                                raise JudgeError("Incomplete citation response.")
                            return json.loads(result.text)

                        support = await evaluate(
                            rs,
                            checker,
                            document_tokens=document_tokens,
                            context_tokens=context_tokens,
                        )
                        topic = build_topic(rs, q, support)
                        if not topic["pool"]["evidence"]:
                            raise JudgeError(
                                "No established evidence for a grounded reference."
                            )
                        request = make_request(topic["pool"], joint_checklist=True)
                        if token_count(json.dumps(request.messages)) > max_input_tokens:
                            raise JudgeError(
                                "Reference input exceeds configured limit; no truncation."
                            )
                        reference, _ = await generate_answer(
                            session,
                            topic["pool"],
                            repair_length=repair_reference_length,
                            max_input_tokens=max_input_tokens,
                            joint_checklist=True,
                        )
                        topic["reference"] = reference.model_dump()
                        # One generated checklist, reused unchanged for every answer.
                        topic["checklist"] = dict(
                            items=[i.model_dump() for i in reference.checklist],
                            gaps=reference.gaps,
                        )
                        topic = await prepare_topic(
                            topic,
                            session,
                            None,  # CompletionSession alone owns network checkpoints.
                            max_input_tokens=max_input_tokens,
                            strict_minimum=True,
                            coverage_mode=coverage_mode,
                            resolve_partial_grounding=resolve_partial_grounding,
                        )
                        topic["citation_audit"] = support
                    preferences = None
                    if method != "evidence":
                        preferences = await judge_topic(
                            packet,
                            session,
                            None,
                            mode="blind",
                            max_pairs=max_pairs,
                            max_input_tokens=max_input_tokens,
                        )
                    for index, r in enumerate(rs):
                        values = {}
                        if topic is not None:
                            record = topic["answers"][index]
                            if record["run_id"] != r.metadata.run_id:
                                raise JudgeError("Answer/run alignment mismatch.")
                            scoring = aggregate(
                                record["evidence"],
                                [i["id"] for i in topic["checklist"]["items"]],
                                track,
                            )
                            checklist_score = checklist_scores(
                                topic["checklist"]["items"],
                                record["checklist_coverage"],
                                record["evidence"]["links"],
                            )
                            grounding_key = (
                                "RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE"
                                if track == "ragtime"
                                else "RAG_NARRATIVE_NUGGET_RECALL_ESTIMATE"
                            )
                            scoring["values"][grounding_key] = checklist_score[
                                "grounded_recall"
                            ]
                            scoring["audit"]["checklist_scores"] = checklist_score
                            resolved_score = checklist_scores(
                                topic["checklist"]["items"],
                                record["checklist_coverage"],
                                record.get(
                                    "resolved_links", record["evidence"]["links"]
                                ),
                            )
                            scoring["values"][
                                f"{track.upper()}_GROUNDED_COVERAGE_PARTIAL_PROXY"
                            ] = resolved_score["grounded_partial_recall"]
                            scoring["audit"]["resolved_checklist_scores"] = (
                                resolved_score
                            )
                            topic["answers"][index]["scoring"] = scoring
                            values.update(scoring["values"])
                            # Availability flags and undefined estimates belong
                            # in the audit, not in competition ranking columns.
                            for diagnostic in (
                                "RAG_PRECISION_DEFINED",
                                "RAGTIME_PRECISION_DEFINED",
                                "RAGTIME_ELIGIBILITY_COMPLETE",
                                "RAGTIME_ELIGIBLE_SENTENCE_PRECISION_ESTIMATE",
                            ):
                                values.pop(diagnostic, None)
                            if track == "rag":
                                values["RAG_GROUNDED_CHECKLIST_RECALL_PROXY"] = (
                                    values.pop("RAG_NARRATIVE_NUGGET_RECALL_ESTIMATE")
                                )
                                scoring["values"][
                                    "RAG_GROUNDED_CHECKLIST_RECALL_PROXY"
                                ] = scoring["values"].pop(
                                    "RAG_NARRATIVE_NUGGET_RECALL_ESTIMATE"
                                )
                                values["RAG_REQUEST_COVERAGE_PROXY"] = checklist_score[
                                    "request_coverage"
                                ]
                                scoring["values"]["RAG_REQUEST_COVERAGE_PROXY"] = (
                                    checklist_score["request_coverage"]
                                )
                            values[f"{track.upper()}_EVIDENCE_COMPOSITE_PROXY"] = fuse(
                                values, 0, track, pairwise_weight
                            )[0]
                        if preferences is not None:
                            values[
                                f"{track.upper()}_PAIRWISE_BLIND_WIN_POINTS_PROXY"
                            ] = preferences["scores"][aliases[r.metadata.run_id]]
                        if method == "combined":
                            values[f"{track.upper()}_COMBINED_PROXY"] = fuse(
                                values,
                                preferences["scores"][aliases[r.metadata.run_id]],
                                track,
                                pairwise_weight,
                            )[1]
                        primary = (
                            f"{track.upper()}_"
                            + {
                                "combined": "COMBINED_PROXY",
                                "evidence": "EVIDENCE_COMPOSITE_PROXY",
                                "pairwise": "PAIRWISE_BLIND_WIN_POINTS_PROXY",
                            }[method]
                        )
                        values = {primary: values[primary], **values}
                        rows.append(
                            dict(run_id=r.metadata.run_id, topic_id=tid, values=values)
                        )
                    artifacts.append(
                        dict(
                            topic_id=tid,
                            evidence=topic,
                            pairwise=preferences,
                            inputs=[
                                dict(
                                    id=aliases[r.metadata.run_id],
                                    run_id=r.metadata.run_id,
                                    input_sha256=input_digest(r, q),
                                )
                                for r in rs
                            ],
                        )
                    )
                write_private_text(
                    outdir / (basename + ".stages.json"),
                    json.dumps(
                        dict(
                            track=track,
                            method=method,
                            pairwise_weight=pairwise_weight,
                            evidence_components=list(EVIDENCE_COMPONENTS[track]),
                            coverage_mode=coverage_mode,
                            resolve_partial_grounding=resolve_partial_grounding,
                            repair_reference_length=repair_reference_length,
                            model=session.cfg.model,
                            endpoint_host=urlparse(session.cfg.base_url).hostname,
                            implementation_sha256=implementation_sha256,
                            topics=artifacts,
                            budget=session.summary(),
                        ),
                        ensure_ascii=False,
                    ),
                )
                return rows
            finally:
                await session.aclose()

        rows = asyncio.run(run())
        fusion_description = (
            " Evidence = arithmetic mean of "
            + ", ".join(EVIDENCE_COMPONENTS[track])
            + f". Combined = {pairwise_weight:g} * pairwise + "
            + f"{1 - pairwise_weight:g} * evidence."
        )
        measures = tuple(
            MeasureSpec(
                name,
                description=(
                    "Experimental factual-core grounded coverage: full supported portion=1, "
                    "partial=0.5, unresolved=0; optional/request items excluded. Saved-evidence "
                    "alignment is controlled by resolve_partial_grounding. Not an official metric."
                    if name.endswith("_GROUNDED_COVERAGE_PARTIAL_PROXY")
                    else "Experimental "
                    + name
                    + "; [0,1], equal topic mean. See unified-judge-plan.md for proxy definitions and unknown organizer rules."
                )
                + (
                    fusion_description
                    if name.endswith(("_EVIDENCE_COMPOSITE_PROXY", "_COMBINED_PROXY"))
                    else ""
                ),
            )
            for name in rows[0]["values"]
        )
        builder = LeaderboardBuilder(LeaderboardSpec(measures=measures))
        for row in rows:
            builder.add(**row)
        return builder.build(expected_topic_ids=list(questions), on_missing="error")
