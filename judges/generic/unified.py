"""Shared cold-start judge: blind pairwise, grounded evidence, or fixed fusion."""

import asyncio
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from minima_llm import MinimaLlmRequest

from .completion_session import CompletionSession
from .concurrency import bounded_map
from .citation_completion import citation_request, complete_citation
from .citation_recovery import recover_citation_ids
from .structured_completion import MESSAGES, StructuredOutputError
from .document_pipeline import evaluate, messages, Verdicts, token_count
from .evidence_bridge import build_topic
from .models import CitationStageError, JudgeError, RunStopped
from .pairwise import judge_topic, preflight_topic, comparison_schedule
from .private_io import private_directory, write_private_text
from .reference_answer import generate_answer, reference_input
from .coverage_join import checklist_scores
from .shared_judge import sentences, input_digest
from .shared_pipeline import aggregate, PARTIAL_CITATION_DESCRIPTIONS
from .shared_semantics import prepare_topic


EVIDENCE_COMPONENTS = {
    "ragtime": (
        "RAGTIME_GROUNDED_COVERAGE_PARTIAL_PROXY",
        "RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY",
    ),
    "rag": (
        "RAG_REQUEST_COVERAGE_PROXY",
        "RAG_CITATION_PRECISION_PARTIAL_PROXY",
        "RAG_CITATION_RECALL_PARTIAL_PROXY",
    ),
}


def _deferred_question_failure(error):
    """Return a closed failure label only for completed, question-local output."""
    if isinstance(error, StructuredOutputError):
        code = getattr(error, "code", None)
        if type(code) is str and code in MESSAGES:
            return "structured_output", code, False
        return None

    if isinstance(error, CitationStageError):
        diagnostics = error.diagnostics
        if error.code == "internal" and type(diagnostics) is dict:
            code = diagnostics.get("completion_code")
            if (
                diagnostics.get("exception_type") == "StructuredOutputError"
                and type(code) is str
                and code in MESSAGES
            ):
                return "structured_output", code, False

    if isinstance(error, RunStopped) and error.code == "transport":
        diagnostics = getattr(error, "diagnostics", None)
        if (
            type(diagnostics) is dict
            and diagnostics.get("stage") == "completion"
            and diagnostics.get("exception_type") == "MinimaLlmFailure"
            and diagnostics.get("failure_type") == "MalformedResponse"
            and type(diagnostics.get("http_status")) is int
            and diagnostics.get("http_status") == 200
            and type(diagnostics.get("attempts")) is int
            and diagnostics.get("attempts") == 1
        ):
            return "transport_ambiguity", "malformed_response", True
    return None


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
        pairwise_opponents=0,
        pairwise_concurrency=1,
        evidence_concurrency=1,
        question_concurrency=1,
        defer_question_failures=False,
        max_input_tokens=45000,
        document_tokens=24000,
        context_tokens=64000,
        citation_max_output_tokens=65536,
        repair_citation_output=True,
        budget_ledger="output/budget20/budget.sqlite3",
        run_budget_usd=3.0,
        coverage_mode="separate",
        resolve_partial_grounding=False,
        partial_max_output_tokens=65536,
        partial_context_tokens=None,
        repair_reference_length=True,
        repair_reference_output=True,
        repair_reference_ids=False,
        repair_coverage_contract=False,
        repair_partial_contract=False,
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
        if type(evidence_concurrency) is not int or evidence_concurrency < 1:
            raise JudgeError("Evidence concurrency must be a positive integer.")
        if type(question_concurrency) is not int or question_concurrency < 1:
            raise JudgeError("Question concurrency must be a positive integer.")
        if type(defer_question_failures) is not bool:
            raise JudgeError("Question failure deferral must be boolean.")
        if (type(citation_max_output_tokens) is not int
                or not 8192 <= citation_max_output_tokens <= 65536
                or type(repair_citation_output) is not bool):
            raise JudgeError("Invalid citation completion budget or repair setting.")
        if type(repair_reference_output) is not bool:
            raise JudgeError("Reference output repair must be boolean.")
        if type(repair_reference_ids) is not bool:
            raise JudgeError("Reference ID repair must be boolean.")
        if type(repair_coverage_contract) is not bool:
            raise JudgeError("Coverage contract repair must be boolean.")
        if type(repair_partial_contract) is not bool:
            raise JudgeError("Partial contract repair must be boolean.")
        if (
            type(partial_max_output_tokens) is not int
            or partial_max_output_tokens not in (16384, 32768, 65536)
        ):
            raise JudgeError("Unsupported partial-grounding output token ceiling.")
        pairwise_measure = ("PAIRWISE_BLIND_BT_PROXY" if pairwise_opponents
                            else "PAIRWISE_BLIND_WIN_POINTS_PROXY")
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
        run_ids = {r.metadata.run_id for r in reports}
        if any({r.metadata.run_id for r in rs} != run_ids for rs in groups.values()):
            raise JudgeError("Incomplete topic/run input.")
        if method != "evidence" and any(
            len(rs) < 2 or len(comparison_schedule(len(rs), pairwise_opponents)) > max_pairs
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
            try:
                progress = dict(
                    total=len(groups), completed=0, deferred=0, fatal=0
                )
                deferred_failures = []
                malformed_transport_failures = 0
                progress_path = outdir / (basename + ".topic-progress.json")
                manifest_path = outdir / (basename + ".deferred-questions.json")

                def write_progress():
                    if defer_question_failures:
                        write_private_text(
                            progress_path,
                            json.dumps(progress, sort_keys=True),
                        )

                def record_deferred(ordinal, category, code):
                    deferred_failures.append(
                        dict(ordinal=ordinal, category=category, code=code)
                    )
                    progress["deferred"] += 1
                    write_private_text(
                        manifest_path,
                        json.dumps(
                            sorted(deferred_failures, key=lambda record: record["ordinal"]),
                            sort_keys=True,
                        ),
                    )
                    write_progress()

                if defer_question_failures:
                    write_progress()

                async def process_topic_work(item):
                    tid, rs = item
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
                            opponents=pairwise_opponents,
                            schedule_seed=tid,
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
                            async def complete(request, expected_ids):
                                return await complete_citation(session, request,
                                    context_tokens=context_tokens,
                                    max_output_tokens=citation_max_output_tokens,
                                    repair_output=repair_citation_output,
                                    expected_ids=expected_ids)

                            expected = {c["id"] for c in payload["claims"]}
                            if len(expected) != len(payload['claims']):
                                raise JudgeError('Duplicate input citation identifiers.')
                            selected, _ = citation_request(session, req,
                                context_tokens=context_tokens,
                                max_output_tokens=citation_max_output_tokens)
                            # Both old 8K and newer larger receipts retain their
                            # exact paths. Unresolved markers block this probe.
                            if session.cached_response(selected) is not None:
                                result = None
                                try:
                                    result = await complete(req, expected)
                                except StructuredOutputError as error:
                                    if error.code != 'claim_ids':
                                        raise
                                ids = [] if result is None else [v.id for v in result.claims]
                                if result is not None and len(ids) == len(set(ids)) and set(ids) == expected:
                                    return result.model_dump()
                            # Retain every legacy request. Only a completed alias
                            # ID mismatch can enter one exact-key correction.
                            mapping = {f"C{i}": c["id"] for i, c in enumerate(payload["claims"], 1)}
                            alias_payload = dict(payload, claims=[
                                dict(c, id=alias) for alias, c in zip(mapping, payload["claims"])
                            ])
                            alias_request = replace(req,
                                request_id="document-support-short-ids-v1",
                                messages=messages(alias_payload))
                            try:
                                result = await complete(alias_request, set(mapping))
                                ids = [v.id for v in result.claims]
                                if len(ids) != len(set(ids)) or set(ids) != set(mapping):
                                    raise StructuredOutputError('claim_ids')
                            except StructuredOutputError as error:
                                if error.code != 'claim_ids':
                                    raise
                                result = await recover_citation_ids(session, alias_request,
                                    expected_ids=mapping, context_tokens=context_tokens,
                                    max_output_tokens=citation_max_output_tokens)
                            result.claims = [v.model_copy(update=dict(id=mapping[v.id]))
                                             for v in result.claims]
                            return result.model_dump()

                        support = await evaluate(
                            rs,
                            checker,
                            document_tokens=document_tokens,
                            context_tokens=context_tokens,
                            concurrency=evidence_concurrency,
                        )
                        topic = build_topic(rs, q, support)
                        if not topic["pool"]["evidence"]:
                            raise JudgeError(
                                "No established evidence for a grounded reference."
                            )
                        reference_packet, compact, reference_audit = reference_input(topic, max_input_tokens)
                        completion_audit = {}
                        reference, _ = await generate_answer(
                            session,
                            reference_packet,
                            repair_length=repair_reference_length,
                            max_input_tokens=max_input_tokens,
                            joint_checklist=True,
                            validated_claims=compact,
                            repair_output=repair_reference_output,
                            repair_reference_ids=repair_reference_ids,
                            audit=completion_audit,
                        )
                        topic["reference_input"] = reference_audit
                        topic["reference_audit"] = completion_audit
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
                            partial_max_output_tokens=partial_max_output_tokens,
                            partial_context_tokens=partial_context_tokens,
                            repair_coverage_contract=repair_coverage_contract,
                            repair_partial_contract=repair_partial_contract,
                            concurrency=evidence_concurrency,
                        )
                        topic["citation_audit"] = support
                    preferences = None
                    if method != "evidence":
                        preferences = await judge_topic(
                            packet,
                            session,
                            None,
                            concurrency=pairwise_concurrency,
                            mode="blind",
                            max_pairs=max_pairs,
                            opponents=pairwise_opponents,
                            schedule_seed=tid,
                            max_input_tokens=max_input_tokens,
                        )
                    topic_rows = []
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
                                f"{track.upper()}_{pairwise_measure}"
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
                                "pairwise": pairwise_measure,
                            }[method]
                        )
                        values = {primary: values[primary], **values}
                        topic_rows.append(
                            dict(run_id=r.metadata.run_id, topic_id=tid, values=values)
                        )
                    return topic_rows, dict(
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

                async def process_topic(indexed_item):
                    nonlocal malformed_transport_failures
                    ordinal, item = indexed_item
                    try:
                        result = await process_topic_work(item)
                    except (StructuredOutputError, CitationStageError, RunStopped) as error:
                        classification = _deferred_question_failure(error)
                        if not defer_question_failures or classification is None:
                            raise
                        category, code, transport_ambiguity = classification
                        if transport_ambiguity:
                            malformed_transport_failures += 1
                        record_deferred(ordinal, category, code)
                        if transport_ambiguity and malformed_transport_failures > 1:
                            raise
                        return None

                    if defer_question_failures:
                        topic_id = item[0]
                        result_dir = outdir / (basename + ".topic-results")
                        private_directory(result_dir)
                        result_path = result_dir / (
                            hashlib.sha256(topic_id.encode()).hexdigest() + ".json"
                        )
                        topic_rows, artifact = result
                        write_private_text(
                            result_path,
                            json.dumps(
                                dict(topic_id=topic_id, rows=topic_rows, artifact=artifact),
                                ensure_ascii=False,
                            ),
                        )
                        progress["completed"] += 1
                        write_progress()
                    return result

                try:
                    topic_results = await bounded_map(
                        enumerate(groups.items(), start=1),
                        process_topic,
                        concurrency=question_concurrency,
                    )
                except BaseException:
                    if defer_question_failures:
                        progress["fatal"] += 1
                        try:
                            write_progress()
                        except Exception:
                            pass
                    raise

                if deferred_failures:
                    raise JudgeError(
                        "Question failures were deferred; no complete leaderboard was emitted."
                    )
                rows = [row for topic_rows, _ in topic_results for row in topic_rows]
                artifacts = [artifact for _, artifact in topic_results]
                write_private_text(
                    outdir / (basename + ".stages.json"),
                    json.dumps(
                        dict(
                            track=track,
                            method=method,
                            pairwise_weight=pairwise_weight,
                            evidence_components=list(EVIDENCE_COMPONENTS[track]),
                            coverage_mode=coverage_mode,
                            evidence_concurrency=evidence_concurrency,
                            question_concurrency=question_concurrency,
                            resolve_partial_grounding=resolve_partial_grounding,
                            partial_max_output_tokens=partial_max_output_tokens,
                            repair_reference_length=repair_reference_length,
                            repair_reference_output=repair_reference_output,
                            repair_reference_ids=repair_reference_ids,
                            repair_coverage_contract=repair_coverage_contract,
                            repair_partial_contract=repair_partial_contract,
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
                    else PARTIAL_CITATION_DESCRIPTIONS.get(
                        name,
                        "Experimental "
                        + name
                        + "; [0,1], equal topic mean. See unified-judge-plan.md for proxy definitions and unknown organizer rules.",
                    )
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
