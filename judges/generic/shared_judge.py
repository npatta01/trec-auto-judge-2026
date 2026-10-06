"""AutoJudge adapter for replaying reusable checklist and evidence-link bundles.

Semantic stages are prepared separately; this adapter never silently rejudges
citations or spends money when a saved artifact is missing.
"""

import hashlib
import json
from pathlib import Path

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec

from .models import JudgeError
from .private_io import write_private_text
from .shared_pipeline import aggregate


def sentences(report):
    result = []
    for i, s in enumerate(report.get_sentences_with_citations()):
        citations = list(s.citations)
        raw = report.responses[i].citations
        if isinstance(raw, dict):
            # Stable sort retains submitted order for ties; no invented ID tie-break.
            citations = sorted(citations, key=lambda d: -raw[d])
        result.append(dict(text=s.text, citations=citations))
    return result


def input_digest(report, question):
    value = dict(
        question=question.model_dump(mode="json"),
        sentences=sentences(report),
        documents={k: v.get_text() for k, v in (report.documents or {}).items()},
    )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class SharedJudge:
    def __init__(self, backend_factory=None):
        # Accepted for test symmetry; replay intentionally has no backend.
        self.backend_factory = backend_factory

    def judge(
        self,
        rag_responses,
        rag_topics,
        llm_config,
        *,
        evidence_bundle=None,
        track="ragtime",
        filebase="shared",
        outdir=Path("."),
        **kwargs,
    ):
        if not evidence_bundle:
            raise JudgeError(
                "A reusable evidence bundle is required; no automatic citation rejudging."
            )
        try:
            data = json.loads(Path(evidence_bundle).read_text())
            topics = {x["topic_id"]: x for x in data["topics"]}
            if len(topics) != len(data["topics"]):
                raise JudgeError("Duplicate evidence topics.")
        except (OSError, ValueError, KeyError, TypeError):
            raise JudgeError("Cannot load shared evidence bundle.") from None
        queries = {q.request_id: q for q in rag_topics}
        if not queries or len(queries) != len(rag_topics):
            raise JudgeError("Invalid question coverage.")
        rows, seen = [], set()
        for report in rag_responses:
            tid, rid = report.metadata.topic_id, report.metadata.run_id
            if tid not in queries or tid not in topics or (rid, tid) in seen:
                raise JudgeError("Unknown or duplicate report input.")
            seen.add((rid, tid))
            topic = topics[tid]
            records = [x for x in topic["answers"] if x["run_id"] == rid]
            if len(records) != 1 or records[0]["input_sha256"] != input_digest(
                report, queries[tid]
            ):
                raise JudgeError("Missing or stale evidence bundle record.")
            evidence = records[0]["evidence"]
            actual = sentences(report)
            saved = [
                {k: s[k] for k in ("text", "citations")} for s in evidence["sentences"]
            ]
            if actual != saved:
                raise JudgeError("Evidence answer text or citations differ from input.")
            for pair in evidence["pairs"]:
                doc = (report.documents or {}).get(pair["document_id"])
                if doc is None or any(
                    not excerpt.strip() or excerpt not in doc.get_text()
                    for excerpt in pair["excerpts"]
                ):
                    raise JudgeError(
                        "Saved citation excerpt is not in the current source."
                    )
            result = aggregate(
                evidence, [i["id"] for i in topic["checklist"]["items"]], track
            )
            rows.append(dict(run_id=rid, topic_id=tid, **result))
        if not rows:
            raise JudgeError("No reports supplied.")
        descriptions = {
            "RAGTIME_GROUNDED_NUGGET_RECALL_ESTIMATE": "Fully evidence-linked estimated items / checklist items; partial=0, equal-item/topic mean; not hidden assessor recall.",
            "RAGTIME_SENTENCE_SUPPORT_PROXY": "Strict full support of top-priority citation / all answer sentences; uncited=0; excludes no sentences. Not official eligible-sentence precision.",
            "RAGTIME_ELIGIBLE_SENTENCE_PRECISION_ESTIMATE": "Strict top-citation support / supplied eligible sentences; only usable when eligibility complete and precision defined; undefined exported as zero.",
            "RAGTIME_ELIGIBILITY_COMPLETE": "1 if every sentence has supplied eligibility, otherwise 0.",
            "RAGTIME_PRECISION_DEFINED": "1 if eligible precision has a complete nonempty denominator, otherwise 0.",
            "RAG_NARRATIVE_NUGGET_RECALL_ESTIMATE": "Fully linked estimated narrative items / checklist size; not official rubric scale.",
            "RAG_CITATION_PRECISION_UNWEIGHTED_PROXY": "Fully supported citation pairs / supplied pairs, unit weights; no pairs exported as zero.",
            "RAG_CITATION_RECALL_UNWEIGHTED_PROXY": "Answer objects with at least one fully supporting citation / all objects; uncited=0; no inferred joint partial support.",
            "RAG_PRECISION_DEFINED": "1 if at least one citation pair exists; otherwise 0.",
        }
        descriptions['RAGTIME_SENTENCE_SUPPORT_PARTIAL_PROXY'] = (
            'Top-citation support: full=1, partial=0.5, otherwise=0; confirmed '
            'non-claims excluded. Experimental proxy, not official eligibility.'
        )
        spec = LeaderboardSpec(
            measures=tuple(
                MeasureSpec(k, description=descriptions[k]) for k in rows[0]["values"]
            )
        )
        builder = LeaderboardBuilder(spec)
        for row in rows:
            builder.add(
                run_id=row["run_id"], topic_id=row["topic_id"], values=row["values"]
            )
        try:
            board = builder.build(expected_topic_ids=list(queries), on_missing="error")
        except ValueError:
            raise JudgeError("Incomplete run/topic coverage.") from None
        path = Path(filebase)
        if ".." in path.parts or (
            path.parent != Path(".") and path.parent.resolve() != Path(outdir).resolve()
        ):
            raise JudgeError("Unsafe artifact basename.")
        write_private_text(
            Path(outdir) / (path.name + ".audit.json"), json.dumps(rows, indent=2)
        )
        return board
