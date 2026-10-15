"""Lossless bridge from document judgments to per-answer and pooled evidence."""

import hashlib

from .models import CitationStageError, JudgeError
from .shared_judge import input_digest, sentences


def build_topic(reports, question, support):
    if support.get("run_failure"):
        raise CitationStageError(support["run_failure"], support.get("run_failure_diagnostics"))
    by_answer = {}
    for p in support["pairs"]:
        if p["topic_id"] == question.request_id:
            by_answer.setdefault(p["run_id"], []).append(p)
    records, pool = [], {}
    for index, report in enumerate(sorted(reports, key=lambda r: r.metadata.run_id)):
        alias = f"A{index + 1:03d}"
        units, pairs = sentences(report), []
        for p in by_answer.get(report.metadata.run_id, []):
            if units[p["claim_index"]]["text"] != p["claim_text"]:
                raise JudgeError("Citation claim changed.")
            document = (report.documents or {}).get(p["document_id"])
            if p["status"] == "missing_document":
                if document is not None and document.text and document.text.strip():
                    raise JudgeError("Missing-source status contradicts supplied text.")
                pairs.append(dict(id=p["id"], sentence_index=p["claim_index"],
                    document_id=p["document_id"], text=p["claim_text"],
                    label="unsupported", uncertain=True, excerpts=[]))
                continue
            agg = p.get("aggregation", {})
            if p["status"] != "complete" or not agg.get("complete"):
                raise JudgeError("Incomplete citation judgment.")
            if document is None:
                raise JudgeError("Citation source missing.")
            text = document.get_text()
            if hashlib.sha256(text.encode()).hexdigest() != p["document_sha256"]:
                raise JudgeError("Citation source changed.")
            excerpts = []
            for chunk in p["chunks"]:
                if chunk["chunk_id"] in agg["winning_chunk_ids"]:
                    start, end = chunk["start"], chunk["end"]
                    if not 0 <= start < end <= len(text):
                        raise JudgeError("Invalid evidence offsets.")
                    excerpts.append(text[start:end])
            excluded = p.get("eligibility") in ("incomplete", "non_claim")
            label = agg["label"] if not excluded else "unsupported"
            uncertain = agg["uncertain"] or agg.get("conflict", False) or excluded
            pair = dict(
                id=p["id"],
                sentence_index=p["claim_index"],
                document_id=p["document_id"],
                text=p["claim_text"],
                label=label,
                uncertain=uncertain,
                excerpts=excerpts,
            )
            pairs.append(pair)
            if not uncertain and label in ("supported", "partially_supported"):
                for excerpt in excerpts:
                    key = excerpt
                    if key not in pool:
                        pool[key] = dict(
                            id=f"E{len(pool) + 1:05d}",
                            claims=[],
                            excerpt=excerpt,
                            pair_ids=[],
                        )
                    pool[key]["pair_ids"].append(p["id"])
                    hint = dict(text=p["claim_text"], label=label)
                    if hint not in pool[key]["claims"]:
                        pool[key]["claims"].append(hint)
        # Citation eligibility is not the organizer's nugget-membership exclusion.
        for i, unit in enumerate(units):
            unit["eligible"] = None
            judgments = [p for p in by_answer.get(report.metadata.run_id, [])
                         if p['claim_index'] == i]
            unit['non_claim'] = bool(judgments) and all(
                p.get('eligibility') == 'non_claim' for p in judgments
            )
        records.append(
            dict(
                run_id=report.metadata.run_id,
                input_sha256=input_digest(report, question),
                evidence=dict(id=alias, sentences=units, pairs=pairs),
            )
        )
    return dict(
        topic_id=question.request_id,
        pool=dict(
            question=question.model_dump(mode="json"), evidence=list(pool.values())
        ),
        answers=records,
    )


def compact_claims(topic):
    """Exact positive claims for synthesis; all source links stay in a local audit.

    A pool entry's pair_ids and claim hints are independent unions. Reconstruct
    their association from answer pairs instead of zipping those unions.
    """
    evidence = {e['excerpt']: e['id'] for e in topic['pool']['evidence']}
    claims, rows, partials = {}, {}, []
    for record in topic['answers']:
        for pair in record['evidence']['pairs']:
            if pair['uncertain'] or pair['label'] not in ('supported', 'partially_supported'):
                continue
            if not pair['excerpts'] or any(e not in evidence for e in pair['excerpts']):
                raise JudgeError('Positive reference claim lacks a canonical source.')
            ids = list(dict.fromkeys(evidence[e] for e in pair['excerpts']))
            key = (pair['text'], pair['label'])
            if key not in claims:
                eid = ids[0] if pair['label'] == 'supported' else None
                claims[key] = dict(text=pair['text'], label=pair['label'],
                    reference_evidence_id=eid, evidence_ids=[], pair_ids=[], sources=[])
                hint = dict(text=pair['text'], label=pair['label'])
                if eid is None:
                    partials.append(hint)
                else:
                    rows.setdefault(eid, dict(id=eid, claims=[]))['claims'].append(hint)
            item = claims[key]
            item['evidence_ids'] = list(dict.fromkeys([*item['evidence_ids'], *ids]))
            item['pair_ids'].append(pair['id'])
            item['sources'].append(dict(pair_id=pair['id'], run_id=record['run_id'],
                answer_id=record['evidence']['id'], sentence_index=pair['sentence_index'],
                document_id=pair['document_id'], evidence_ids=ids))
    if not rows:
        raise JudgeError('No fully supported claims for a factual reference.')
    return dict(question=topic['pool']['question'], evidence=list(rows.values()),
                partial_claims=partials), list(claims.values())
