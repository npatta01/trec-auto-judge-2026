"""Lossless bridge from document judgments to per-answer and pooled evidence."""

import hashlib

from .models import JudgeError
from .shared_judge import input_digest, sentences


def build_topic(reports, question, support):
    if support.get("run_failure"):
        raise JudgeError("Citation stage incomplete; cannot build evidence.")
    by_answer = {}
    for p in support["pairs"]:
        if p["topic_id"] == question.request_id:
            by_answer.setdefault(p["run_id"], []).append(p)
    records, pool = [], {}
    for index, report in enumerate(sorted(reports, key=lambda r: r.metadata.run_id)):
        alias = f"A{index + 1:03d}"
        units, pairs = sentences(report), []
        for p in by_answer.get(report.metadata.run_id, []):
            agg = p.get("aggregation", {})
            if p["status"] != "complete" or not agg.get("complete"):
                raise JudgeError("Incomplete citation judgment.")
            document = (report.documents or {}).get(p["document_id"])
            if document is None:
                raise JudgeError("Citation source missing.")
            text = document.get_text()
            if hashlib.sha256(text.encode()).hexdigest() != p["document_sha256"]:
                raise JudgeError("Citation source changed.")
            if units[p["claim_index"]]["text"] != p["claim_text"]:
                raise JudgeError("Citation claim changed.")
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
