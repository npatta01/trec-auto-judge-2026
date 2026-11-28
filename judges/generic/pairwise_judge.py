"""Offline AutoJudge adapter for completed pairwise experiments."""

import json
import math
from pathlib import Path

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec

from .models import JudgeError
from .pairwise import GUIDANCE, aggregate_decisions, prompt_input_digest
from .shared_judge import input_digest, sentences


class PairwiseJudge:
    def judge(
        self,
        rag_responses,
        rag_topics,
        llm_config,
        *,
        pairwise_bundle=None,
        track="rag",
        filebase="pairwise",
        outdir=Path("."),
        **kwargs,
    ):
        if track not in {"rag", "ragtime"} or not pairwise_bundle:
            raise JudgeError("Pairwise replay requires a track and completed bundle.")
        data = json.loads(Path(pairwise_bundle).read_text())
        queries = {q.request_id: q for q in rag_topics}
        topics = {t["topic_id"]: t for t in data["topics"]}
        if (
            not queries
            or len(queries) != len(rag_topics)
            or len(topics) != len(data["topics"])
            or set(topics) != set(queries)
        ):
            raise JudgeError("Pairwise topic coverage differs from input.")
        modes = {t["result"]["mode"] for t in topics.values()}
        if len(modes) != 1 or not modes <= set(GUIDANCE):
            raise JudgeError("Mixed or invalid pairwise modes.")
        name = f"{track.upper()}_PAIRWISE_{next(iter(modes)).upper()}_WIN_POINTS_PROXY"
        builder = LeaderboardBuilder(
            LeaderboardSpec(
                measures=(
                    MeasureSpec(
                        name,
                        description="Mean win=1/tie=.5/loss=0 over both orders and every opponent in the supplied cohort; topic mean. Experimental preference, not official battle scoring.",
                    ),
                )
            )
        )
        expected = {}
        for tid, topic in topics.items():
            records = topic["inputs"]
            ids = [r["id"] for r in records]
            scores = topic["result"]["scores"]
            recomputed = aggregate_decisions(
                ids, topic["result"].get("pairs", []), topic["result"]["mode"]
            )
            if recomputed != topic["result"]:
                raise JudgeError(
                    "Pairwise scores or diagnostics differ from saved decisions."
                )
            if len(ids) < 2 or len(set(ids)) != len(ids) or set(ids) != set(scores):
                raise JudgeError("Incomplete pairwise answer scores.")
            for record in records:
                key = (tid, record["run_id"])
                if key in expected:
                    raise JudgeError("Duplicate pairwise input.")
                expected[key] = (
                    record["input_sha256"],
                    record["prompt_input_sha256"],
                    scores[record["id"]],
                )
        seen = set()
        for report in rag_responses:
            key = (report.metadata.topic_id, report.metadata.run_id)
            if key not in expected or key in seen:
                raise JudgeError("Unknown or duplicate pairwise report.")
            digest, prompt_digest, score = expected[key]
            if digest != input_digest(report, queries[key[0]]):
                raise JudgeError("Stale pairwise input.")
            if prompt_digest != prompt_input_digest(
                queries[key[0]].model_dump(mode="json"), sentences(report)
            ):
                raise JudgeError("Prompted answer differs from original input.")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise JudgeError("Invalid pairwise score.")
            seen.add(key)
            builder.add(run_id=key[1], topic_id=key[0], values={name: score})
        if seen != set(expected):
            raise JudgeError("Pairwise cohort differs from supplied reports.")
        return builder.build(expected_topic_ids=list(queries), on_missing="error")
