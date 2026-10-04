"""Framework adapter for document-first support; private provenance sidecars."""

import asyncio
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from urllib.parse import urlparse

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from minima_llm import MinimaLlmRequest, MinimaLlmResponse

from .client import make_backend
from .budget import BudgetBackend
from .document_pipeline import evaluate, messages, Verdicts
from .models import JudgeError, RunStopped
from .private_io import private_directory, write_private_text

SPEC = LeaderboardSpec(
    measures=(
        MeasureSpec(
            "DOCUMENT_SUPPORT_PROXY",
            description="Accepted original claim/citation pairs divided by eligible pairs; incomplete/non-claim text excluded, partial=0. Equal topic mean. No eligible pairs=0 in leaderboard only; sidecar score=null. Not official citation precision.",
        ),
        MeasureSpec(
            "HAS_CITATIONS",
            description="1 when answer has original citation pairs before eligibility exclusions; otherwise 0.",
        ),
        MeasureSpec(
            "HAS_ELIGIBLE_CITATIONS",
            description="1 when answer has at least one eligible claim/citation pair; otherwise 0. Distinguishes undefined support denominator after eligibility exclusions.",
        ),
    )
)


class DocumentJudge:
    def __init__(self, backend_factory=None):
        self.backend_factory = backend_factory or make_backend

    def judge(
        self,
        rag_responses,
        rag_topics,
        llm_config,
        filebase="document",
        outdir=Path("."),
        document_tokens=24000,
        overlap_tokens=256,
        context_tokens=64000,
        output_tokens=8192,
        safety_tokens=2048,
        batch_size=5,
        budget_ledger="output/budget20/budget.sqlite3",
        budget_cap_usd=16.0,
        run_budget_usd=0.50,
        **kwargs,
    ):
        path = Path(filebase)
        if not path.name or path.name in (".", "..") or ".." in path.parts:
            raise JudgeError("Unsafe artifact basename.")
        if path.parent != Path(".") and path.parent.resolve() != Path(outdir).resolve():
            raise JudgeError("Artifact path is outside the output directory.")
        filebase = path.name
        topics = [q.request_id for q in rag_topics]
        reports = list(rag_responses)
        if (
            not topics
            or len(set(topics)) != len(topics)
            or any(r.metadata.topic_id not in topics for r in reports)
        ):
            raise JudgeError("Invalid topic coverage.")

        async def run():
            backend = None
            model_name = None

            async def checker(payload):
                nonlocal backend, model_name
                if backend is None:
                    try:
                        backend = self.backend_factory(llm_config)
                        cfg = getattr(backend, "cfg", None)
                        if (
                            urlparse(str(getattr(cfg, "base_url", ""))).hostname
                            == "openrouter.ai"
                        ):
                            if not budget_ledger:
                                raise JudgeError(
                                    "OpenRouter requires a persistent budget ledger."
                                )
                            backend = BudgetBackend(
                                backend,
                                budget_ledger,
                                cap=budget_cap_usd,
                                incremental_cap=run_budget_usd,
                                completion_price=4,
                            )
                    except Exception:
                        raise RunStopped("configuration") from None
                model_name = getattr(getattr(backend, "cfg", None), "model", None)
                req = MinimaLlmRequest(
                    request_id="document-support",
                    messages=messages(payload),
                    max_tokens=output_tokens,
                    extra={
                        "reasoning": {"effort": "high"},
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ClaimSupport",
                                "strict": True,
                                "schema": Verdicts.model_json_schema(),
                            },
                        },
                    },
                )
                # Never surface provider errors containing request data.
                with (
                    open(os.devnull, "w") as sink,
                    redirect_stdout(sink),
                    redirect_stderr(sink),
                ):
                    try:
                        result = await backend.generate(req)
                    except RunStopped:
                        raise
                    except JudgeError:
                        raise RunStopped("configuration") from None
                    except Exception:
                        raise RunStopped("transport") from None
                if not isinstance(result, MinimaLlmResponse):
                    raise RunStopped("transport")
                return json.loads(result.text)

            try:
                output = await evaluate(
                    reports,
                    checker,
                    document_tokens=document_tokens,
                    overlap_tokens=overlap_tokens,
                    context_tokens=context_tokens,
                    output_tokens=output_tokens,
                    safety_tokens=safety_tokens,
                    batch_size=batch_size,
                )
                output["configured_model"] = model_name
                output["budget"] = (
                    backend.summary() if isinstance(backend, BudgetBackend) else None
                )
                output["run_budget_usd"] = (
                    run_budget_usd if isinstance(backend, BudgetBackend) else None
                )
                return output
            finally:
                if backend is not None:
                    await backend.aclose()

        result = asyncio.run(run())
        # Exact endpoint model identity is supplied by framework configuration,
        # never hardcoded here; resolved configuration is also exported by runner.
        result["reasoning_effort"] = "high"
        outdir = Path(outdir)
        private_directory(outdir)
        write_private_text(
            outdir / f"{filebase}.support.json",
            json.dumps(result, ensure_ascii=False, indent=2),
        )
        write_private_text(
            outdir / f"{filebase}.supported-claims.jsonl",
            "".join(
                json.dumps(p, ensure_ascii=False) + "\n"
                for p in result["supported_claims"]
            ),
        )
        if not result["answers"] or any(not a["complete"] for a in result["answers"]):
            raise JudgeError(
                "Incomplete citation assessment; diagnostics saved, leaderboard withheld."
            )
        builder = LeaderboardBuilder(SPEC)
        for a in result["answers"]:
            builder.add(
                run_id=a["run_id"],
                topic_id=a["topic_id"],
                values={
                    "DOCUMENT_SUPPORT_PROXY": a["score"]
                    if a["score"] is not None
                    else 0.0,
                    "HAS_CITATIONS": float(a["expected_pairs"] > 0),
                    "HAS_ELIGIBLE_CITATIONS": float(a["eligible_pairs"] > 0),
                },
            )
        return builder.build(expected_topic_ids=topics, on_missing="error")
