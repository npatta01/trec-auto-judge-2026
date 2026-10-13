"""Budgeted pairwise preparation from a saved shared bundle (no new citations).

Run once per mode. Completed calls resume from immutable content-keyed receipts.
The output replays through pairwise-workflow.yml without a model or credentials.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .budget import BudgetBackend
from .client import make_backend
from .models import JudgeError
from .pairwise import GUIDANCE, judge_topic, prompt_input_digest
from .private_io import write_private_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument(
        "--answers", nargs="+", required=True, help="Saved anonymous answer aliases"
    )
    parser.add_argument("--mode", choices=GUIDANCE, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--ledger", default="output/budget20/budget.sqlite3")
    parser.add_argument("--run-budget", type=float, default=2)
    parser.add_argument("--max-pairs", type=int, default=100)
    args = parser.parse_args()

    async def run():
        topics = json.loads(Path(args.bundle).read_text())["topics"]
        matches = [t for t in topics if t["topic_id"] == args.topic]
        if len(matches) != 1 or len(set(args.answers)) != len(args.answers):
            raise JudgeError("Invalid topic or duplicate answer selection.")
        topic = matches[0]
        # Preserve source run order before assigning internal sorting IDs. Aliases
        # are used only for reports; no run identifier reaches the model.
        records = sorted(
            [r for r in topic["answers"] if r["evidence"]["id"] in args.answers],
            key=lambda r: r["run_id"],
        )
        if len(records) != len(args.answers):
            raise JudgeError("Requested answers are missing or duplicated.")
        packet = dict(
            question=topic["pool"]["question"],
            checklist=topic.get("checklist"),
            answers=[
                dict(r["evidence"], id=f"C{i:04d}") for i, r in enumerate(records)
            ],
        )
        if args.reference:
            packet["reference"] = json.loads(Path(args.reference).read_text())
        backend = make_backend(SimpleNamespace(raw=None))
        try:
            backend = BudgetBackend(
                backend,
                args.ledger,
                incremental_cap=args.run_budget,
                prompt_price=2,
                completion_price=10,
                max_output_tokens=8192,
            )
            backend.run_id = (
                "pairwise-"
                + hashlib.sha256(
                    str(Path(args.checkpoints).resolve()).encode()
                ).hexdigest()
            )
            result = await judge_topic(
                packet,
                backend,
                args.checkpoints,
                mode=args.mode,
                max_pairs=args.max_pairs,
            )
            output = dict(
                configured_model=backend.cfg.model,
                prompt_version="pairwise-v1",
                packet_sha256=hashlib.sha256(
                    json.dumps(packet, sort_keys=True).encode()
                ).hexdigest(),
                topics=[
                    dict(
                        topic_id=args.topic,
                        inputs=[
                            dict(
                                run_id=r["run_id"],
                                id=f"C{i:04d}",
                                alias=r["evidence"]["id"],
                                input_sha256=r["input_sha256"],
                                prompt_input_sha256=prompt_input_digest(
                                    packet["question"], r["evidence"]["sentences"]
                                ),
                            )
                            for i, r in enumerate(records)
                        ],
                        result=result,
                    )
                ],
                budget=backend.summary(),
            )
            write_private_text(
                Path(args.output), json.dumps(output, ensure_ascii=False, indent=2)
            )
            print(
                json.dumps(
                    dict(
                        mode=args.mode,
                        pairs=len(result["pairs"]),
                        disagreements=sum(
                            p["order_disagreement"] for p in result["pairs"]
                        ),
                        budget=backend.summary(),
                    )
                )
            )
        finally:
            await backend.aclose()

    try:
        asyncio.run(run())
    except JudgeError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
