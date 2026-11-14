"""Local/TIRA entry point. Runs a selected stage; never uploads anything."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("citation", "shared", "pairwise", "unified")
    )
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--responses", type=Path, help="Override the dataset's report directory."
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Optional topic-ID filter, repeatable.",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        help="Required when the dataset has multiple topic files.",
    )
    parser.add_argument("--track", choices=("rag", "ragtime"))
    parser.add_argument(
        "--method", choices=("pairwise", "evidence", "combined"), default="combined"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Prepared shared/pairwise bundle; never generated implicitly.",
    )
    parser.add_argument("--run-budget-usd", type=float, default=0.5)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Disable the LLM endpoint; cache misses fail.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the local command without executing.",
    )
    args = parser.parse_args()
    runs = args.responses or args.input_dataset / "runs"
    if not runs.is_dir():
        parser.error("Dataset runs directory is missing.")
    if args.responses is None:
        children = [p for p in runs.iterdir() if p.is_dir()]
        if children:
            if len(children) != 1 or any(p.is_file() for p in runs.iterdir()):
                parser.error("Select one report directory with --responses.")
            runs = children[0]
    topics = (
        [args.topics]
        if args.topics
        else sorted((args.input_dataset / "topics").glob("*.jsonl"))
    )
    if len(topics) != 1 or not topics[0].is_file():
        parser.error("Select exactly one existing topic file with --topics.")
    if args.stage in ("shared", "pairwise") and (
        not args.track or not args.bundle or not args.bundle.is_file()
    ):
        parser.error(
            "Shared/pairwise replay requires --track and an existing --bundle."
        )
    if args.stage == "unified" and not args.track:
        parser.error("Unified judging requires --track.")
    if args.stage in ("citation", "unified") and (
        not math.isfinite(args.run_budget_usd) or args.run_budget_usd <= 0
    ):
        parser.error("Run budget must be finite and positive.")
    if args.stage in ("citation", "unified") and not os.environ.get("CACHE_DIR"):
        parser.error("Set CACHE_DIR for portable citation replay.")
    filename = (
        "document-workflow.yml"
        if args.stage == "citation"
        else args.stage + "-workflow.yml"
    )
    command = [
        sys.executable,
        "-m",
        "autojudge_base.cli",
        "run",
        "--workflow",
        str(Path(__file__).with_name(filename)),
        "--rag-responses",
        str(runs),
        "--rag-topics",
        str(topics[0]),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.stage in ("citation", "unified"):
        command += ["--set", f"run_budget_usd={args.run_budget_usd}"]
        if args.stage == "unified":
            command += ["--variant", f"{args.track}-{args.method}"]
    else:
        key = "evidence_bundle" if args.stage == "shared" else "pairwise_bundle"
        command += ["--jset", f"track={args.track}", "--jset", f"{key}={args.bundle}"]
    for topic in args.topic:
        command += ["--topic", topic]
    if args.dry_run:
        print(json.dumps(dict(command=command, replay=args.replay)))
        return 0
    env = dict(os.environ)
    if args.replay:
        env.update(
            OPENAI_BASE_URL="EMPTY", OPENAI_API_KEY="EMPTY", CACHE_FORCE_REFRESH="0"
        )
    # Framework reads endpoint/model/cache from the environment. No hardcoded model.
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
