"""Local runner never uploads and validates stage inputs before execution."""

import json
import os
import subprocess
import sys


def invoke(tmp_path, *options):
    dataset = tmp_path / "dataset"
    (dataset / "runs/repgen").mkdir(parents=True, exist_ok=True)
    (dataset / "topics").mkdir(exist_ok=True)
    (dataset / "topics/topics.jsonl").touch()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "judges.generic.runner",
            "--input-dataset",
            str(dataset),
            "--out-dir",
            str(tmp_path / "out"),
            *options,
        ],
        env=dict(os.environ, CACHE_DIR=str(tmp_path / "cache")),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_citation_dry_run_is_local_and_bounded(tmp_path):
    result = invoke(tmp_path, "--stage", "citation", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["command"]
    assert command[1:4] == ["-m", "autojudge_base.cli", "run"]
    assert "run_budget_usd=0.5" in command
    assert not (tmp_path / "out").exists()


def test_shared_requires_track_and_prepared_bundle(tmp_path):
    result = invoke(tmp_path, "--stage", "shared", "--track", "ragtime", "--dry-run")
    assert result.returncode != 0
    assert "bundle" in result.stderr.lower()


def test_shared_config_preserves_track_and_bundle(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    result = invoke(
        tmp_path,
        "--stage",
        "shared",
        "--track",
        "rag",
        "--bundle",
        str(bundle),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["command"]
    assert "track=rag" in command
    assert f"evidence_bundle={bundle}" in command
