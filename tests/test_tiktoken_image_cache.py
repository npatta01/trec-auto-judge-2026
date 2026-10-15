"""Container-only offline smoke test for the estimator's tokenizer asset."""

import os
import re
import subprocess

import pytest


def test_o200k_base_is_baked_into_the_image_for_offline_use():
    image = os.environ.get("TIKTOKEN_TEST_IMAGE")
    if not image:
        pytest.skip("set TIKTOKEN_TEST_IMAGE to run the container packaging check")
    assert re.fullmatch(r"[A-Za-z0-9_./:@-]+", image)

    code = (
        "import os, tiktoken; "
        "encoding = tiktoken.get_encoding('o200k_base'); "
        "assert encoding.encode('synthetic offline tokenizer check', "
        "disallowed_special=()); "
        "assert os.environ.get('TIKTOKEN_CACHE_DIR') == "
        "'/opt/trec-autojudge/tiktoken-cache'"
    )
    result = subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/venv/bin/python",
            image,
            "-c",
            code,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, "o200k_base was not available without network access"
