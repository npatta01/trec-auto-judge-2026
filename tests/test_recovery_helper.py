"""Synthetic restore controls: real GnuPG; no competition data or credentials."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts/decrypt-recovery.sh"


@pytest.fixture
def recovery(tmp_path):
    if not all(shutil.which(x) for x in ["gpg", "zstd", "bash", "sha256sum"]):
        pytest.skip("recovery CLI tools unavailable")
    plain = tmp_path / "synthetic.tar"
    content = b"manufactured recovery fixture\n"
    with tarfile.open(plain, "w") as tf:
        entry = tarfile.TarInfo("synthetic.txt")
        entry.size = len(content)
        tf.addfile(entry, io.BytesIO(content))
    compressed = subprocess.check_output(["zstd", "-q", "-c", str(plain)])
    reader, writer = os.pipe()
    os.write(writer, b"synthetic-only-passphrase\n")
    os.close(writer)
    try:
        cipher = subprocess.check_output(
            ["gpg", "--no-options", "--batch", "--pinentry-mode", "loopback",
             "--passphrase-fd", str(reader), "--symmetric", "--cipher-algo", "AES256",
             "--compress-algo", "none", "--output", "-"],
            input=compressed, stderr=subprocess.DEVNULL, pass_fds=(reader,))
    finally:
        os.close(reader)
    (tmp_path / "decrypt-recovery.sh").write_bytes(HELPER.read_bytes())
    name = "private-recovery.tar.zst.gpg.part-000"
    (tmp_path / name).write_bytes(cipher)
    manifest = {"parts": [{"name": name, "size": len(cipher)}]}
    env = dict(os.environ, key="synthetic-only-passphrase")
    return tmp_path, manifest, env


def write_controls(folder, manifest):
    (folder / "recovery-manifest.json").write_text(json.dumps(manifest))
    names = ["decrypt-recovery.sh", "recovery-manifest.json"] + [x["name"] for x in manifest["parts"]]
    (folder / "SHA256SUMS").write_text("".join(
        hashlib.sha256((folder / name).read_bytes()).hexdigest() + "  " + name + "\n"
        for name in names))


def test_existing_destination_created_during_decrypt_is_preserved(recovery):
    folder, manifest, env = recovery
    write_controls(folder, manifest)
    real_gpg = shutil.which("gpg")
    bindir = folder / "bin"
    bindir.mkdir()
    gate = bindir / "gpg"
    gate.write_text("#!/bin/sh\n: > \"$RESTORE_STARTED\"\n"
                    "while [ ! -f \"$RESTORE_CONTINUE\" ]; do sleep 0.02; done\n"
                    'exec "' + real_gpg + '" "$@"\n')
    gate.chmod(0o755)
    started, proceed = folder / "started", folder / "proceed"
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"],
               RESTORE_STARTED=str(started), RESTORE_CONTINUE=str(proceed))
    destination = folder / "result.tar.zst"
    proc = subprocess.Popen(["bash", "decrypt-recovery.sh", str(destination)],
                            cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists(), "GnuPG gate did not start"
        destination.write_bytes(b"existing user file")
        proceed.touch()
        proc.communicate(timeout=20)
        assert proc.returncode != 0
        assert destination.read_bytes() == b"existing user file"
        assert not list(folder.glob("*.partial.*"))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def test_optimized_python_cannot_disable_part_size_validation(recovery):
    folder, manifest, env = recovery
    manifest["parts"][0]["size"] += 1
    write_controls(folder, manifest)
    env["PYTHONOPTIMIZE"] = "1"
    destination = folder / "result.tar.zst"
    result = subprocess.run(["bash", "decrypt-recovery.sh", str(destination)],
                            cwd=folder, env=env, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert not destination.exists()
    assert not list(folder.glob("*.partial.*"))
