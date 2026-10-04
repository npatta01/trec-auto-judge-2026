"""Owner-only local artifacts; callers should use dedicated output directories."""

import os
from pathlib import Path

from .models import JudgeError


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise JudgeError("Private artifact directory cannot be a symlink.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Do not alter permissions on a workspace or filesystem root when a caller
    # uses the framework's default outdir='.'. Individual files remain private.
    if path.resolve() not in (Path.cwd().resolve(), Path(path.anchor or "/").resolve()):
        path.chmod(0o700)


def prepare_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def write_private_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.truncate(0)
        stream.write(text)
