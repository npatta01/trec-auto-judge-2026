"""Owner-only local artifacts; callers should use dedicated output directories."""

import os
import sys
from threading import RLock
from pathlib import Path

from .models import JudgeError


class _QuietOutput:
    """Share process-wide output suppression across overlapping contexts."""

    _lock = RLock()
    _active = 0
    _saved_stdout = None
    _saved_stderr = None
    _stdout_sink = None
    _stderr_sink = None

    def __enter__(self):
        cls = type(self)
        with cls._lock:
            if cls._active == 0:
                stdout_sink = open(os.devnull, "w", encoding="utf-8")
                try:
                    stderr_sink = open(os.devnull, "w", encoding="utf-8")
                except BaseException:
                    stdout_sink.close()
                    raise
                cls._saved_stdout, cls._saved_stderr = sys.stdout, sys.stderr
                cls._stdout_sink, cls._stderr_sink = stdout_sink, stderr_sink
                sys.stdout, sys.stderr = stdout_sink, stderr_sink
            cls._active += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cls = type(self)
        with cls._lock:
            if cls._active <= 0:
                raise RuntimeError("quiet_output context exited without entry")
            cls._active -= 1
            if cls._active == 0:
                stdout_sink, stderr_sink = cls._stdout_sink, cls._stderr_sink
                sys.stdout, sys.stderr = cls._saved_stdout, cls._saved_stderr
                cls._saved_stdout = cls._saved_stderr = None
                cls._stdout_sink = cls._stderr_sink = None
                stdout_sink.close()
                stderr_sink.close()
        return False


def quiet_output() -> _QuietOutput:
    """Suppress stdout/stderr until the last overlapping context exits."""
    return _QuietOutput()


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
