"""No network calls: reject unsafe configuration before constructing a client."""

from types import SimpleNamespace

import pytest

from judges.generic.client import make_backend
from judges.generic.models import JudgeError


@pytest.mark.parametrize("variable", ["MINIMA_TRACE_FILE", "MINIMA_DEBUG"])
def test_trace_configuration_is_rejected(variable, monkeypatch):
    monkeypatch.setenv(variable, "enabled")
    with pytest.raises(JudgeError, match="Disable"):
        make_backend(SimpleNamespace(raw={}))


def test_invalid_configuration_does_not_echo_values(monkeypatch):
    monkeypatch.delenv("MINIMA_TRACE_FILE", raising=False)
    monkeypatch.delenv("MINIMA_DEBUG", raising=False)
    with pytest.raises(JudgeError) as caught:
        make_backend(SimpleNamespace(raw={"max_attempts": "PRIVATE_SENTINEL"}))
    assert "PRIVATE_SENTINEL" not in str(caught.value)
