"""Real client/cache with a synthetic HTTP response; no external network."""

import json
from types import SimpleNamespace

import pytest
from autojudge_base import Request
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

from judges.generic.document_judge import DocumentJudge
from judges.generic.models import JudgeError
from tests.report_fixtures import report


def test_unknown_replay_provider_fails_before_backend_creation(tmp_path):
    def forbidden_backend(_):
        pytest.fail("Invalid replay configuration constructed a backend")

    with pytest.raises(JudgeError, match="replay provider"):
        DocumentJudge(backend_factory=forbidden_backend).judge(
            [report()],
            [Request(request_id="t", title="Test")],
            SimpleNamespace(),
            outdir=tmp_path,
            replay_provider="typo",
        )


def test_openrouter_cache_replays_offline_without_budget_or_network(
    tmp_path, monkeypatch
):
    online_calls = []

    async def post(self, url, payload):
        assert self.cfg.base_url == "https://openrouter.ai/api/v1"
        online_calls.append(payload)
        assert payload["provider"] == {
            "max_price": {"prompt": 1, "completion": 4, "request": 0},
            "require_parameters": True,
            "allow_fallbacks": False,
            "data_collection": "deny",
        }
        assert payload["reasoning"] == {"effort": "high"}
        data = json.loads(payload["messages"][1]["content"])
        result = {
            "claims": [
                dict(
                    id=c["id"],
                    eligibility="claim",
                    label="supported",
                    uncertain=False,
                    contradiction=False,
                )
                for c in data["claims"]
            ]
        }
        wire = {
            "choices": [
                {"message": {"content": json.dumps(result)}, "finish_reason": "stop"}
            ],
            "usage": {"cost": 0.001},
        }
        return 200, {}, json.dumps(wire).encode()

    monkeypatch.setattr(OpenAIMinimaLlm, "_post_json", post)

    def backend(endpoint):
        return OpenAIMinimaLlm(
            MinimaLlmConfig(
                base_url=endpoint,
                model="synthetic-luna",
                api_key="synthetic-only",
                cache_dir=str(tmp_path / "cache"),
                max_attempts=1,
                rpm=0,
            )
        )

    def run(client, folder, **settings):
        return DocumentJudge(backend_factory=lambda _: client).judge(
            [report()],
            [Request(request_id="t", title="Test")],
            SimpleNamespace(),
            outdir=tmp_path / folder,
            **settings,
        )

    online = backend("https://openrouter.ai/api/v1")
    first = run(online, "online", budget_ledger=str(tmp_path / "paid.db"))
    assert len(online_calls) == 2

    offline_misses = []

    async def forbidden_post(self, url, payload):
        # The real transport rejects EMPTY as an invalid URL before networking.
        # Model that failure, and prove cache hits never reach transport at all.
        assert url == "EMPTY/v1/chat/completions"
        offline_misses.append(url)
        raise ValueError("Invalid offline endpoint")

    monkeypatch.setattr(OpenAIMinimaLlm, "_post_json", forbidden_post)
    offline = backend("EMPTY")
    second = run(
        offline,
        "offline",
        replay_provider="openrouter",
        budget_ledger=str(tmp_path / "must-not-create.db"),
        run_budget_usd=0,
    )
    assert offline._pulse.cache_hits == 2
    assert not offline_misses
    assert not (tmp_path / "must-not-create.db").exists()
    assert first == second
    for filename in ["document.supported-claims.jsonl"]:
        assert (tmp_path / "online" / filename).read_text() == (
            tmp_path / "offline" / filename
        ).read_text()
    a = json.loads((tmp_path / "online/document.support.json").read_text())
    b = json.loads((tmp_path / "offline/document.support.json").read_text())
    assert a["pairs"] == b["pairs"]
    assert a["answers"] == b["answers"]
    assert a["budget"]["calls"] == 2 and b["budget"] is None

    # A genuinely new claim is not satisfied by the earlier cached judgments.
    changed = report()
    changed.responses[0].text = "A new claim absent from this cache."
    with pytest.raises(JudgeError, match="Incomplete"):
        DocumentJudge(backend_factory=lambda _: backend("EMPTY")).judge(
            [changed],
            [Request(request_id="t", title="Test")],
            SimpleNamespace(),
            outdir=tmp_path / "miss",
            replay_provider="openrouter",
        )
    assert not (tmp_path / "miss/document.supported-claims.jsonl").read_text()
    assert len(offline_misses) == 1
