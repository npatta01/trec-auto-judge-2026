"""Synthetic historical citation receipts through the production cache writer."""

import asyncio
import json
from dataclasses import replace

from minima_llm import MinimaLlmRequest

from judges.generic.completion_session import CompletionSession
from judges.generic.document_pipeline import Verdicts, evaluate, messages


def seed_legacy_receipts(backend, ledger, reports, *, alias=False, repair=False, max_tokens=8192):
    session = CompletionSession(None, ledger, 3, factory=lambda _: backend)

    async def checker(payload):
        if alias:
            payload = dict(payload, claims=[dict(claim, id=f'C{i}')
                for i, claim in enumerate(payload['claims'], 1)])
        req = MinimaLlmRequest(
            'document-support-short-ids-v1' if alias else 'document-support',
            messages(payload), max_tokens=max_tokens,
            extra=dict(reasoning=dict(effort='high'), response_format=dict(
                type='json_schema', json_schema=dict(name='ClaimSupport', strict=True,
                                                    schema=Verdicts.model_json_schema()))))
        result = await session.generate(req)
        if repair:
            result = await session.generate(replace(req,
                request_id=req.request_id + '-output-repair-v1', max_tokens=32768))
        return json.loads(result.text)

    asyncio.run(evaluate(reports, checker))
