"""Streaming, reference-free direct and staged AutoJudge implementation."""
import asyncio
import json
from pathlib import Path

from autojudge_base import LeaderboardBuilder

from .client import JsonClient, make_backend
from .models import Assessment, Audit, Brief, DirectResult, Evidence, JudgeError, SpanAudit
from .scoring import DESCRIPTIONS, SPEC, normalize_report, request_payload, scores, validate_evidence
from .spans import source_units, materialize


def chunks(text, size, overlap):
    start = 0
    while start < len(text):
        yield text[start:start + size]
        if start + size >= len(text):
            break
        start += size - overlap


def merge_evidence(labels):
    statuses = {e.status for e in labels}
    # Do not allow a positive chunk to hide explicit conflicting evidence.
    if 'contradicted' in statuses and statuses & {'supported', 'partial'}:
        return next(e for e in labels if e.status in {'supported', 'partial'}).model_copy(update={'status': 'partial'})
    for status in ('supported', 'partial', 'contradicted', 'unsupported'):
        for e in labels:
            if e.status == status:
                return e
    raise JudgeError('No usable document-chunk labels.')


class GenericJudge:
    def __init__(self, backend_factory=None):
        self.backend_factory = backend_factory

    def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, qrels=None,
              filebase='generic', outdir=Path('.'), mode='staged', max_prompt_chars=120000,
              max_answer_chars=40000, document_chunk_chars=12000, chunk_overlap=256,
              max_tokens=8192, schema_attempts=2, structured_output=False, evidence_format='quotes', request_extra=None, **kwargs):
        sizes = (max_prompt_chars, max_answer_chars, document_chunk_chars, max_tokens, schema_attempts)
        if mode not in ('direct', 'staged') or any(type(v) is not int or v <= 0 for v in sizes):
            raise JudgeError('Invalid judge mode or positive-integer budget.')
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < document_chunk_chars:
            raise JudgeError('Chunk overlap must be nonnegative and smaller than chunk size.')
        if evidence_format not in ('quotes', 'spans') or (mode == 'direct' and evidence_format != 'quotes'):
            raise JudgeError('Numbered spans require staged mode.')
        return asyncio.run(self._run(rag_responses, rag_topics, llm_config, mode,
            max_prompt_chars, max_answer_chars, document_chunk_chars, chunk_overlap, max_tokens, schema_attempts, structured_output, evidence_format, request_extra))

    async def _run(self, reports, topics, llm_config, mode, prompt_limit, answer_limit,
                   chunk_size, overlap, max_tokens, attempts, structured_output, evidence_format, request_extra):
        topic_map = {q.request_id: request_payload(q) for q in topics}
        if (len(topic_map) != len(topics) or not topic_map or 'all' in topic_map
                or any(not t.strip() for t in topic_map)):
            raise JudgeError('Topic IDs must be nonempty, unique and not the aggregate ID.')
        builder = LeaderboardBuilder(SPEC)
        seen, briefs = set(), {}
        backend = client = None
        try:
            for report in reports:
                run, topic = report.metadata.run_id, report.metadata.topic_id
                if topic not in topic_map or (run, topic) in seen:
                    raise JudgeError('Unknown topic or duplicate run/topic input.')
                seen.add((run, topic))
                answer = normalize_report(report)
                if sum(len(s['text'] or '') for s in answer['sentences']) > answer_limit:
                    raise JudgeError('Answer character budget exceeded; no text was truncated.')
                if not any((s['text'] or '').strip() for s in answer['sentences']):
                    values = {m: 0. for m in DESCRIPTIONS}
                else:
                    if client is None:
                        backend = (self.backend_factory or make_backend)(llm_config)
                        client = JsonClient(backend, max_prompt_chars=prompt_limit,
                                            max_tokens=max_tokens, schema_attempts=attempts,
                                            structured_output=structured_output, request_extra=request_extra)
                    request = topic_map[topic]
                    missing = [Evidence(sentence_id=s['sentence_id'], document_id=d,
                               status='unverified', quote='') for s in answer['sentences']
                               for d in s['citations'] if d not in answer['documents']]
                    if mode == 'direct':
                        expected = [{'sentence_id': s['sentence_id'], 'document_id': d}
                                    for s in answer['sentences'] for d in s['citations'] if d in answer['documents']]
                        result = await client.ask('direct', {'request': request, 'answer': answer,
                            'required_evidence_pairs': expected}, DirectResult,
                            validator=lambda r: scores(answer, r.assessment, r.evidence + missing))
                        assessment, evidence = result.assessment, result.evidence + missing
                    else:
                        key = json.dumps(request, sort_keys=True, ensure_ascii=False)
                        if key not in briefs:
                            briefs[key] = await client.ask('interpret', {'request': request}, Brief)
                        evidence = await self._audit(client, request, answer, chunk_size, overlap, evidence_format)
                        evidence += missing
                        assessment = await client.ask('assess', {
                            'request': request, 'brief': briefs[key].model_dump(),
                            'answer': {'sentences': answer['sentences']},
                            'evidence': [e.model_dump() for e in evidence],
                        }, Assessment, validator=lambda a: scores(answer, a, evidence))
                    values = scores(answer, assessment, evidence)
                builder.add(run_id=run, topic_id=topic, values=values)
            if not seen:
                raise JudgeError('No reports supplied.')
            try:
                return builder.build(expected_topic_ids=list(topic_map), on_missing='error')
            except ValueError:
                raise JudgeError('Incomplete run/topic coverage; leaderboard not produced.') from None
        finally:
            if backend is not None:
                await backend.aclose()

    async def _audit(self, client, request, answer, chunk_size, overlap, evidence_format):
        merged = []
        for doc_id, text in answer['documents'].items():
            sentences = [s for s in answer['sentences'] if doc_id in s['citations']]
            labels = {s['sentence_id']: [] for s in sentences}
            expected = {(i, doc_id) for i in labels}
            for chunk in chunks(text, chunk_size, overlap):
                payload = {'request': request, 'sentences': sentences,
                                        'answer_context': answer['sentences'],
                                        'document_id': doc_id,
                                        'required_evidence_pairs': [{'sentence_id': i, 'document_id': doc_id}
                                                                    for i in sorted(labels)]}
                if evidence_format == 'spans':
                    payload['document_units'] = source_units(chunk)
                    audit = await client.ask('audit_spans', payload, SpanAudit,
                        validator=lambda a: materialize(a, chunk, doc_id, set(labels)))
                    chunk_evidence = materialize(audit, chunk, doc_id, set(labels))
                else:
                    payload['document'] = chunk
                    audit = await client.ask('audit', payload, Audit,
                        validator=lambda a: validate_evidence(a.evidence, expected, {doc_id: chunk}))
                    chunk_evidence = audit.evidence
                for label in chunk_evidence:
                    labels[label.sentence_id].append(label)
            merged.extend(merge_evidence(v) for v in labels.values())
        return merged
