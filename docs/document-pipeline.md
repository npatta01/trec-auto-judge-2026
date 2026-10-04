# Document-first citation support

The selected workflow is `judges/generic/document-workflow.yml`. Install the
project's `minima-llm` and `test` extras. Configure the endpoint through the
organizer's environment contract; the tested model is `openai/gpt-6-luna` with
high reasoning. Never put credentials in commands or commit runtime artifacts.

```bash
auto-judge run --workflow judges/generic/document-workflow.yml \
  --rag-responses data/kiddie/runs/repgen/ \
  --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
  --out-dir output/document-support-kiddie
```

## Input and processing

- Input: framework reports, original answer units, citations and supplied documents.
- Group by topic, document ID and content hash across answers, preserving each
  occurrence's run, statement index, citation position and source identity.
- Judge up to five units per request with indexed answer context. Other answers
  are never evidence. No new retrieval or generated reference answer is used.
- Reserve document, output and safety tokens. Documents above the 24,000-token
  estimate are split with overlap; reduce claim batch size for context pressure.
  Tokenization is an estimate, not a verified exact provider tokenizer. Never
  silently truncate; unresolvable overflow remains an explicit failure.
- Three support classes: supported, partially supported, unsupported. Strongest
  chunk wins; partial chunks do not combine into full support. Uncertainty and
  contradiction prevent acceptance. Missing or failed chunks are not negatives.
- Same-call eligibility marks claims, non-claims or incomplete units. Exclude
  only on complete, unanimous confident agreement across all citations/chunks.
  Otherwise retain the unit, without promoting its support judgment.

## Outputs and scoring

The framework writes numeric leaderboards. Private `.support.json` sidecars
retain per-pair/chunk judgments, provenance, prompt version/hash, model and budget
summary. `.supported-claims.jsonl` exports accepted original units and their source
pointers. These are not atomic gold claims or a generated reference bank.

`DOCUMENT_SUPPORT_PROXY = accepted pairs / eligible citation pairs`.
Partial support receives no fractional credit. Sidecars also expose each class's
percentage and supported-or-partial percentage. Empty denominators are null in
sidecars and zero in the leaderboard, with `HAS_CITATIONS` and
`HAS_ELIGIBLE_CITATIONS` distinguishing them. Incomplete assessments withhold
the leaderboard. These are research proxies, not official competition formulas.

## Execution safeguards

OpenRouter calls use a persistent ledger: default $16 cumulative cap and $0.50
per invocation, no retries, price ceilings $1/M input and $4/M output, no request
fee or provider fallback. Set `MAX_ATTEMPTS=1`. Existing ledger caps cannot be
increased; do not change ledger files to bypass previous spending. Unknown costs
remain reserved. Non-OpenRouter endpoints do not have this dollar-budget guard.

Artifacts and ledgers are owner-only. Restricted evaluation text must not be
inspected or published. Budget/configuration/transport failures stop further
calls; diagnostics contain safe categories, not provider exception bodies.
The current implementation holds reports in memory and is not a streaming engine.

## Modules and verification

`document_pipeline.py` owns preparation, batching, judging, eligibility and answer
summaries. `chunk_aggregation.py` owns strongest-chunk aggregation.
`document_judge.py` adapts this to the framework. `budget.py` and `private_io.py`
provide bounded spending and artifact handling. Shared client/model/normalization
modules and the earlier generic workflow remain for compatibility and regression
tests; they are not the selected document-stage scoring procedure.

Run `python -m pytest -q`. Tests include synthetic labels, chunk coverage, failure
handling, retention, per-invocation accounting and real CLI/loopback integration.
The [v4 live check](v4-live-results.md) processed 80 real citation pairs for
$0.020631725. This verifies a small workflow sample, not broad semantic accuracy.
