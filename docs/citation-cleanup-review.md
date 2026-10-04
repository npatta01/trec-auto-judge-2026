# Citation-stage cleanup and review

## Objective and scope

Keep the citation stage self-contained and simple enough for research. Remove
the older direct/staged judge and its exclusive tests without changing the v4
prompt, support scoring, chunk handling, budget behavior or provenance contract.
Worktree: `/tmp/trec-citation-pr-v4`; branch: `codex/citation-support-v4`.
The original development checkout is untouched. This follow-up is local only.

## Changes

- Extract report normalization into `reports.py`; remove legacy scoring,
  prompts, span handling, grade schemas and retry client.
- Move the synthetic report builder into test utilities.
- Make the citation workflow the standard `workflow.yml`, replacing the
  older experimental `document-workflow.yml` command path.
- Retain normalization coverage independently and test CLI loopback replay.

## Verification

- Full suite after cleanup and formatting: 77 passed, 2 expected failures.
  The count is lower because legacy-only tests were removed with their code.
- Ruff F checks and whitespace checks pass.
- AST comparisons confirm report normalization and backend construction are
  unchanged. The pipeline has only an import change; adapter, aggregation,
  budget and private artifact modules are byte-identical to the PR version.
- No new paid model calls; the prior live v4 result is historical evidence.

## Independent feedback

Astra: clean enough for a research merge; no cleanup regression and no need for
additional abstractions. It confirmed one material preexisting limitation with
a synthetic check against minima-llm's cache-key function: OpenRouter provider
options enter the online key, but offline EMPTY-endpoint requests omit them.
Therefore loopback replay does not establish OpenRouter offline reproducibility.

Opus: looks good to merge with minor issues; no stale imports, extraction is
behavior-preserving, and the loopback replay test is valid. It could not execute
tests because its shell sandbox failed; test evidence is from the primary agent.
Minor observations: DocumentJudge lacks a blank-topic-ID guard; the shared ledger
path is relative to the working directory; the $0.50 per-run cap is a development
cap, not a full-track budget. Optional style/doc nits do not justify more refactoring.

## Replay fix follow-up (2026-10-04)

The user authorized the replay bug fix. A failing real-client/cache regression
reproduced the mismatch. Request-option construction now lives in a shared helper;
`replay_provider=openrouter` restores those options for the EMPTY endpoint without
using the paid budget adapter. Existing v4 request keys are preserved. Live calls
still require the budget guard, and cache misses still fail. Documentation records
the explicit offline setting rather than guessing the provider from a model name.

Verification: real-client/cache replay with synthetic HTTP responses, live-budget
denial even with the replay option, unknown-provider rejection, and full suite.
Result: 79 passed, 2 expected failures; Ruff F and diff whitespace checks pass.
No credentials, existing caches or paid endpoints were accessed for these tests.
The cleanup and fix remain local; do not merge automatically.
