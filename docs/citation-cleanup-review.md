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

## Next action

Before relying on OpenRouter offline replay, add a synthetic regression for that
exact provider-configured path and preserve identical semantic request options
online/offline without weakening the paid-call budget guard. This is a separate
behavior fix, not hidden inside the cleanup. Review this result before publishing
the local follow-up; do not merge automatically.
