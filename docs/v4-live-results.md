# V4 live verification — 2026-10-03

Current code and prompt `document-support-v4-indexed-context` were run through the
production DocumentJudge/BudgetBackend on the same five unrestricted RAGTIME 2025
answers for topic 1033. Model: `openai/gpt-6-luna`, high reasoning. Fixed prompt,
no retries or selection sweep. All 37 requests were fresh and used full documents.

- All 80 citation pairs completed; `run_failure` is null and the adapter returned
  a verified leaderboard object. CLI file serialization is separately tested.
- All 36 substantive units (66 pairs) retained as eligible claims.
- Introduction retained; unfinished fragment's 3 citation pairs excluded.
- 14 supported pair exports; 30 partially supported, 33 unsupported, 3 excluded.
- Provider-reported cost **$0.020631725**, matched to this invocation's 37 ledger
  rows. Shared legacy budget history was preserved through the schema upgrade.
- Fresh automated suite: **105 passed, 2 expected failures**.

| Answer alias | Eligible pairs | Supported | Partial | Unsupported |
|---|---:|---:|---:|---:|
| A2 | 12 | 9 | 3 | 0 |
| A4 | 50 | 1 | 16 | 33 |
| A5 | 15 | 4 | 11 | 0 |

A1/A3 have no cited units and undefined sidecar scores. Four previously supported
v3 pairs became partial, and another became supported. These changes demonstrate
that prompt changes/sampling affect semantic judgments; no improved accuracy is
claimed. In particular, preserving eligibility is not the same as accepting a
claim as supported. This is a small workflow/retention check, not a guarantee of
individual labels or leaderboard-ranking accuracy.

Private local artifacts: `output/budget20/v4-live-2026-10-03/`.
Reproducer: `output/budget20/v4_live.py`, with a started-file rebilling guard.
Historical v3 artifacts are unchanged.

Integration: prepared for a scoped feature-branch PR under an explicit one-time
publication authorization. Raw artifacts, caches, credentials and datasets remain
local. This authorization does not include merging the PR.
