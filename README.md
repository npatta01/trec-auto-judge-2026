# TREC AutoJudge 2026

A shared RAG/RAGTIME judge built on the official [participant starter kit](https://github.com/trec-auto-judge/auto-judge-starter-kit). The six variants implement pairwise, evidence and combined ranking with separate track-specific proxies.

## Source and results

The shared judge and its tests are preserved from frozen source `c37a65716df83ba181f7080a38fdbfb117bad82b`, used for the four evidence/combined data uploads. The two pairwise submissions have their own frozen source bundles and provenance. See the [submission ledger](docs/submission-ledger.md), [implementation guide](docs/unified-judge-plan.md), and [recovery guide](docs/recovery.md).

Six numeric submission packages and two sanitized pairwise receipts are stored under `output/submissions/`. Raw reports, cached completions, private judgments, endpoint secrets and historical research files are retained only inside the encrypted recovery release.

As checked on October 15, all six TIRA data groups have an attached evaluation record. Matching code registrations exist for the two pairwise versions; four evidence/combined code registrations remain pending. All six data runs remain blinded, unpublished and unreviewed. These receipts do not establish official effectiveness or completed remote judge execution.

## Setup and verification

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[all]'
pytest -q
bash run_kiddie.sh
```

Tests use synthetic fixtures and mock endpoints. The smoke command runs the deterministic example on bundled synthetic data. Neither command requires paid judging. The historical starter examples remain available; prepare a separate clean package before a new TIRA code submission.

Use `judges/generic/unified-workflow.yml` for the shared judge. LLM endpoint, model and credentials come from the standard `OPENAI_BASE_URL`, `OPENAI_MODEL` and `OPENAI_API_KEY` environment variables. Set `CACHE_DIR` to private storage. Never print credentials or commit plaintext endpoint configuration.

## Evaluation data policy

Read the organizer's [data-handling policy](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/data-policy.md) before touching evaluation inputs or derived judgments. Coding agents may inspect only unrestricted releases and the first ten topics in the organizer's original topic order. Preserve anonymization; use the organizer's `scrub` tooling for restricted-record diagnostics. Restoring a backup does not relax these rules.

The original starter kit is MIT licensed.
