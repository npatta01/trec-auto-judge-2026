# Competition scoring context

This preservation checkpoint retains the submitted shared judge without changing its scoring. It keeps blind pairwise, evidence, and combined approaches separate, with track-specific RAG and RAGTIME proxy mappings.

RAG evidence combines request coverage and partial-credit citation precision/recall. Citation precision averages retained individual citation pairs; recall uses the best individual citation for each substantive segment. RAGTIME evidence combines factual grounded checklist coverage and top-priority citation support. Partial support contributes 0.5. Combined defaults to equal pairwise and evidence weights. All component columns remain available. These are experimental proxies, not confirmed final organizer weights.

Preserve existing judgments and provenance. Do not automatically rejudge citations, replace the baseline, tune on restricted topics, or infer semantic quality from successful replay. Generated reference/checklist content is not human gold.

The original detailed scoring investigation, source links, decisions and historical test results are preserved in the encrypted release's `working-snapshot/docs/competition-scoring-context.md`. The public [implementation guide](unified-judge-plan.md), [submission ledger](submission-ledger.md) and [recovery guide](recovery.md) describe the retained checkpoint.

Official sources: [AutoJudge Participant HowTo](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/README.md), [submission requirements](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/07-submit-to-tira.md), [data policy](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/data-policy.md).
