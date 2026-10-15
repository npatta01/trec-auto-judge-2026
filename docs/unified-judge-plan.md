# Shared judge implementation

`judges/generic/unified-workflow.yml` selects six variants: RAG/RAGTIME × pairwise/evidence/combined. Source and tests are preserved from frozen `c37a65716df83ba181f7080a38fdbfb117bad82b`; no scoring change is part of the preservation PR.

Pairwise compares answers blindly in both orders and supports a sparse opponent schedule. Evidence reuses document citation judgments, deduplicates supported evidence, prepares one question-guided reference/checklist, and grades each answer against that checklist. Optional supported-portion alignment preserves the distinction between coverage and citation support. Combined ranking defaults to equal pairwise/evidence weight.

The RAG and RAGTIME mappings are [experimental metric proxies](competition-scoring-context.md), not verified implementations of the organizer's final metrics. Full support receives 1, partial support 0.5, and unsupported/unresolved evidence 0. Only explicitly flagged non-claims are excluded from citation proxy denominators.

Endpoint and model are injected. Portable completion caching, bounded completed-output recovery, persistent budget reservations, cohort validation and offline replay are covered by synthetic tests. Replaying requires the same model, settings and saved judgments. A code registration alone does not prove that TIRA received the warm cache.

Use the [ledger](submission-ledger.md) for exact submitted artifact/source distinctions and the [recovery guide](recovery.md) for private artifacts and original source bundles. Historical experiments remain preserved inside the encrypted release.
