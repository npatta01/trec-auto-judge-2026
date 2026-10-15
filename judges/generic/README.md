# Shared judging workflows

- `workflow.yml`: existing document-first citation baseline (unchanged).
- `unified-workflow.yml`: six shared pipeline variants, two tracks × three methods.
- `shared-workflow.yml` and `pairwise-workflow.yml`: score saved stage bundles.
- `runner.py`: config-driven local entry point with explicit offline replay.

See [architecture, scoring and limitations](../../docs/unified-judge-plan.md).
Tests use synthetic fixtures. Six numeric submission packages and two sanitized
receipts are included. Raw reports, caches, credentials and private stage outputs
are excluded from public commits; see the [recovery guide](../../docs/recovery.md).
