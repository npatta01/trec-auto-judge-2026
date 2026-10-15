# AutoJudge preservation plan

Goal: recover useful code and artifacts after loss of the original project folder or machine. User explicitly authorized a PR or release on 2026-10-15.

Scope: public code/numeric artifacts in a PR based on current main; private artifacts in authenticated-encrypted release assets. Do not merge automatically, rerun judging, change budgets, or submit to TIRA.

- [x] Refresh origin/main and inspect remote visibility, existing PR and receipts.
- [x] Confirm six live TIRA data groups with evaluation records; only two matching final code registrations.
- [x] Establish isolated integration checkout outside the original folder.
- [x] Preserve exact frozen c37 submitted source and tests; preserve stale/experimental working source separately in encrypted archive.
- [ ] Close integrity and useful-file coverage of private archive; retain source history, inputs, caches, judgments, accounting, image and encrypted receipts.
- [x] Build multi-part AES-256 OpenPGP with integrity protection ciphertext with recovery key protected by existing GCP KMS via SOPS; never publish plaintext private payloads or keys.
- [x] Verify synthetic tests, numeric packages, publication boundary, encryption authentication, private restore and portable recovery docs.
- [ ] Create PR and publish encrypted release; verify remote assets and fresh source recovery.

Decisions: remote main c8b0ee5; frozen evidence/combined source c37a65716df83ba181f7080a38fdbfb117bad82b; original work branch 2dc5990. Repo is public. Existing 4,081,981-file private backup was not closed; fresh integrity verification started. Plaintext credentials and raw logs are excluded; reusable encrypted configuration remains inside ciphertext. Recovery requires authorized access to existing GCP KMS, independent of the original folder.

Current phase: integration and private backup verification.

Verification: 587 passed, 1 skipped, 2 expected failures; deterministic kiddie smoke passed; 20 numeric/receipt files byte-match originals; no high-confidence credential matches; all public guide links resolve. Full GnuPG authentication and zstd validation passed; all 680,129 tar members parsed, source/ref restoration and standalone history clone passed. Both pairwise source bundles verify from the independent archive. Structural artifact scan visited 4,085,394 files and found zero post-snapshot artifact candidates. Full private Git fsck remains in progress.

Final publication checkpoint: PR #4 is open, all thirteen ciphertext parts are uploaded and their GitHub sizes/SHA256 digests match. Updated restore controls pass race, optimization and environment-isolation regressions. Final local suite: 590 passed, 1 skipped, 2 expected failures, with TMPDIR on the backup filesystem after /tmp reached its inode limit. Native connectivity and strict loose-object fsck passed. The full native pack-inclusive scan remains running; do not claim it passed or authorize deletion of local copies. The optional custom checker is not used as a closure gate. Release publication is the next external step; recovery-verification.json is authoritative for its final result.
