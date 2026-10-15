# Recover this project after loss of the original folder

The preservation release is [recovery-2026-10-15](https://github.com/npatta01/trec-auto-judge-2026/releases/tag/recovery-2026-10-15). Its tag anchors the PR source even before the PR is merged into main.

## Public source and submitted scores

Clone the release tag to recover the frozen c37 shared judge, its synthetic tests, six numeric submission packages, and two sanitized receipts:

```bash
git clone --branch recovery-2026-10-15 https://github.com/npatta01/trec-auto-judge-2026.git trec-autojudge
cd trec-autojudge
uv venv
source .venv/bin/activate
uv pip install -e '.[all]'
pytest -q
```

The release also carries a standalone `public-source.bundle` for recovery without relying on the PR branch. Package/source distinctions are recorded in the [ledger](submission-ledger.md). Four TIRA code submissions remain pending; this preservation release does not complete them.

## Private artifacts

The release contains thirteen ordered ciphertext parts, a SOPS-encrypted passphrase, a manifest, checksums and the decryption helper. The underlying archive includes:

- The independent private Git snapshot of 4,081,981 selected useful files: release inputs, saved references/checklists/judgments, current and historical caches/stages, numeric/replay proofs and accounting.
- Both pairwise frozen source bundles, the c37 evidence/combined bundle, all current source refs, and the original working source/documentation snapshot.
- The frozen container image and four previously SOPS-encrypted TIRA claim receipts.

Reproducible virtual environments, compiled caches, plaintext credentials and raw logs are excluded. Original detailed research docs and the private answer explorer are in `working-snapshot/`, not exposed by this public repository.

Install Git, Bash, Python 3, GnuPG, zstd, SOPS and the GitHub CLI. Recovery requires authorized decrypt access to the existing Google Cloud KMS key `projects/np-public-training/locations/global/keyRings/sops/cryptoKeys/sops-key`. Authenticate on the replacement machine with Google Application Default Credentials; no key or credential from the original project folder is required. Keep that KMS key available.

Download the release assets into a new private directory:

```bash
umask 077
mkdir trec-private-recovery
cd trec-private-recovery
gh release download recovery-2026-10-15 --repo npatta01/trec-auto-judge-2026
sops exec-env recovery-key.sops.json 'bash ./decrypt-recovery.sh private.tar.zst'
```

The helper checks every ciphertext part, rejects missing/reordered filenames, verifies GnuPG integrity, and tests zstd before offering the archive for extraction. It never overwrites an existing output. Do not extract a partial or unauthenticated result. Allow at least 90 GB for downloaded parts, the authenticated archive and restored Git objects, plus space for any artifact tree you choose to expand.

Recover the private Git storage and source snapshot:

```bash
mkdir recovered
tar --zstd -xf private.tar.zst -C recovered
git --git-dir=recovered/private-artifacts.git show-ref
git clone --branch codex/shared-track-judge recovered/source-history-20261015.private.bundle working-history
```

`recovered/working-snapshot/` retains uncommitted working source and historical documents. `private-artifacts.git` contains the `private-snapshot` and `archive-control` branches. The latter retains the frozen image and encrypted claim receipts. Prefer selective extraction from `private-snapshot`: expanding all historical duplicate paths uses far more space than the compressed Git storage. Restore raw artifacts only into owner-only local storage. The [organizer data policy](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/data-policy.md) continues to apply.

`recovery-verification.json` on the release records the final verification gates. The older `status.json` inside the archived 2026-10-12 backup is historical; it must not be mistaken for the release's current closure record.

## Verification limit at publication

The ciphertext transfer, full OpenPGP authentication, zstd/tar structure, safe source/ref restoration, native Git connectivity and strict loose-object checks passed. The full native scan including the multi-gigabyte packed blobs is still running; its success is not claimed. Keep the original and independent local backup while that deep scan is pending. The release already preserves their selected useful files off-machine.
