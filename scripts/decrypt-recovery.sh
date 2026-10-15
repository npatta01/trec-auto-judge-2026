#!/usr/bin/env bash
# Run with: sops exec-env recovery-key.sops.json 'bash ./decrypt-recovery.sh /new/path/private.tar.zst'
set -euo pipefail
umask 077
: "${key:?Run this script through sops exec-env recovery-key.sops.json}"
task_recovery_passphrase=$key
export -n task_recovery_passphrase
unset key
if [[ $# -ne 1 || -e "$1" || -L "$1" ]]; then
  echo 'Supply one new output filename; existing files are never overwritten.' >&2
  exit 2
fi
for tool in python3 sha256sum gpg zstd; do command -v "$tool" >/dev/null; done
sha256sum --check --strict SHA256SUMS >/dev/null
part_list=$(mktemp)
partial=$(mktemp "${1}.partial.XXXXXX")
trap 'rm -f "$part_list" "$partial"' EXIT
python3 - <<'PY_PARTS' > "$part_list"
import json
from pathlib import Path
parts=json.loads(Path('recovery-manifest.json').read_text())['parts']
if not parts:
    raise ValueError("Recovery manifest contains no parts")
for i,p in enumerate(parts):
    if p['name'] != f'private-recovery.tar.zst.gpg.part-{i:03d}':
        raise ValueError('Unexpected part name or order')
    f=Path(p['name'])
    if not f.is_file() or f.is_symlink() or f.stat().st_size != p['size']:
        raise ValueError('Missing, linked or incorrectly sized part')
    print(p['name'])
PY_PARTS
parts=()
while IFS= read -r part; do parts+=("$part"); done < "$part_list"
# GnuPG verifies OpenPGP integrity before this script offers an archive for extraction.
cat "${parts[@]}" | env -u key gpg --no-options --batch --yes --pinentry-mode loopback \
  --no-symkey-cache --passphrase-fd 3 --output "$partial" --decrypt 3<<<"$task_recovery_passphrase"
unset task_recovery_passphrase
zstd --test --quiet "$partial"
# Both paths are in the same directory. Atomic hard-link creation never replaces
# a file, directory or symlink that appeared while decryption was running.
python3 - "$partial" "$1" <<'PY_FINALIZE'
import os,sys
os.link(sys.argv[1],sys.argv[2])
os.unlink(sys.argv[1])
PY_FINALIZE
echo 'Authenticated private archive ready. Restore only into owner-only local storage.'
