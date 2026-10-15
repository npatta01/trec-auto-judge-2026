#!/usr/bin/env bash
# Run with: sops exec-env recovery-key.sops.json 'bash ./decrypt-recovery.sh /new/path/private.tar.zst'
set -euo pipefail
umask 077
: "${key:?Run this script through sops exec-env recovery-key.sops.json}"
if [[ $# -ne 1 || -e "$1" ]]; then
  echo 'Supply one new output filename; existing files are never overwritten.' >&2
  exit 2
fi
for tool in python3 sha256sum gpg zstd; do command -v "$tool" >/dev/null; done
sha256sum --check --strict SHA256SUMS >/dev/null
part_list=$(mktemp)
partial=$(mktemp "${1}.partial.XXXXXX")
trap 'rm -f "$part_list" "$partial"' EXIT
python3 - <<'PY_PARTS' > "$part_list"
import json,re
from pathlib import Path
parts=json.loads(Path('recovery-manifest.json').read_text())['parts']
assert parts
for i,p in enumerate(parts):
    assert p['name']==f'private-recovery.tar.zst.gpg.part-{i:03d}'
    f=Path(p['name']); assert f.is_file() and not f.is_symlink()
    assert f.stat().st_size==p['size']
    print(p['name'])
PY_PARTS
mapfile -t parts < "$part_list"
# GnuPG verifies OpenPGP integrity before this script offers an archive for extraction.
cat "${parts[@]}" | env -u key gpg --no-options --batch --yes --pinentry-mode loopback \
  --no-symkey-cache --passphrase-fd 3 --output "$partial" --decrypt 3<<<"$key"
unset key
zstd --test --quiet "$partial"
mv -- "$partial" "$1"
echo 'Authenticated private archive ready. Restore only into owner-only local storage.'
