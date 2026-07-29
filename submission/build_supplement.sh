#!/bin/bash
# Build the anonymous code-and-data supplement.
#
# Whitelist, not blacklist. The repo working tree also holds `companion/` (which
# contains an OpenReview PDF listing all author names) and `meta/`; an exclude-based
# archive is one forgotten pattern away from breaking double-blind review, so this
# script copies only paths named explicitly below and then greps the result for
# leaks before packaging.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d)"
OUT="${1:-$HERE/midc_supplement.zip}"
trap 'rm -rf "$STAGE"' EXIT

DEST="$STAGE/midc_supplement"
mkdir -p "$DEST"

copy() {  # copy() <relative path under submission/>
  local src="$HERE/$1"
  if [ ! -e "$src" ]; then echo "  MISSING (skipped): $1"; return; fi
  mkdir -p "$DEST/$(dirname "$1")"
  cp -R "$src" "$DEST/$(dirname "$1")/"
}

echo "staging code ..."
for f in round1 round2/*.py round2/*.sh; do copy "$f"; done

echo "staging data ..."
# Manifests live inside each results tree.
copy round2/results_r2/manifests
copy round2/results_flux2/manifests
copy round2/results_clip
copy round2/results_blur_cf
copy round2/human_eval
copy round2/b1_reanalysis.json
copy round2/REPORT_seed012.md
for d in "$HERE"/round2/results_r2/merged/*/ "$HERE"/round2/results_flux2/*/ \
         "$HERE"/round2/results_ablation/*/; do
  [ -f "$d/records.jsonl" ] || continue
  rel="${d#$HERE/}"; rel="${rel%/}"
  mkdir -p "$DEST/$rel"
  cp "$d/records.jsonl" "$DEST/$rel/"
done

echo "staging docs ..."
copy paper/SUPPLEMENTARY.md
copy paper/ReproducibilityChecklist.pdf
copy README.md

# Never ship generated images, build artifacts, caches or virtualenvs.
find "$DEST" \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.log' \
  -o -name '*.aux' -o -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true
find "$DEST" \( -name '__pycache__' -o -name '.venvs' -o -name 'images' \) \
  -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "anonymity check ..."
fail=0
scan() {  # scan <label> <grep-args...>
  local label="$1"; shift
  local hits
  hits="$(grep -rIl "$@" "$DEST" 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    echo "  LEAK ($label):"; echo "$hits" | sed 's/^/    /'; fail=1
  else
    echo "  ok: no $label"
  fi
}
scan "server IPs"        -E '216\.81\.[0-9]+\.[0-9]+'
scan "home paths"        -F '/Users/'
scan "author names"      -iE 'zhihan|yuhuan|yijie|mengcong|suwen|qiuyang|ejzhu'
# `bytedance/UMO` and `bytedance-research/UMO` are the public repositories of a
# cited baseline that any reader would clone; they are not an affiliation leak.
# Flag the org name only where it is NOT part of those public paths.
hits="$(grep -rIn -iE 'bytedance' "$DEST" 2>/dev/null \
        | grep -viE 'bytedance/UMO|bytedance-research/UMO|ByteDance-FanQie' || true)"
if [ -n "$hits" ]; then
  echo "  LEAK (org name outside public repo paths):"; echo "$hits" | sed 's/^/    /'; fail=1
else
  echo "  ok: org name only in public baseline repo paths"
fi
[ "$fail" -eq 0 ] || { echo; echo "REFUSING TO PACKAGE: fix the leaks above."; exit 1; }

echo
echo "packaging ..."
rm -f "$OUT"
( cd "$STAGE" && zip -q -r "$OUT" midc_supplement )
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "files: $(find "$DEST" -type f | wc -l | tr -d ' ')"
