#!/usr/bin/env bash
# Probe FLUX.2 multi-reference capacity BEFORE the 6/8-subject scaling run.
#
# This is a one-time, ~5-min job on a single GPU. It loads FLUX.2 and tries
# generating with 2/4/6/8 reference images to find the max supported refs.
# The result determines whether the 6/8-subject scaling experiment is feasible
# on FLUX.2 or whether we need to fall back / reduce subject count.
#
# Run on the server (from round2/ dir):
#   MIE_CKPT=<ckpt> bash probe_flux2.sh
#
# Output: prints max supported refs to stdout + saves to results_flux2/probe.json
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

OUT="${OUT:-$PWD/../round2/results_flux2}"
PY="${PY:-../.venvs/omni/bin/python}"
export ROUND1_WORK="$OUT"
mkdir -p "$OUT"

echo "[probe] loading FLUX.2 and testing multi-ref capacity ..."
"$PY" -c "
import sys, json, os
sys.path.insert(0, '.')
from external_generators import _probe_flux2_capacity
max_refs = _probe_flux2_capacity()
out = os.path.join(os.environ.get('ROUND1_WORK', '.'), 'probe.json')
with open(out, 'w') as f:
    json.dump({'max_refs': max_refs, 'feasible_6_8': max_refs >= 6}, f, indent=2)
print(f'[probe] saved -> {out}')
"
echo "[probe] done. Check $OUT/probe.json for max supported refs."
