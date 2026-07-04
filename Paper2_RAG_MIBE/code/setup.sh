#!/usr/bin/env bash
# MISC environment setup for A100 / H100.
# - installs deps
# - (optionally) pre-downloads FLUX.2 weights
# - self-checks that Flux2Pipeline imports and whether caption_upsample is supported
#
# Usage:
#   bash setup.sh                 # install + self-check (downloads FLUX.2 on first real run)
#   bash setup.sh --download      # also pre-download FLUX.2 weights now
#   bash setup.sh --skip-gpu      # CPU-only: install + mock smoke check, skip FLUX.2
set -euo pipefail

DOWNLOAD=0
SKIP_GPU=0
for arg in "$@"; do
  case "$arg" in
    --download) DOWNLOAD=1 ;;
    --skip-gpu) SKIP_GPU=1 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> installing python deps"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

: "${FLUX2_MODEL_ID:=black-forest-labs/FLUX.2-dev}"
echo "==> FLUX2_MODEL_ID = ${FLUX2_MODEL_ID}"

if [[ "$SKIP_GPU" -eq 1 ]]; then
  echo "==> --skip-gpu: running CPU mock smoke test"
  python run.py --name _setup_smoke --method misc --generator mock --critic mock \
      --limit 2 --no_metrics
  python aggregate.py summary runs/_setup_smoke
  echo "==> CPU smoke OK"
  exit 0
fi

echo "==> self-check: import Flux2Pipeline and inspect caption_upsample support"
python - <<'PY'
import inspect
try:
    from diffusers import Flux2Pipeline
except Exception as e:
    raise SystemExit(f"Flux2Pipeline import failed (need diffusers>=0.36, py>=3.10): {e}")
sig = inspect.signature(Flux2Pipeline.__call__)
print("Flux2Pipeline OK. __call__ params:", list(sig.parameters))
print("caption_upsample supported:", "caption_upsample_temperature" in sig.parameters)
PY

if [[ "$DOWNLOAD" -eq 1 ]]; then
  echo "==> pre-downloading ${FLUX2_MODEL_ID}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${FLUX2_MODEL_ID}")
print("downloaded")
PY
fi

echo "==> setup done. Next: bash run_all.sh  (or run.py for a single config)"
