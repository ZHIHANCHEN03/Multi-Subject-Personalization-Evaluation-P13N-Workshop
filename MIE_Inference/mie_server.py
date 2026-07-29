"""Persistent MIE scoring server: JSON lines in on stdin, JSON lines out on stdout.

MIE needs Unsloth + a Qwen vision stack whose torch/transformers pins conflict with
OmniGen2 and FLUX.2, so it cannot share a process with the generators. This server
runs inside the MIE venv and is driven over pipes from whatever venv the caller uses.

Protocol (one JSON object per line, newline-delimited):

    server -> client   MIE_READY\t{"ok":true,"report":{...}}
    client -> server   {"image_path":..., "ref_paths":[...], "prompt":...}
    server -> client   MIE_RESULT\t{"ok":true,"total":..,"existence":..,
                                    "appearance":..,"interaction":..}
    client -> server   {"command":"shutdown"}
    server -> client   MIE_RESULT\t{"ok":true,"shutdown":true}

Every failure is reported as a line with "ok": false rather than by dying, so a
single bad image cannot take down a long scoring run. Wire-compatible with
`submission/round1/mie_server.py`, so `MieSubprocessCritic` can point here unchanged.

    python mie_server.py --checkpoint /path/to/...-lora_layer-best
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from mie_loader import load_runtime, score

READY_PREFIX = "MIE_READY\t"
RESULT_PREFIX = "MIE_RESULT\t"


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir (defaults to $MIE_CKPT)")
    args = ap.parse_args()

    try:
        runtime = load_runtime(args.checkpoint)
    except Exception as exc:
        emit(READY_PREFIX, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()})
        raise

    emit(READY_PREFIX, {"ok": True, "report": runtime["report"]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit(RESULT_PREFIX, {"ok": True, "shutdown": True})
                break
            result = score(
                runtime,
                image_path=request["image_path"],
                ref_paths=request["ref_paths"],
                prompt=request["prompt"],
            )
            emit(RESULT_PREFIX, {"ok": True, **result})
        except Exception as exc:
            emit(RESULT_PREFIX, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
