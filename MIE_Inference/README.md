# MIE_Inference

Standalone inference for the trained **MIE** decomposed evaluator (existence /
appearance / interaction). Replaces the loader that used to live inside
`submission/round1/mie_server.py`, which hard-coded `<repo>/MIBE_Core/...` and
broke when the repo was restructured into `submission/ companion/ meta/`.

Nothing here trains. It loads a finished checkpoint and returns scores.

## Files

| File | Purpose |
|---|---|
| `mie_loader.py` | `load_runtime()` + `score()`. All paths from env vars. Verifies weights actually applied. |
| `mie_server.py` | Persistent JSON-lines server, wire-compatible with the old one, so `MieSubprocessCritic` can point here unchanged. |
| `score_batch.py` | Score a whole manifest in one model load. Resumable. |
| `smoke_test.py` | Run this first on any new box or checkpoint. |

## Why a separate process

MIE needs Unsloth + a Qwen vision stack whose `torch` / `transformers` pins
conflict with OmniGen2 and FLUX.2. It cannot share a process with the
generators, hence the two-venv, pipe-driven design:

- `.venvs/omni` — OmniGen2 / FLUX.2 generation + DINOv2 scoring (parent)
- `.venvs/mie` — Unsloth + Qwen3.5 (this code, child)

Two consequences that are easy to get wrong:

1. **`import unsloth` must precede `transformers`.** Unsloth monkey-patches
   transformers on import; importing transformers first silently gives you an
   unpatched model.
2. **stdout is a protocol channel.** Unsloth and transformers both print banners,
   so every noisy section is wrapped in `contextlib.redirect_stdout(sys.stderr)`.

## Environment

| Variable | Meaning |
|---|---|
| `MIE_CKPT` | checkpoint directory (required) |
| `MIE_CODE` | directory holding the importable `mie` package (MIBE `Model_Training_Paper_Coding`) |
| `HF_HOME` | Hugging Face cache holding the base backbone |
| `MIE_DEVICE` | `cuda` / `cpu` (default: cuda when available) |

`MIE_CODE` falls back to the in-repo `companion/MIBE_Core/.../Model_Training_Paper_Coding`,
then to the RunPod default, so on the project boxes it can usually be omitted.

## Checkpoint layout

`mie_*` are the current names; `lens_*` are the pre-rename names and are still
accepted (the trained checkpoints use them).

```
<MIE_CKPT>/
├── mie_config.json         or lens_config.json
├── mie_heads.pt            or lens_heads.pt
├── lora_adapter/           required when mode is lora / lora_layer
└── trainable_backbone.pt   required when mode is layer_only / partial / lora_layer / full
```

`mode` in the config drives what gets loaded:

| mode | LoRA adapter | trainable backbone |
|---|---|---|
| `head_only` | — | — |
| `lora` | ✓ | — |
| `layer_only`, `partial`, `full` | — | ✓ |
| `lora_layer` | ✓ | ✓ |

## Quick start

```bash
export HF_HOME=/workspace/misc/models/hf_cache
export MIE_CODE="/workspace/misc/MIBE_Core/Multi-Subject-Personalization-Evaluation-P13N-Workshop-feat-neurips-lens/Model_Training_Paper_Coding"
export MIE_CKPT=/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best

/workspace/misc/.venvs/mie/bin/python smoke_test.py --refs /workspace/misc/refs
```

Batch scoring:

```bash
/workspace/misc/.venvs/mie/bin/python score_batch.py \
    --manifest /workspace/misc/round2/results_r2/manifests/round2_full.jsonl \
    --images   /path/to/candidate/pngs \
    --refs_root /workspace/misc/refs \
    --out      scores.jsonl
```

As a server (from any venv):

```bash
/workspace/misc/.venvs/mie/bin/python mie_server.py --checkpoint "$MIE_CKPT"
# then write one JSON object per line to its stdin:
# {"image_path": "...", "ref_paths": ["...", "..."], "prompt": "..."}
```

## Scoring contract

Input to the model is a chat turn holding **the reference images first, the
candidate last**, then this text:

> You are evaluating a multi-subject personalization result. The first images are
> subject references. The last image is the generated candidate. Prompt: {prompt}

Output:

| Field | Source | Range |
|---|---|---|
| `total` | `score_head` (regression) | unbounded; higher is better |
| `existence`, `appearance`, `interaction` | `sigmoid(classification_head)` | 0–1 |

Reference order matters — it is the order the heads were trained on. Do not
shuffle refs between scoring runs you intend to compare.

## Verified checkpoint (2026-07-26)

`20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best`

```
base_model                unsloth/Qwen3.5-4B     mode  lora_layer
lora_params                     688
backbone_tensors_in_file         53
backbone_tensors_applied         53      <- all applied
backbone_unexpected_keys          0      <- no key-name mismatch
device                         cuda      (A100-SXM4-80GB)
```

Smoke test passes all four checks, including the discrimination check (matched vs
mismatched candidates differ on all four outputs; `total` differs by 0.145).

**`-best` and `-epoch1` are byte-identical** (MD5-verified on all four
artifacts): training stopped after one epoch at its 600-optimizer-step budget, so
`best == last == epoch1`. Note that "best" therefore carries no selection
information — there was no second candidate to choose between. Val monitor loss
was 0.1179.

The sibling run `20260503_044916` is the same backbone in `layer_only` mode (no
LoRA); the paper describes MIE as LoRA-tuned, which matches `lora_layer`.

## A note on `strict=False`

`trainable_backbone.pt` holds only the trainable subset of the backbone, so it
*must* be loaded with `strict=False`. That makes a total key-name mismatch look
exactly like a clean load. `mie_loader.py` therefore inspects the returned
`_IncompatibleKeys` and raises if nothing applied, and `smoke_test.py` additionally
checks that the model's output actually changes with its input. Report both counts
when quoting a successful load; "it loaded without error" is not evidence.
