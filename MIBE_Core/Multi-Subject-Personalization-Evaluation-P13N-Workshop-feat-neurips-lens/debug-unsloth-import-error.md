# Debug Session: unsloth-import-error [OPEN]

## Symptom

- Running `Evaluation_Pipeline/run_export_lens_scores.sh` fails before inference starts.
- Observed error:
  - `ImportError: cannot import name '_unsloth_get_mm_token_id' from 'unsloth_zoo.rl_replacements'`

## Scope

- Affects the `Evaluation_Pipeline/export_lens_scores.py` entrypoint.
- Failure occurs during Python import / environment bootstrap, before model scoring.

## Initial Hypotheses

1. `unsloth` and `unsloth_zoo` versions are incompatible on the server.
2. `export_lens_scores.py` import order triggers an Unsloth patching path that fails in this environment.
3. The server Python environment differs from the one used during training, so runtime dependencies are not aligned.
4. A stale global site-packages install is shadowing the expected versions for `unsloth` / `unsloth_zoo`.
5. The current script can avoid the failing path by importing `unsloth` first and/or by running in a dedicated venv with pinned packages.

## Evidence To Collect

- Installed versions for `unsloth`, `unsloth_zoo`, `transformers`, `peft`, and `torch`
- Exact import behavior for `import unsloth`
- Whether import order alone changes the failure mode
- Whether the current shell is using system Python or a project venv

## Status

- Waiting to collect environment evidence before applying a minimal fix.

## Evidence Collected

- Runtime selected Python:
  - `/usr/bin/python3`
- Runtime version:
  - `3.11.10`
- Observed package versions:
  - `unsloth = IMPORT_FAIL`
  - `unsloth_zoo = 2026.4.9`
  - `transformers = 5.5.0`
  - `peft = 0.19.1`
  - `torch = 2.5.1+cu121`
- Exact failure:
  - `ImportError: cannot import name '_unsloth_get_mm_token_id' from 'unsloth_zoo.rl_replacements'`

## Hypothesis Evaluation

1. `unsloth` and `unsloth_zoo` versions are incompatible on the server.
   - Status: CONFIRMED
   - Reason: import fails inside `unsloth` while resolving a symbol from `unsloth_zoo`.

2. `export_lens_scores.py` import order triggers an Unsloth patching path that fails in this environment.
   - Status: PARTIALLY MITIGATED
   - Reason: import order was corrected, but failure still happens, so import order is not the root cause.

3. The server Python environment differs from the one used during training, so runtime dependencies are not aligned.
   - Status: CONFIRMED
   - Reason: script is running under `/usr/bin/python3` instead of the project training venv.

4. A stale global site-packages install is shadowing the expected versions for `unsloth` / `unsloth_zoo`.
   - Status: VERY LIKELY
   - Reason: system Python is importing global packages from `/usr/local/lib/python3.11/dist-packages`.

5. The current script can avoid the failing path by importing `unsloth` first and/or by running in a dedicated venv with pinned packages.
   - Status: PARTIALLY CONFIRMED
   - Reason: importing `unsloth` first was not sufficient; using a dedicated venv remains the primary fix path.

## Root Cause

- The export pipeline is not running inside the intended project venv.
- The active global Python environment contains an incompatible `unsloth` / `unsloth_zoo` combination.

## Next Minimal Fix

1. Ensure `Model_Training/.venv-a100-unsloth` exists on the server.
2. Run the export pipeline with that venv Python explicitly.
3. If that venv does not exist or is also broken, reinstall pinned `unsloth` + `unsloth_zoo` inside the venv and retry.
