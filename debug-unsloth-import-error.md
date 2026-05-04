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
