# MISC — MIE-guided Inference-time Self-Correction (code)

Implements the training-free paradigm in `../idea.md`: a frozen structured
verifier (MIE) drives a frozen generator (FLUX.2) through an inference-time
self-correction loop. Nothing is trained.

## What you plug in (not shipped in this repo)

| Asset | How to wire it | If missing |
|---|---|---|
| **FLUX.2** | `FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-4B` (or `FLUX.2-dev`) | use `--generator mock` |
| **MIE** | ready-made adapter `mie_adapter.py`: set `MISC_CRITIC=mie_checkpoint MIE_ADAPTER=mie_adapter MIE_REPO=.../Model_Training MIE_CKPT=.../-best`. Returns `total`=MIE preference score (unbounded, comparison-only, higher=better) + `existence/appearance/interaction`=sigmoid logits ∈[0,1] (higher=better) | use `--critic mock` (or `vlm_judge` with an API) |
| **Data (MIB-Gold)** | JSONL, `MISC_DATA=/path/mib_gold.jsonl` (schema in `data.py`) | built-in mock tasks |
| **LLM API** | `LLM_API_BASE`, `LLM_API_KEY` (OpenAI-compatible) | `prompt_llm` action & `vlm_judge` disabled; rule/mock used |

## Files

| File | Role |
|---|---|
| `config.py` | all hyperparameters (idea.md §7) + env wiring |
| `data.py` | MIB-Gold loader (+ mock tasks) |
| `critic.py` | MIE critic: `mie_checkpoint` / `vlm_judge` / `mock` |
| `generator.py` | FLUX.2 wrapper (multi-ref, caption_upsample, quantization) + mock |
| `actions.py` | typed actions: prompt rewrite (rule/LLM) + reference reweight (P2) |
| `pipeline.py` | MISC loop + `best_of_n` / `one_shot` / `caption_upsample` baselines |
| `metrics.py` | independent metrics: CLIP-I, DINO, CLIP-text (semantic guard) |
| `run.py` | run one config over the dataset -> `records.jsonl` |
| `aggregate.py` | per-run summary + paired win-rate with Wilson 95% CI |
| `setup.sh` / `run_all.sh` | env setup + full experiment matrix |

## Quick start

CPU smoke (no GPU / weights / data / MIE / LLM):
```bash
bash setup.sh --skip-gpu
# or directly:
python run.py --name smoke --method misc --generator mock --critic mock --limit 4 --no_metrics
python aggregate.py summary runs/smoke
```

Real run on A100/H100:
```bash
bash setup.sh --download
export FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-4B
export MISC_CRITIC=mie_checkpoint MIE_ADAPTER=my_mie.adapter
export MISC_DATA=/path/mib_gold.jsonl
export LLM_API_BASE=... LLM_API_KEY=...
LIMIT="--limit 500" bash run_all.sh
python aggregate.py winrate runs/misc runs/bon_scalar --metric final_total   # load-bearing
python aggregate.py winrate runs/misc runs/one_shot   --metric clip_i        # independent
```

## Notes

- **Compute alignment**: every method is capped by generation budget `B` (number
  of FLUX.2 calls). MISC uses `N` init + up to `K` correction steps; baselines
  spend the same `B`. This is what makes "structured vs scalar" a fair test.
- **No-regression**: a correction step is accepted only if the MIE total
  strictly improves and no non-target dimension drops beyond `DELTA`. The output
  is the best state over the whole trajectory (idea.md properties 1–2).
- **FLUX.2 [dev] 32B memory**: on <80GB or A100, set `FLUX2_QUANTIZE=nf4`.
