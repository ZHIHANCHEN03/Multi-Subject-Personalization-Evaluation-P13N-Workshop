# Round 2 — Main experiments, scaling, ablation, human eval

Round 2 runs the full-scale evaluation: 500 tasks × 3 seeds on OmniGen2 (4 methods),
plus a FLUX.2 scaling study (6/8 subjects) and an ablation of each MIDC component.
It reuses Round 1's environment, models, frozen calibration, and pipelines — Round 2
only adds orchestration and analysis scripts.

## Files
```
run_round2_main.sh     500-task main run: build manifest → reuse calibration → shard → one tmux per GPU
run_shard.sh           single-GPU single-shard run of 4 methods (one_shot/best_of_n/ours_v2/umo)
split_manifest.py      split manifest into K shards
merge_shards.py        merge per-shard records
analyze.py             mean + 95% CI + paired bootstrap significance (p / CI / win rate)
export_human_eval.py   export blind A/B pairs + ballot.csv + key.json
aggregate_human_eval.py aggregate human eval: win rate + 95% CI (majority vote across labelers)
run_ablation.sh        calibrated routing vs raw, +refset vs prompt-only
run_flux2_scaling.sh  FLUX.2 6/8-subject scaling
run_scaling.sh        budget-B sweep of ours vs best-of-N
score_precomputed.py  score pre-generated images from cross-system baselines (MOSAIC etc.) by task_id
select_hard_cases_6_8.py build 6/8-subject manifest (disjoint from prior splits)
b1_reanalysis.py      recompute gating-table numbers from FLUX.2 records (pooled over (task,seed) rows)
```

## How to run (on the server)

### Main 500-task run with multi-seed significance

`run_shard.sh` supports a `SEEDS` env var (comma-separated); output dirs are namespaced
by seed (`<method>_s<seed>/`) so runs don't overwrite. The submission requires ≥3 seeds
with 95% CI and paired significance, so we stage in two passes:

```bash
cd round2

# Pass 1: all 4 shards × seed 0 → first signal
NGPU=4 MIE_CKPT=<ckpt> SEEDS=0 bash run_round2_main.sh
# after all *_DONE, merge + analyze seed 0:
../.venvs/omni/bin/python merge_shards.py --shard_glob 'results_r2/shard_*' --out results_r2/merged --seeds 0
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --entities 4 2 --metric scr --seeds 0 \
    --out results_r2/analysis_scr_s0.json

# Pass 2: if seed 0 signal is OK, continue with seeds 1,2 (same shard paths, just change SEEDS):
SEEDS=1,2 bash run_shard.sh   # in each tmux; SHARD_MANIFEST/RESULTS_DIR unchanged
# after all done, 3-seed joint analysis (per-task mean across seeds, then bootstrap over tasks):
../.venvs/omni/bin/python merge_shards.py --shard_glob 'results_r2/shard_*' --out results_r2/merged --seeds 0 1 2
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --entities 4 2 --metric scr --seeds 0 1 2 \
    --out results_r2/analysis_scr_3seed.json
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --metric dino_mean --seeds 0 1 2 \
    --out results_r2/analysis_dino_3seed.json
```
A comparison is significant when the 95% CI excludes 0 (p<0.05). With `--seeds 0 1 2`,
each task's metric is first averaged across the 3 seeds (subject-level), then bootstrapped
over tasks — the standard multi-seed significance protocol.

### Human eval (no GPU needed; uses already-generated images)
```bash
../.venvs/omni/bin/python export_human_eval.py --results results_r2/merged \
    --main ours_v2 --vs umo best_of_n --entities 4 --per 80 --out human_eval
# send human_eval/pairs/*.png + ballot.csv to ≥3 labelers, each fills a ballot
../.venvs/omni/bin/python aggregate_human_eval.py --key human_eval/key.json \
    --ballots ann1.csv ann2.csv ann3.csv
```

### Ablation and scaling (GPU; small subset suffices)
```bash
DATA=results_r2/manifests/r2_hard.jsonl CALIBRATION=<frozen> MIE_CKPT=<ckpt> bash run_ablation.sh
DATA=results_r2/manifests/r2_hard.jsonl CALIBRATION=<frozen> MIE_CKPT=<ckpt> bash run_flux2_scaling.sh
../.venvs/omni/bin/python analyze.py --results results_ablation --main ours_full --others ours_rawroute ours_promptonly
```

### Cross-system baselines (MOSAIC etc.)
1. Clone each repo, download checkpoints, run inference on `results_r2/manifests/round2_full.jsonl`,
   saving each task's image as `<dir>/<task_id>.png`.
2. Score:
```bash
../.venvs/omni/bin/python score_precomputed.py --data results_r2/manifests/round2_full.jsonl \
    --images /path/to/mosaic_images --name mosaic --out_dir results_r2/merged
```

## Notes
- Fully training-free; MIE is used only as an in-loop verifier; final metrics use SCR/DINO + human eval.
- FLUX.2 6/8-subject scaling requires FLUX.2 to natively support ≥6 reference images; verify with `probe_flux2.sh` first.
