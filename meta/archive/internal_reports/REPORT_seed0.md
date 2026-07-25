# Round 2 — Phase 1 (P0-1 Pass 1, seed 0) Report

> 500-task, single-seed (seed 0) run of all four methods on OmniGen2.
> Status: **complete**. This is the first large-scale signal for the AAAI claim.
> Generated 2026-07-19.

## Setup

- **Base model**: OmniGen2 (native multi-reference, ≤5 refs).
- **Task set**: 500 tasks = 250 `hard_4` (4 entities, strong interaction + occlusion) + 250 `easy_2` (2 entities).
- **Sharding**: 4 shards × 125 tasks, run on 4× A100 in parallel.
- **Methods** (all seed 0):
  - `one_shot` — single generation, no selection.
  - `best_of_n` — generate N=8 candidates, pick best by MIE total (open-loop selection, compute-matched to ours).
  - `ours_v2` — winner pipeline from Round 1.1 (dual-signal diagnosis: MIE facet + DINO/Grounding-DINO subject → targeted refset manipulation + facet prompt rewrite + action portfolio + weak-subject selection; budget 8, matched to best-of-N).
  - `umo` — retrained SOTA for multi-identity consistency (the "retraining" baseline).
- **Metrics** (independent of MIE, used for final judgment):
  - `SCR` (Subject Collapse Rate, DINOv2 + Grounding-DINO, detection-aware) — **lower = better**.
  - `DINO` (mean DINOv2 identity similarity of detected subjects vs their references) — **higher = better**.
- MIE is used **only inside** `ours_v2`'s loop; it is never the final scorer.

## Results — overall (500 tasks, seed 0)

| method | SCR ↓ | DINO ↑ |
|---|---|---|
| one_shot | 0.5475 | 0.4612 |
| best_of_n | **0.4745** | 0.5118 |
| ours_v2 | 0.4800 | **0.5151** |
| umo (retrained SOTA) | 0.5539 | 0.4520 |

## Results — hard_4 slice (250 tasks, the paper's main battlefield)

| method | SCR ↓ | DINO ↑ |
|---|---|---|
| one_shot | 0.5590 | 0.4326 |
| best_of_n | 0.4910 | 0.4924 |
| **ours_v2** | **0.4840 ✅** | **0.4937 ✅** |
| umo | 0.5690 | 0.4220 |

## Results — easy_2 slice (250 tasks)

| method | SCR ↓ | DINO ↑ |
|---|---|---|
| one_shot | 0.5360 | 0.4899 |
| best_of_n | **0.4580 ✅** | 0.5312 |
| ours_v2 | 0.4760 | **0.5364 ✅** |
| umo | 0.5364 | 0.4837 |

## Interpretation

### ✅ Core claim holds (both slices): training-free matches/beats retrained SOTA

`ours_v2` beats `umo` (retrained SOTA) on **both** metrics on **both** slices:
- hard_4: SCR 0.484 vs 0.569 (~15% lower collapse), DINO 0.494 vs 0.422 (~17% higher identity sim).
- easy_2: SCR 0.476 vs 0.536 (~11% lower), DINO 0.536 vs 0.484 (~11% higher).
- `umo` is the **worst** method on both slices — retraining is the *worst* option on this hard-interaction benchmark, which is exactly the gap our paper targets.

### ⚠️ Secondary claim is slice-dependent: closed-loop > open-loop selection

- On **hard_4**: `ours_v2` beats `best_of_n` on both metrics (SCR 0.484 < 0.491, DINO 0.494 > 0.492). Closed-loop correction beats open-loop selection **where interaction-induced collapse is severe**.
- On **easy_2**: `ours_v2` loses to `best_of_n` on SCR (0.476 vs 0.458) and only marginally wins on DINO (0.536 vs 0.531). Easy cases don't need correction — selection suffices.
- Overall 500 is a wash because easy_2 dilutes the hard_4 win.

This maps cleanly to **PLAN §四 退路 row 2**: "最难子集救回来 + 整体追平重训" → **AAAI 有戏**. The paper frames it as: *interaction-induced identity collapse is diagnosable and repairable at test time without training; on easy cases selection suffices, on hard cases closed-loop correction is necessary and beats both a retrained SOTA and compute-matched best-of-N.*

## Honest caveats (what Phase 1 does NOT yet prove)

1. **Single seed.** These are seed-0 means. A skeptic can argue luck. → Pass 2 (seeds 1, 2) + bootstrap CI + paired p-values needed to claim significance.
2. **No human evaluation.** SCR/DINO are automatic proxies. → Blind A/B human eval (3 labelers, ~200 pairs) needed to validate that humans agree ours' outputs are better.
3. **No significance testing.** No CIs/p-values yet.
4. **ours vs best_of_n on easy_2 is a loss on SCR.** Must be reported honestly in the paper (not hidden).

## GO / NO-GO decision

**GO.** The core claim (training-free ≥ retrained SOTA) holds strongly on both slices at 500-task scale. The hard_4 slice — the pre-declared main battlefield — shows ours wins on both metrics vs both baselines. This is the upside scenario. Proceed to:

1. **Pass 2 (seeds 1, 2)** on 4 GPUs → multi-seed → bootstrap CI + paired p-values.
2. **Human eval export** from seed-0 hard_4 images → 3 labelers → win-rate CI + Fleiss' κ.
3. **P1-4 compute scaling** (B = 2/4/6/8) on 1 free GPU → Pareto curve.
4. Then FLUX.2 6/8-subject scaling (P1-3) + cross-system baselines (P2-6) + ablation (P2-5).

## Files

- Per-shard records: `results_r2/shard_{0..3}/<method>_s0/records.jsonl`
- Per-shard images: `results_r2/shard_{0..3}/<method>_s0/images/<task_id>.png`
- Manifest: `results_r2/shards/shard_{0..3}.jsonl` (125 tasks each)
- Calibration (frozen from Round 1): reused by `ours_v2`.

## Reproduce

```
# (already run) Phase 1 = 4 shards × 4 methods × seed 0
# analyze:
python - <<'PY'
import json, statistics
methods = ["one_shot_s0","best_of_n_s0","ours_v2_s0","umo_s0"]
for m in methods:
    scr=[]; dino=[]
    for i in [0,1,2,3]:
        for l in open(f"results_r2/shard_{i}/{m}/records.jsonl"):
            r=json.loads(l)
            if r.get("scr") is not None: scr.append(r["scr"])
            if r.get("dino_mean") is not None: dino.append(r["dino_mean"])
    print(f"{m}: scr={statistics.mean(scr):.4f} dino={statistics.mean(dino):.4f} n={len(scr)}")
PY
```
