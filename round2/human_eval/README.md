# Multi-Subject Human Eval — Blind A/B Package

This package lets 3 labelers blindly compare `ours` against the retrained SOTA
(`umo`) and against compute-matched best-of-N (`best_of_n`) on hard 4-subject
interaction prompts, then export their votes as JSON for aggregation.

## For the labeler (no install, works offline)

1. Unzip this folder.
2. Open `index.html` in any browser (Chrome/Firefox/Safari). It works offline —
   no server, no internet needed.
3. Enter your labeler ID (e.g. your initials) and click **Start labeling**.
4. For each pair you will see:
   - the **prompt** that was given to the generator,
   - the **reference subjects** (the images identities should be judged against),
   - a **LEFT | RIGHT** composite of two generated images (which side is ours
     is randomized and hidden).
5. Answer two forced-binary questions (no ties):
   - **Q1 (identity):** Which side better preserves the identity of ALL referenced subjects?
   - **Q2 (overall):** Which side has better overall image quality?
6. Use **Next →** to advance (disabled until both Q1 and Q2 are answered).
   Progress is saved in the browser's localStorage, so you can close and resume.
7. When done (or any time), click **Export votes JSON** → downloads `votes_<yourID>.json`.
8. Send that file back to the organizer.

## For the organizer

### 1. Generate the package (on the server, from `/workspace/misc/round2`)

```
python export_human_eval.py \
    --results results_r2 --shards 0 1 2 3 \
    --main ours_v2_s0 --vs umo_s0 best_of_n_s0 \
    --entities 4 --per 100 --seed 0 --out human_eval
```

This writes:
- `human_eval/pairs/<pair_id>.png` — composite LEFT|RIGHT blind image (512px)
- `human_eval/pairs/<pair_id>_refs.png` — reference subjects strip
- `human_eval/manifest.js` — pair list loaded by `index.html` (blinded)
- `human_eval/key.json` — **hidden** mapping pair_id → which side is ours (for unblinding)
- `human_eval/sample.jsonl` — fixed sampled task_ids (anti-p-hacking; attach to supplementary)
- `human_eval/ballot.csv` — legacy blank ballot (CSV labeling, if no browser)

### 2. Assemble the zip

Copy `index.html` + `aggregate_human_eval.py` + `README.md` into `human_eval/`,
then zip the whole `human_eval/` folder. Distribute to 3 labelers.

```
cp index.html aggregate_human_eval.py README.md human_eval/
zip -r human_eval.zip human_eval/
```

### 3. Collect votes & aggregate

When 3 labelers return `votes_AB.json`, `votes_CD.json`, `votes_EF.json`:

```
python aggregate_human_eval.py \
    --key human_eval/key.json \
    --votes votes_AB.json votes_CD.json votes_EF.json \
    --out human_eval_summary.json
```

Outputs, per comparison and per question:
- ours' **win-rate** (share of pairs where ours' side was picked, majority vote, ties excluded)
- **bootstrap 95% CI** on the win-rate (10k resamples)
- **Fleiss' kappa** across the 3 labelers (inter-annotator agreement)
- significance flag if CI lower bound > 50%

### Anti-p-hacking

The sampled task_ids are fixed by `--seed 0` and recorded in `sample.jsonl`
**before** any labeler sees a vote. Include `sample.jsonl` in the paper
supplementary so reviewers can verify the sampling was pre-registered.

## Design decisions (defaults, adjustable)

- **Comparisons**: `ours_v2_s0` vs `umo_s0` + `ours_v2_s0` vs `best_of_n_s0` (the two core claims).
- **Slice**: `num_subjects=4` (hard_4, the pre-declared main battlefield where the claim lives).
- **Per comparison**: 100 pairs → 200 pairs total.
- **Questions**: 2 (identity + overall), forced binary (no tie).
- **Labelers**: 3, majority vote, Fleiss' κ reported.
- **Seed**: 0 (fixed). Change `--seed` only for a new, separately-pre-registered sample.
