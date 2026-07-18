"""Round-2 statistical analysis: mean +/- 95% CI and paired significance.

Reads <results>/<method>/records.jsonl (or <results>/<method>_s<seed>/records.jsonl
when --seeds is given) and, per entity-count slice, reports for each method:
  - mean SCR / DINO with bootstrap 95% CI
And for the main method vs each other method:
  - paired mean difference with bootstrap 95% CI + two-sided p-value
  - paired win-rate

When --seeds is given, each method's per-task metric is first AVERAGED across
the listed seeds (task-level mean), then bootstrap is run over tasks. This is
the standard subject-level bootstrap for multi-seed runs (P0 for AAAI).

No GPU needed. Metrics: SCR (lower=better), DINO identity (higher=better).

Usage:
  # single-seed (legacy)
  python analyze.py --results /path/to/merged --main ours_v2 \
      --others umo best_of_n one_shot --entities 4 2 --metric scr

  # multi-seed (AAAI P0)
  python analyze.py --results /path/to/merged --main ours_v2 \
      --others umo best_of_n one_shot --entities 4 2 --metric scr --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
from statistics import mean


def load_seed(results_dir: str, method: str, seed: int | None) -> dict:
    """Load records for a method. If seed is None, read <method>/; else <method>_s<seed>/."""
    sub = method if seed is None else f"{method}_s{seed}"
    path = os.path.join(results_dir, sub, "records.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["task_id"]] = r
    return out


def load(results_dir: str, method: str, seeds: list[int] | None) -> dict:
    """Return task_id -> record dict. With multiple seeds, average the metric
    fields across seeds per task (kept as 'scr'/'dino_mean' on the merged record)."""
    if not seeds:
        return load_seed(results_dir, method, None)
    per_seed = [load_seed(results_dir, method, s) for s in seeds]
    # union of task_ids that appear in ANY seed (missing seed -> skip in avg)
    all_ids = set()
    for d in per_seed:
        all_ids.update(d.keys())
    merged = {}
    for tid in all_ids:
        recs = [d[tid] for d in per_seed if tid in d]
        if not recs:
            continue
        base = dict(recs[0])
        for metric in ("scr", "dino_mean"):
            vals = [r.get(metric) for r in recs if r.get(metric) is not None]
            if vals:
                base[metric] = mean(vals)
        # also average per-subject dino_sims if present
        if all("dino_sims" in r for r in recs):
            sims_len = min(len(r["dino_sims"]) for r in recs)
            base["dino_sims"] = [
                mean(r["dino_sims"][i] for r in recs if i < len(r["dino_sims"]))
                for i in range(sims_len)
            ]
        base["n_seeds"] = len(recs)
        merged[tid] = base
    return merged


def boot_ci(values, n=10000, alpha=0.05, seed=0):
    if not values:
        return (None, None, None)
    rng = random.Random(seed)
    m = mean(values)
    stats = []
    k = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        stats.append(mean(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n)]
    hi = stats[int((1 - alpha / 2) * n)]
    return (round(m, 4), round(lo, 4), round(hi, 4))


def paired_diff(main_vals, other_vals, n=10000, seed=0):
    """Bootstrap CI + two-sided p for mean(main - other) on paired tasks."""
    diffs = [a - b for a, b in zip(main_vals, other_vals)]
    if not diffs:
        return None
    rng = random.Random(seed)
    k = len(diffs)
    md = mean(diffs)
    boot = []
    for _ in range(n):
        s = [diffs[rng.randrange(k)] for _ in range(k)]
        boot.append(mean(s))
    boot.sort()
    lo = boot[int(0.025 * n)]
    hi = boot[int(0.975 * n)]
    prop_le0 = sum(1 for x in boot if x <= 0) / n
    p = 2 * min(prop_le0, 1 - prop_le0)
    return {"mean_diff": round(md, 4), "ci": (round(lo, 4), round(hi, 4)),
            "p": round(p, 4), "n": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--main", default="ours_v2")
    ap.add_argument("--others", nargs="+", default=["umo", "best_of_n", "one_shot", "freegraftor"])
    ap.add_argument("--entities", nargs="+", type=int, default=[4, 2])
    ap.add_argument("--metric", default="scr", choices=["scr", "dino_mean"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seeds to average per task (e.g. 0 1 2). Omit for single-seed layout.")
    ap.add_argument("--out", default=None, help="optional json dump")
    args = ap.parse_args()

    lower_better = args.metric == "scr"
    methods = [args.main] + [m for m in args.others if m != args.main]
    data = {m: load(args.results, m, args.seeds) for m in methods}
    main = data[args.main]
    if not main:
        print(f"no records for main method {args.main} in {args.results}")
        return

    seed_tag = f" (seeds={args.seeds}, per-task averaged)" if args.seeds else " (single seed)"
    report = {"metric": args.metric, "main": args.main, "seeds": args.seeds, "slices": {}}
    for n in args.entities:
        keys = [k for k in main if main[k].get("num_subjects") == n]
        print(f"\n================  {n}-entity  (n={len(keys)})  metric={args.metric}{seed_tag}  ================")
        slice_rep = {"n": len(keys), "means": {}, "paired_vs_main": {}}
        for m in methods:
            vals = [data[m][k].get(args.metric) for k in keys if k in data[m]]
            vals = [v for v in vals if v is not None]
            mci = boot_ci(vals)
            slice_rep["means"][m] = {"mean": mci[0], "ci95": [mci[1], mci[2]], "n": len(vals)}
            print(f"  {m:12s} mean={mci[0]}  95%CI=[{mci[1]}, {mci[2]}]  (n={len(vals)})")
        print(f"  -- paired: {args.main} vs others (diff = main - other; "
              f"{'negative' if lower_better else 'positive'} favors main) --")
        for m in methods:
            if m == args.main:
                continue
            pk = [k for k in keys if k in data[m]
                  and main[k].get(args.metric) is not None
                  and data[m][k].get(args.metric) is not None]
            mv = [main[k][args.metric] for k in pk]
            ov = [data[m][k][args.metric] for k in pk]
            pd = paired_diff(mv, ov)
            if pd is None:
                continue
            wins = sum(1 for a, b in zip(mv, ov) if ((a < b) if lower_better else (a > b)))
            sig = ""
            if pd["ci"][0] > 0 or pd["ci"][1] < 0:
                sig = "  *SIGNIFICANT(CI excludes 0)*"
            favor = (pd["mean_diff"] < 0) if lower_better else (pd["mean_diff"] > 0)
            slice_rep["paired_vs_main"][m] = {**pd, "winrate": f"{wins}/{pd['n']}",
                                              "main_better": bool(favor)}
            print(f"    vs {m:12s} diff={pd['mean_diff']:+.4f} CI={pd['ci']} "
                  f"p={pd['p']} winrate={wins}/{pd['n']} "
                  f"{'main_better' if favor else 'main_worse'}{sig}")
        report["slices"][str(n)] = slice_rep

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
