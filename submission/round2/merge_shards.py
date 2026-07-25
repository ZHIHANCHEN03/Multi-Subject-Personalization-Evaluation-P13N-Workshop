"""Merge per-shard results into a single results dir (concatenate records.jsonl).

Handles seed-namespaced method dirs produced by run_shard.sh, e.g.
`one_shot_s0`, `best_of_n_s0`, `ours_v2_s0`, `umo_s0`, `..._s1`, `..._s2`.
Each (method, seed) pair is merged into `<out>/<method>_s<seed>/records.jsonl`.
"""
import argparse
import glob
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--shard_glob", required=True, help="e.g. 'results_r2/shard_*'")
ap.add_argument("--out", required=True, help="merged results dir")
ap.add_argument(
    "--methods",
    nargs="+",
    default=["one_shot", "best_of_n", "ours_v2", "umo"],
    help="base method names (without seed suffix)",
)
ap.add_argument(
    "--seeds",
    nargs="+",
    default=None,
    help="seeds to merge (e.g. 0 1 2). Default: auto-discover all <method>_s* dirs.",
)
args = ap.parse_args()


def discover_seeds(shards, methods):
    seeds = set()
    for shard in shards:
        if not os.path.isdir(shard):
            continue
        for name in os.listdir(shard):
            for m in methods:
                if name.startswith(m + "_s"):
                    tail = name[len(m) + 2:]  # skip "<m>_s"
                    if tail.isdigit():
                        seeds.add(int(tail))
    return sorted(seeds)


shards = sorted(glob.glob(args.shard_glob))
seeds = args.seeds
if seeds is not None:
    seeds = [int(s) for s in seeds]
else:
    seeds = discover_seeds(shards, args.methods)
if not seeds:
    seeds = [0]

os.makedirs(args.out, exist_ok=True)
total_merged = 0
for m in args.methods:
    for s in seeds:
        target = f"{m}_s{s}"
        seen, rows = set(), []
        for shard in shards:
            p = os.path.join(shard, target, "records.jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                tid = (
                    line.split('"task_id":', 1)[1].split(",", 1)[0]
                    if '"task_id"' in line
                    else line
                )
                if tid in seen:
                    continue
                seen.add(tid)
                rows.append(line)
        if rows:
            out_dir = os.path.join(args.out, target)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
                f.write("\n".join(rows) + "\n")
            print(f"{target}: merged {len(rows)} records -> {out_dir}/records.jsonl")
            total_merged += 1
        else:
            print(f"{target}: no shard records found")

print(f"[merge] total method-seed groups merged: {total_merged}")
