"""Ad-hoc paired comparison of methods on the Round-1 results dir.

Usage: python peek_compare.py [results_dir] [main_method]
Reads <dir>/<method>/records.jsonl for known methods and prints, per entity
count, mean SCR/DINO and paired SCR win-rates of the main method vs others.
"""
import json
import os
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
MAIN = sys.argv[2] if len(sys.argv) > 2 else "ours_v2"
METHODS = [MAIN, "ours", "one_shot", "best_of_n", "umo", "freegraftor"]


def load(m):
    p = os.path.join(BASE, m, "records.jsonl")
    d = {}
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["task_id"]] = r
    return d


data = {m: load(m) for m in METHODS}
main = data[MAIN]
if not main:
    print(f"no records for main method {MAIN}")
    sys.exit(0)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


for n in (4, 2):
    keys = [k for k in main if main[k].get("num_subjects") == n]
    if not keys:
        continue
    print(f"\n=== {n}-entity  n={len(keys)} (by {MAIN} completed) ===")
    for m in METHODS:
        d = data[m]
        if not d:
            continue
        scr = mean([d.get(k, {}).get("scr") for k in keys])
        dino = mean([d.get(k, {}).get("dino_mean") for k in keys])
        print(f"  {m:11s} SCR={scr}  DINO={dino}")
    for m in METHODS:
        if m == MAIN or not data[m]:
            continue
        w = t = 0
        for k in keys:
            b = data[m].get(k, {}).get("scr")
            a = main[k].get("scr")
            if a is None or b is None:
                continue
            t += 1
            w += (a < b)
        print(f"  {MAIN}<{m} SCR winrate: {w}/{t}")
