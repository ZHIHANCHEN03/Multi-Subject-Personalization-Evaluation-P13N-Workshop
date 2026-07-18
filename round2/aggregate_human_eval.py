"""Aggregate human-eval ballots into per-comparison win-rate + 95% CI.

Takes one or more filled ballot CSVs (each = one annotator; column
'annotator_choice(LEFT/RIGHT)') plus key.json, resolves LEFT/RIGHT to
ours/other via the hidden key, majority-votes across annotators per pair, and
reports ours' win-rate vs each comparison method with a bootstrap 95% CI.

Usage:
  python aggregate_human_eval.py --key round2/human_eval/key.json \
      --ballots ann1.csv ann2.csv ann3.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict


def boot_ci(bools, n=10000, seed=0):
    if not bools:
        return (None, None, None)
    rng = random.Random(seed)
    k = len(bools)
    m = sum(bools) / k
    stats = []
    for _ in range(n):
        s = [bools[rng.randrange(k)] for _ in range(k)]
        stats.append(sum(s) / k)
    stats.sort()
    return (round(m, 4), round(stats[int(0.025 * n)], 4), round(stats[int(0.975 * n)], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--ballots", nargs="+", required=True)
    args = ap.parse_args()

    key = json.load(open(args.key))
    # collect votes: pair_id -> list of "ours"/"other"
    votes = defaultdict(list)
    for bpath in args.ballots:
        for row in csv.DictReader(open(bpath)):
            pid = row["pair_id"].strip()
            choice = row.get("annotator_choice(LEFT/RIGHT)", "").strip().upper()
            if pid not in key or choice not in ("LEFT", "RIGHT"):
                continue
            ours_side = key[pid]["ours_side"]
            votes[pid].append("ours" if choice == ours_side else "other")

    # majority vote per pair, grouped by comparison method
    by_method = defaultdict(list)  # other_method -> list[bool ours_won]
    ties = 0
    for pid, vs in votes.items():
        other = key[pid]["other"]
        n_ours = vs.count("ours")
        n_other = vs.count("other")
        if n_ours == n_other:
            ties += 1
            continue
        by_method[other].append(n_ours > n_other)

    print(f"pairs with votes: {len(votes)}  ties(excluded): {ties}\n")
    for other, bools in sorted(by_method.items()):
        m, lo, hi = boot_ci(bools)
        sig = "  *sig (CI>50%)*" if (lo is not None and lo > 0.5) else ""
        print(f"ours vs {other:12s} win-rate={m}  95%CI=[{lo}, {hi}]  (n={len(bools)}){sig}")


if __name__ == "__main__":
    main()
