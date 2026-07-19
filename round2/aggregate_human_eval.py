"""Aggregate human-eval votes (JSON from the frontend) into win-rate + 95% CI + Fleiss' kappa.

Each labeler exports a `votes_<labeler>.json` from index.html with shape:
  { labeler, n_pairs, n_voted, votes: { pair_id: {q1:"LEFT"/"RIGHT", q2:..., comparison, pair_id} } }

This script takes several such JSONs + the hidden key.json (pair_id -> ours_side /
other / task_id) and, per comparison method and per question (Q1 identity, Q2
overall), reports:
  - ours' win-rate (share of pairs where the labeler picked ours' side)
  - bootstrap 95% CI on the win-rate
  - Fleiss' kappa across labelers (agreement)

Usage:
  python aggregate_human_eval.py --key human_eval/key.json \
      --votes votes_AB.json votes_CD.json votes_EF.json
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict


def boot_ci(picks, n=10000, seed=0):
    """picks: list of bool (True = ours won). Returns (mean, lo, hi)."""
    if not picks:
        return (None, None, None)
    rng = random.Random(seed)
    k = len(picks)
    m = sum(picks) / k
    stats = []
    for _ in range(n):
        s = sum(picks[rng.randrange(k)] for _ in range(k)) / k
        stats.append(s)
    stats.sort()
    return (round(m, 4), round(stats[int(0.025 * n)], 4), round(stats[int(0.975 * n)], 4))


def fleiss_kappa(table):
    """table: list of rows, each row = list of counts per category (n labelers).
    Returns Fleiss' kappa across subjects (rows) and categories (columns)."""
    n_rows = len(table)
    if n_rows == 0:
        return None
    n_raters = sum(table[0])
    if n_raters <= 1:
        return None
    # category totals
    cats = len(table[0])
    col_totals = [0] * cats
    for row in table:
        for j in range(cats):
            col_totals[j] += row[j]
    N = n_rows
    p_j = [c / (N * n_raters) for c in col_totals]
    P_i = []
    for row in table:
        s = sum(c * c for c in row)
        P_i.append((s - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 0)
    P_bar = sum(P_i) / N
    Pe = sum(p * p for p in p_j)
    if Pe == 1:
        return None
    return round((P_bar - Pe) / (1 - Pe), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--votes", nargs="+", required=True, help="votes_<labeler>.json files")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    key = json.load(open(args.key))
    # load labelers
    labeler_votes = []  # list of {pair_id: {q1,q2}}
    labeler_names = []
    for vf in args.votes:
        d = json.load(open(vf))
        labeler_names.append(d.get("labeler", "?"))
        labeler_votes.append(d.get("votes", {}))

    n_labelers = len(labeler_votes)
    # per comparison, per question: list of (pair_id, [bool ours_won per labeler])
    by_cmp = defaultdict(lambda: defaultdict(lambda: {"q1": [], "q2": []}))
    all_pids = set()
    for lv in labeler_votes:
        all_pids.update(lv.keys())
    for pid in all_pids:
        if pid not in key:
            continue
        ours_side = key[pid]["ours_side"]
        other = key[pid]["other"]
        for q in ("q1", "q2"):
            picks = []
            for lv in labeler_votes:
                rec = lv.get(pid)
                if rec and rec.get(q) in ("LEFT", "RIGHT"):
                    picks.append(rec[q] == ours_side)
                else:
                    picks.append(None)
            by_cmp[other][pid][q] = picks

    summary = {}
    print(f"labelers: {labeler_names}  (n={n_labelers})")
    print(f"pairs in key: {len(key)}\n")
    for cmp_name in sorted(by_cmp):
        pids = by_cmp[cmp_name]
        print(f"=== ours vs {cmp_name}  (n_pairs={len(pids)}) ===")
        summary[cmp_name] = {}
        for q in ("q1", "q2"):
            # majority vote per pair (ignore pairs with <1 vote)
            maj_picks = []
            kappa_rows = []
            for pid, rec in pids.items():
                picks = rec[q]
                voted = [p for p in picks if p is not None]
                if not voted:
                    continue
                # majority: ours wins if strictly more True than False
                t = sum(1 for x in voted if x)
                f = sum(1 for x in voted if not x)
                if t == f:
                    continue  # tie -> exclude from win-rate
                maj_picks.append(t > f)
                # kappa row: [n_ours, n_other] (labelers who voted)
                kappa_rows.append([t, f])
            m, lo, hi = boot_ci(maj_picks)
            sig = "  *sig (CI>50%)*" if (lo is not None and lo > 0.5) else ""
            k = fleiss_kappa(kappa_rows) if kappa_rows else None
            qlabel = "Q1(identity)" if q == "q1" else "Q2(overall)"
            print(f"  {qlabel}: win-rate={m}  95%CI=[{lo}, {hi}]  (n={len(maj_picks)})  Fleiss kappa={k}{sig}")
            summary[cmp_name][q] = {"win_rate": m, "ci": [lo, hi], "n": len(maj_picks), "fleiss_kappa": k}
        print()

    if args.out:
        json.dump({"labelers": labeler_names, "summary": summary}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
