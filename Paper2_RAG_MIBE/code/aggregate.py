"""Aggregate run records into the paper's numbers.

  - per-run summary (mean final_total, per-dim, collateral damage, gen calls)
  - paired win-rate + 95% CI (Wilson) between two runs, aligned by task_id,
    using an independent metric (default: MIE final_total; pass --metric to use
    clip_i / dino / clip_t)
  - scaling helper: point summary per (N,K) run

Decision lines (idea.md section 8.4): a comparison "passes" when the win-rate
CI lower bound > 0.5.

Usage:
    python aggregate.py summary runs/misc_main
    python aggregate.py winrate runs/misc_main runs/bon_scalar --metric final_total
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import config


def _resolve(run_dir):
    """Accept an absolute/relative path OR a bare run name under WORK_DIR."""
    p = Path(run_dir)
    if (p / "records.jsonl").exists():
        return p
    alt = config.WORK_DIR / p.name
    if (alt / "records.jsonl").exists():
        return alt
    return p


def load_records(run_dir) -> dict:
    path = _resolve(run_dir) / "records.jsonl"
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["task_id"]] = r
    return out


def _get_metric(rec, metric):
    if metric in ("clip_i", "dino", "clip_t"):
        return (rec.get("independent") or {}).get(metric)
    return rec.get(metric)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, center - half, center + half)


def cmd_summary(args):
    recs = load_records(args.run)
    n = len(recs)
    keys = ["final_total", "init_total", "collateral_damage_rate", "gen_calls",
            "accepted_steps", "rejected_steps"]
    agg = {k: 0.0 for k in keys}
    dims = {"existence": 0.0, "appearance": 0.0, "interaction": 0.0}
    ind = {"clip_i": [], "dino": [], "clip_t": []}
    for r in recs.values():
        for k in keys:
            agg[k] += r.get(k, 0) or 0
        for d in dims:
            dims[d] += (r.get("final_dims") or {}).get(d, 0) or 0
        for m in ind:
            v = (r.get("independent") or {}).get(m)
            if v is not None:
                ind[m].append(v)
    print(f"run: {args.run}  (n={n})")
    for k in keys:
        print(f"  mean {k:24s}: {agg[k]/max(n,1):.4f}")
    for d in dims:
        print(f"  mean dim.{d:19s}: {dims[d]/max(n,1):.4f}")
    for m, vals in ind.items():
        if vals:
            print(f"  mean {m:24s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")


def cmd_winrate(args):
    a = load_records(args.run_a)
    b = load_records(args.run_b)
    common = sorted(set(a) & set(b))
    wins = losses = ties = usable = 0
    for tid in common:
        va, vb = _get_metric(a[tid], args.metric), _get_metric(b[tid], args.metric)
        if va is None or vb is None:
            continue
        usable += 1
        if va > vb + args.eps:
            wins += 1
        elif vb > va + args.eps:
            losses += 1
        else:
            ties += 1
    decisive = wins + losses
    p, lo, hi = wilson_ci(wins, decisive)
    print(f"A={args.run_a}  vs  B={args.run_b}   metric={args.metric}")
    print(f"  paired tasks usable: {usable}  (ties={ties}, decisive={decisive})")
    print(f"  A win-rate (excl. ties): {p:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    verdict = "PASS (CI lower bound > 0.5)" if lo > 0.5 else "not significant"
    print(f"  verdict: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary")
    s.add_argument("run")
    s.set_defaults(func=cmd_summary)

    w = sub.add_parser("winrate")
    w.add_argument("run_a")
    w.add_argument("run_b")
    w.add_argument("--metric", default="final_total")
    w.add_argument("--eps", type=float, default=1e-6)
    w.set_defaults(func=cmd_winrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
