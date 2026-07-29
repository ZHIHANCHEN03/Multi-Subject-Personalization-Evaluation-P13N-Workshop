"""Recompute every number in the paper's Tables 1-5 from the committed records.

Single source of truth for "do the numbers in main.tex match the data?". Reads
only files tracked in this repo (no server access, no GPU), recomputes each
table, and diffs against the values hard-coded in the paper.

    python3 verify_paper_numbers.py            # human-readable report
    python3 verify_paper_numbers.py --json out.json

Exit code is 1 if any recomputed value disagrees with the paper.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(HERE, "results_r2", "merged")
FLUX2 = os.path.join(HERE, "results_flux2")
ABL = os.path.join(HERE, "results_ablation")
HUMAN = os.path.join(HERE, "human_eval", "HUMAN_EVAL", "aggregate_result_3labeler.json")

SEEDS = (0, 1, 2)
TOL = 0.0006  # paper rounds to 3 decimals


def read_records(path):
    """Tolerant JSONL reader: skips blank lines and truncated write fragments."""
    if not os.path.exists(path):
        return []
    out, bad = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  ! {os.path.relpath(path, HERE)}: skipped {bad} malformed line(s)")
    return out


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def bootstrap_ci(vals, n_boot=10000, seed=0):
    """Percentile bootstrap CI of the mean."""
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def per_task_mean(dirs, metric, keep):
    """Average `metric` across seeds per task_id, for tasks passing `keep`."""
    acc = defaultdict(list)
    for d in dirs:
        for r in read_records(os.path.join(d, "records.jsonl")):
            if not keep(r) or r.get(metric) is None:
                continue
            acc[r["task_id"]].append(r[metric])
    return [mean(v) for v in acc.values()]


# --- paper values, transcribed from main.tex -------------------------------
PAPER = {
    "t1": {  # (method, slice, metric) -> value
        ("one_shot", "hard_4", "scr"): 0.536, ("one_shot", "easy_2", "scr"): 0.511,
        ("best_of_n", "hard_4", "scr"): 0.492, ("best_of_n", "easy_2", "scr"): 0.467,
        ("umo", "hard_4", "scr"): 0.531, ("umo", "easy_2", "scr"): 0.517,
        ("ours_v2", "hard_4", "scr"): 0.470, ("ours_v2", "easy_2", "scr"): 0.438,
        ("one_shot", "hard_4", "dino_mean"): 0.455, ("one_shot", "easy_2", "dino_mean"): 0.507,
        ("best_of_n", "hard_4", "dino_mean"): 0.492, ("best_of_n", "easy_2", "dino_mean"): 0.526,
        ("umo", "hard_4", "dino_mean"): 0.455, ("umo", "easy_2", "dino_mean"): 0.499,
        ("ours_v2", "hard_4", "dino_mean"): 0.509, ("ours_v2", "easy_2", "dino_mean"): 0.546,
    },
    "t2": {
        ("oneshot", 6, "scr"): 0.615, ("oneshot", 8, "scr"): 0.671,
        ("bon", 6, "scr"): 0.607, ("bon", 8, "scr"): 0.645,
        ("ours", 6, "scr"): 0.580, ("ours", 8, "scr"): 0.630,
        ("oneshot", 6, "dino_mean"): 0.393, ("oneshot", 8, "dino_mean"): 0.346,
        ("bon", 6, "dino_mean"): 0.408, ("bon", 8, "dino_mean"): 0.369,
        ("ours", 6, "dino_mean"): 0.428, ("ours", 8, "dino_mean"): 0.379,
    },
    "t3": {  # variant -> (SCR, delta% vs full)
        "ours_full": (0.475, None), "ours_rawroute": (0.524, 10.3),
        "ours_strictaccept": (0.497, 4.7), "ours_promptonly": (0.491, 3.4),
        "ours_noportfolio": (0.489, 2.9), "ours_nodual": (0.481, 1.3),
    },
    # generator calls: reported inline in Sec. 4.3 (no longer its own table)
    "t4": {"one_shot": 1.0, "umo": 1.0, "best_of_n": 8.0, "ours_v2": 4.6, "flux2_ours_8": 4.3},
    # Table 1 now also carries CLIP-I / CLIP-T (seed 0, n=500 over both slices).
    "clip": {"one_shot": (0.317, 0.656), "best_of_n": (0.323, 0.657),
             "umo": (0.316, 0.666), "ours_v2": (0.321, 0.662)},
    "t5": {  # (baseline, question) -> (win_rate, n, kappa)
        ("umo_s0", "q1"): (0.846, 52, 0.32), ("umo_s0", "q4"): (0.589, 73, 0.15),
        ("best_of_n_s0", "q1"): (0.500, 32, 0.48), ("best_of_n_s0", "q4"): (0.523, 65, 0.17),
    },
}

failures = []


def check(label, got, want, tol=TOL):
    ok = want is None or (got == got and abs(got - want) <= tol)
    if not ok:
        failures.append(f"{label}: recomputed {got:.4f} vs paper {want:.4f}")
    return "OK " if ok else "MISMATCH"


def table1(report):
    print("\n=== Table 1: OmniGen2 main, 3 seeds ===")
    slices = {"hard_4": lambda r: r.get("num_subjects") == 4,
              "easy_2": lambda r: r.get("num_subjects") == 2}
    for metric in ("scr", "dino_mean"):
        print(f"\n  {metric}")
        for method in ("one_shot", "best_of_n", "umo", "ours_v2"):
            dirs = [os.path.join(R2, f"{method}_s{s}") for s in SEEDS]
            row = []
            for sl, keep in slices.items():
                vals = per_task_mean(dirs, metric, keep)
                m = mean(vals)
                lo, hi = bootstrap_ci(vals)
                want = PAPER["t1"][(method, sl, metric)]
                status = check(f"T1 {method}/{sl}/{metric}", m, want)
                report[f"t1.{method}.{sl}.{metric}"] = {
                    "mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                    "n": len(vals), "paper": want, "status": status}
                row.append(f"{sl}: {m:.3f} [{lo:.3f},{hi:.3f}] n={len(vals)} (paper {want:.3f}) {status}")
            print(f"    {method:12} " + "  |  ".join(row))


def table2(report):
    print("\n=== Table 2: FLUX.2-klein-9B scaling, 3 seeds ===")
    for metric in ("scr", "dino_mean"):
        print(f"\n  {metric}")
        for method in ("oneshot", "bon", "ours"):
            row = []
            for ent in (6, 8):
                dirs = [os.path.join(FLUX2, f"flux2_{ent}_{method}_s{s}") for s in SEEDS]
                vals = per_task_mean(dirs, metric, lambda r: True)
                m = mean(vals)
                lo, hi = bootstrap_ci(vals)
                want = PAPER["t2"][(method, ent, metric)]
                status = check(f"T2 {method}/{ent}ent/{metric}", m, want)
                report[f"t2.{method}.{ent}.{metric}"] = {
                    "mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                    "n": len(vals), "paper": want, "status": status}
                row.append(f"{ent}ent: {m:.3f} [{lo:.3f},{hi:.3f}] n={len(vals)} (paper {want:.3f}) {status}")
            print(f"    {method:12} " + "  |  ".join(row))


def table3(report):
    print("\n=== Table 3 (paper: Table 3): Ablation, hard 4-entity, 2 seeds ===")
    by_variant = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ABL, "*", "records.jsonl"))):
        variant = os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0]
        by_variant[variant] += [r["scr"] for r in read_records(p) if r.get("scr") is not None]
    full = mean(by_variant["ours_full"])
    for variant, vals in sorted(by_variant.items(), key=lambda kv: mean(kv[1])):
        m = mean(vals)
        delta = None if variant == "ours_full" else 100 * (m - full) / full
        want_scr, want_delta = PAPER["t3"][variant]
        s1 = check(f"T3 {variant} SCR", m, want_scr)
        s2 = "OK " if delta is None else check(f"T3 {variant} delta", delta, want_delta, tol=0.06)
        report[f"t3.{variant}"] = {"scr": round(m, 4), "delta_pct": None if delta is None else round(delta, 2),
                                   "n": len(vals), "paper_scr": want_scr, "status": s1}
        d = "  ---  " if delta is None else f"{delta:+6.1f}%"
        print(f"    {variant:20} {m:.4f} {d}  n={len(vals)} (paper {want_scr:.3f}) {s1} {s2}")


def clip_table(report):
    """CLIP-T / CLIP-I as printed in Table 1 (seed 0)."""
    print("\n=== Table 1, CLIP columns (seed 0) ===")
    for method, (want_t, want_i) in PAPER["clip"].items():
        path = os.path.join(HERE, "results_clip", f"clip_{method}_s0.jsonl")
        rows = [r for r in read_records(path) if "clip_t" in r]
        if not rows:
            print(f"    {method:12} no CLIP records at {path}")
            continue
        t = mean([r["clip_t"] for r in rows])
        i = mean([r["clip_i"] for r in rows])
        s1 = check(f"CLIP-T {method}", t, want_t)
        s2 = check(f"CLIP-I {method}", i, want_i)
        report[f"clip.{method}"] = {"clip_t": round(t, 4), "clip_i": round(i, 4),
                                    "n": len(rows), "paper": [want_t, want_i]}
        print(f"    {method:12} CLIP-T {t:.3f} (paper {want_t}) {s1}  "
              f"CLIP-I {i:.3f} (paper {want_i}) {s2}  n={len(rows)}")


def table4(report):
    print("\n=== Generator calls per task (Sec. 4.3, inline) ===")
    for method in ("one_shot", "umo", "best_of_n", "ours_v2"):
        dirs = [os.path.join(R2, f"{method}_s{s}") for s in SEEDS]
        vals = [r["gen_calls"] for d in dirs for r in read_records(os.path.join(d, "records.jsonl"))
                if r.get("gen_calls") is not None]
        m = mean(vals)
        want = PAPER["t4"][method]
        status = check(f"T4 {method}", m, want, tol=0.05)
        report[f"t4.{method}"] = {"mean": round(m, 3), "n": len(vals), "paper": want, "status": status}
        print(f"    OmniGen2 {method:12} {m:.2f}  n={len(vals)} (paper {want}) {status}")
    vals = [r["gen_calls"] for s in SEEDS
            for r in read_records(os.path.join(FLUX2, f"flux2_8_ours_s{s}", "records.jsonl"))
            if r.get("gen_calls") is not None]
    m = mean(vals)
    status = check("T4 flux2_ours_8", m, PAPER["t4"]["flux2_ours_8"], tol=0.05)
    report["t4.flux2_ours_8"] = {"mean": round(m, 3), "n": len(vals), "paper": 4.3, "status": status}
    print(f"    FLUX.2-8 ours        {m:.2f}  n={len(vals)} (paper 4.3) {status}")


def gating(report):
    """Table (tab:gating): selective correction, pooled over (task, seed) rows."""
    print("\n=== Table (gating): selective correction on FLUX.2 ===")
    for ent in (8, 6):
        rows = {}
        for method in ("oneshot", "bon", "ours"):
            for s in SEEDS:
                for r in read_records(os.path.join(FLUX2, f"flux2_{ent}_{method}_s{s}", "records.jsonl")):
                    rows.setdefault((r["task_id"], s), {})[method] = r
        triggered = {k: v for k, v in rows.items()
                     if v.get("ours", {}).get("accepted_steps", 0) > 0}
        noop = {k: v for k, v in rows.items() if k not in triggered}
        total = len(rows)
        for name, subset in (("triggered", triggered), ("no-op", noop)):
            vals = {m: mean([v[m]["scr"] for v in subset.values() if m in v])
                    for m in ("oneshot", "bon", "ours")}
            pct = 100 * len(subset) / total if total else float("nan")
            report[f"gating.{ent}.{name}"] = {
                "n": len(subset), "pct": round(pct, 1),
                **{m: round(v, 4) for m, v in vals.items()}}
            print(f"    {ent}-entity {name:9} n={len(subset):3} ({pct:.0f}%)  "
                  f"one-shot {vals['oneshot']:.3f}  best-of-8 {vals['bon']:.3f}  MIDC {vals['ours']:.3f}")


def table5(report):
    print("\n=== Human eval (paper: Table 4), 3 labelers ===")
    if not os.path.exists(HUMAN):
        print("    ! aggregate_result_3labeler.json missing")
        return
    summary = json.load(open(HUMAN))["summary"]
    for (baseline, q), (want_wr, want_n, want_k) in PAPER["t5"].items():
        got = summary[baseline][q]
        s1 = check(f"T5 {baseline}/{q} win", got["win_rate"], want_wr, tol=0.001)
        s2 = "OK " if got["n"] == want_n else "MISMATCH"
        if s2 != "OK ":
            failures.append(f"T5 {baseline}/{q} n: {got['n']} vs paper {want_n}")
        report[f"t5.{baseline}.{q}"] = {**got, "paper_win": want_wr, "paper_n": want_n, "status": s1}
        print(f"    {baseline:14} {q}  win={got['win_rate']:.3f} n={got['n']:3} "
              f"kappa={got['fleiss_kappa']:+.2f} (paper {want_wr:.3f}/n={want_n}/k={want_k:+.2f}) {s1}{s2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full recomputed report here")
    args = ap.parse_args()

    report = {}
    table1(report)
    table2(report)
    table3(report)
    clip_table(report)
    table4(report)
    gating(report)
    table5(report)

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} MISMATCH(es) between recomputed data and main.tex:")
        for f in failures:
            print("  -", f)
    else:
        print("All paper numbers reproduce from the committed records.")
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
