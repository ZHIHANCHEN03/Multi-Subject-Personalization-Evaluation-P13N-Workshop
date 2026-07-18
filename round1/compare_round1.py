"""Aggregate Round-1 results and print the three go/no-go signals.

Reads runs/<method>/records.jsonl for each method and reports, on the hard set:
  mean SCR (lower=better) and mean DINO identity (higher=better), plus the
  three decision signals:
    1. OURS beats best-of-N / one-shot ?          (method works)
    2. OURS >= UMO ?                               (training-free rivals retrained)
    3. OURS beats FreeGraftor ?                    (closed-loop beats open-loop)

Paired win-rate (per task_id, on SCR) is reported where both methods have the
same tasks.

Usage:
  python compare_round1.py --runs runs \
      --ours ours --oneshot one_shot --bon best_of_n --umo umo --fg freegraftor
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def load(run_dir: Path, name: str) -> dict:
    p = run_dir / name / "records.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def finite_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def method_stats(recs: dict) -> dict:
    routes = Counter()
    for record in recs.values():
        for step in record.get("step_log", []):
            if step.get("routed_dim"):
                routes[step["routed_dim"]] += 1
    return {
        "n": len(recs),
        "mean_scr": finite_mean([r.get("scr") for r in recs.values()]),
        "mean_dino": finite_mean([r.get("dino_mean") for r in recs.values()]),
        "failure_rate": finite_mean(
            [float(r.get("generation_failed", False)) for r in recs.values()]
        ),
        "mean_accepted_steps": finite_mean(
            [r.get("accepted_steps") for r in recs.values()]
        ),
        "routing_distribution": dict(routes),
    }


def summarize(recs: dict, label: str):
    scr = mean([r.get("scr") for r in recs.values()])
    dino = mean([r.get("dino_mean") for r in recs.values()])
    fail = mean([float(r.get("generation_failed", False)) for r in recs.values()])
    print(
        f"  {label:14s}  n={len(recs):3d}  mean_SCR={scr:.3f}  "
        f"mean_DINO={dino:.3f}  failure={fail:.1%}"
    )
    return scr, dino


def paired_winrate(a: dict, b: dict, metric: str = "scr", lower_better: bool = True):
    """Fraction of shared tasks where a is better than b on `metric`."""
    keys = set(a) & set(b)
    if not keys:
        return float("nan"), 0
    wins, valid = 0, 0
    for k in keys:
        va, vb = a[k].get(metric), b[k].get(metric)
        if va is None or vb is None:
            continue
        valid += 1
        if (va < vb) if lower_better else (va > vb):
            wins += 1
    return (wins / valid if valid else float("nan")), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ours", default="ours")
    ap.add_argument("--oneshot", default="one_shot")
    ap.add_argument("--bon", default="best_of_n")
    ap.add_argument("--umo", default="umo")
    ap.add_argument("--fg", default="freegraftor")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--noninferiority_margin", type=float, default=0.05)
    args = ap.parse_args()

    rd = Path(args.runs)
    recs = {
        "ours": load(rd, args.ours),
        "one_shot": load(rd, args.oneshot),
        "best_of_n": load(rd, args.bon),
        "umo": load(rd, args.umo),
        "freegraftor": load(rd, args.fg),
    }

    def subset(source: dict, predicate):
        return {
            k: r for k, r in source.items()
            if predicate(int(r.get("meta", {}).get("total_entities", 0)))
        }

    def report_slice(title: str, sliced: dict):
        print(f"\n=== {title} ===")
        for label, r in sliced.items():
            if r:
                summarize(r, label)
            else:
                print(f"  {label:14s}  (no records)")

        print("  -- paired go/no-go signals (SCR; lower is better) --")

        def report(name, a_key, b_key):
            if sliced[a_key] and sliced[b_key]:
                wr, n = paired_winrate(sliced[a_key], sliced[b_key])
                verdict = "PASS" if wr > 0.5 else "FAIL"
                print(
                    f"  {name:34s} ours-vs-{b_key:12s} "
                    f"winrate={wr:.2%} (n={n})  [{verdict}]"
                )
            else:
                print(f"  {name:34s} (missing ours or {b_key})")

        report("1. method works", "ours", "best_of_n")
        report("1b. vs one-shot", "ours", "one_shot")
        report("2. rivals retrained (UMO)", "ours", "umo")
        report("3. vs open-loop FreeGraftor", "ours", "freegraftor")

    report_slice("Round-1 overall (all cases)", recs)
    report_slice(
        "Hard slice (4 entities; primary fair comparison, OmniGen2 max refs=5)",
        {k: subset(v, lambda n: n == 4) for k, v in recs.items()},
    )
    report_slice(
        "Easy contrast slice (2 entities)",
        {k: subset(v, lambda n: n == 2) for k, v in recs.items()},
    )

    slices = {
        "overall": recs,
        "hard_4_entities": {
            k: subset(v, lambda n: n == 4) for k, v in recs.items()
        },
        "easy_2_entities": {
            k: subset(v, lambda n: n == 2) for k, v in recs.items()
        },
    }
    primary = {
        name: method_stats(records)
        for name, records in slices["hard_4_entities"].items()
    }

    def scr(name):
        return primary[name]["mean_scr"]

    # FreeGraftor is an OPTIONAL cross-system reference (different base model);
    # the verdict rests on the same-base OmniGen2 comparisons.
    required = ["ours", "one_shot", "best_of_n", "umo"]
    missing = [name for name in required if scr(name) is None]
    fg_available = scr("freegraftor") is not None
    if missing:
        verdict = "INCOMPLETE"
        signals = {}
        reason = f"missing scored outputs for core methods: {', '.join(missing)}"
    else:
        # Claim = "training-free test-time repair rivals retraining". So the GO
        # gate is: (1) the loop clearly helps over naive one-shot, and (2) it is
        # non-inferior to the same-base RETRAINED SOTA (UMO). best-of-N is a
        # compute-matched *reference* (also training-free test-time), NOT a gate:
        # we deliberately do NOT claim "correction > selection" (that is Ma et al.).
        signals = {
            "beats_one_shot": scr("ours") < scr("one_shot"),
            "noninferior_to_umo": (
                scr("ours") <= scr("umo") + args.noninferiority_margin
            ),
        }
        # context-only comparisons (reported, not gating):
        signals_context = {
            "vs_best_of_n(context)": scr("ours") <= scr("best_of_n") + args.noninferiority_margin,
        }
        if fg_available:
            signals_context["vs_freegraftor(context)"] = scr("ours") < scr("freegraftor")
        signals.update(signals_context)

        core_ok = signals["beats_one_shot"] and signals["noninferior_to_umo"]
        if core_ok:
            verdict = "GO"
            reason = (
                "training-free loop beats one-shot and is non-inferior to the "
                "same-base retrained SOTA (UMO); best-of-N is a compute-matched "
                "reference, not a gate"
            )
        elif signals["beats_one_shot"]:
            verdict = "CONDITIONAL"
            reason = (
                "loop helps over one-shot but did not reach UMO parity; "
                "consider stronger correction actions / budget reallocation"
            )
        else:
            verdict = "STOP"
            reason = "loop did not beat one-shot"

    evaluation = {
        "verdict": verdict,
        "reason": reason,
        "noninferiority_margin_scr": args.noninferiority_margin,
        "signals": signals,
        "missing_methods": missing,
        "slices": {
            slice_name: {
                method: method_stats(records)
                for method, records in slice_records.items()
            }
            for slice_name, slice_records in slices.items()
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    signal_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in signals.items()
    ) or "- unavailable"
    (out_dir / "DECISION.md").write_text(
        "# Round 1 decision\n\n"
        f"## {verdict}\n\n"
        f"{reason}.\n\n"
        "### Pre-registered signals\n"
        f"{signal_lines}\n\n"
        "Primary decision uses the 4-entity hard slice (hardest OmniGen2 "
        "natively supports; max 5 reference images). 6/8-entity extreme "
        "collapse is deferred to Round 2 on a base supporting more refs.\n",
        encoding="utf-8",
    )
    print(f"\n=== FINAL ROUND-1 VERDICT: {verdict} ===")
    print(reason)
    print(f"Reports: {out_dir / 'evaluation.json'} and {out_dir / 'DECISION.md'}\n")


if __name__ == "__main__":
    main()
