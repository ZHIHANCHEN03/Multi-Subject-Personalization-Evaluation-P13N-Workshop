"""Human evaluation: does the retrained SOTA (UMO) beat plain one-shot generation?

The paper's central diagnostic claim is that retraining the identity-consistency
axis buys almost nothing on hard interaction cases -- UMO vs one_shot is a near-tie
on SCR (Δ=-0.005) and DINO (Δ=-0.0002) over 500x3 pairs. Those are automatic
metrics computed by the same DINOv2 pipeline, so the obvious objection is that the
metric is blind to a real improvement. This blind A/B asks humans directly.

Note the direction of the test: **a tie is the predicted outcome**, not a failure.
We are not hunting for significance; we are checking whether a difference exists
that the automatic metrics missed. A confidence interval covering 0.5 supports the
paper; a decisive win for UMO would falsify it and we would have to say so.

Questions on the ballot (blind, left/right randomised per pair):
    Q1  existence   -- which side shows all the requested subjects
    Q2  identity/appearance -- which side matches the references better

    python analyze_umo_vs_oneshot.py --key key.json \
        --votes votes_labeler_A.json votes_labeler_B.json \
        --records results_r2/merged --out umo_vs_oneshot.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

QUESTIONS = {"q1": "existence", "q2": "identity/appearance"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial test against p = 0.5."""
    if n == 0:
        return float("nan")
    pmf = lambda i: math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Unweighted Cohen's kappa for two raters over the same items."""
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def read_records(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["task_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="key.json from the blinded export")
    ap.add_argument("--votes", nargs="+", required=True)
    ap.add_argument("--records", default=None,
                    help="results_r2/merged, to compare against the automatic metrics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    ballots = []
    for p in args.votes:
        d = json.loads(Path(p).read_text())
        ballots.append((d.get("labeler", Path(p).stem), d["votes"]))
    print(f"key: {len(key)} pairs | labelers: {[n for n, _ in ballots]}")

    # Decode LEFT/RIGHT to a method name using each pair's randomised side.
    # key[pid]['main'] is umo_s0 and sits on key[pid]['ours_side'].
    decoded: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)  # q -> labeler -> pid -> choice
    per_q: dict[str, dict[str, dict[str, str]]] = {q: defaultdict(dict) for q in QUESTIONS}
    for name, votes in ballots:
        for pid, v in votes.items():
            if pid not in key:
                continue
            k = key[pid]
            main_side = k["ours_side"]                       # side showing 'main' (= umo)
            for q in QUESTIONS:
                choice = v.get(q)
                if choice is None:
                    continue
                if choice == "TIE":
                    per_q[q][name][pid] = "tie"
                elif choice == main_side:
                    per_q[q][name][pid] = k["main"]           # umo_s0
                else:
                    per_q[q][name][pid] = k["other"]          # one_shot_s0
    MAIN, OTHER = key[next(iter(key))]["main"], key[next(iter(key))]["other"]
    print(f"main={MAIN}  other={OTHER}\n")

    report: dict = {"main": MAIN, "other": OTHER, "n_pairs": len(key),
                    "labelers": [n for n, _ in ballots], "questions": QUESTIONS}

    # ---- per labeler ----
    print("=" * 74)
    print("每位标注员（在非平票上的 UMO 胜率）")
    print("=" * 74)
    per_labeler = {}
    for q, qname in QUESTIONS.items():
        for name, _ in ballots:
            d = per_q[q][name]
            tie = sum(1 for v in d.values() if v == "tie")
            umo = sum(1 for v in d.values() if v == MAIN)
            one = sum(1 for v in d.values() if v == OTHER)
            n = umo + one
            lo, hi = wilson(umo, n)
            p = binom_p(umo, n)
            per_labeler[f"{q}.{name}"] = {
                "tie": tie, "umo": umo, "one_shot": one, "n_nontie": n,
                "umo_win_rate": round(umo / n, 4) if n else None,
                "ci95": [round(lo, 4), round(hi, 4)], "binom_p": round(p, 4),
            }
            wr = f"{umo/n:.1%}" if n else "  n/a"
            print(f"  {qname:20} {name:8} tie={tie:3}  UMO={umo:3} one_shot={one:3}  "
                  f"胜率={wr}  CI[{lo:.2f},{hi:.2f}]  p={p:.3f}")
    report["per_labeler"] = per_labeler

    # ---- pooled over labelers (each vote counts once) ----
    print("\n" + "=" * 74)
    print("合并两位标注员的全部投票")
    print("=" * 74)
    pooled = {}
    for q, qname in QUESTIONS.items():
        umo = one = tie = 0
        for name, _ in ballots:
            for v in per_q[q][name].values():
                umo += v == MAIN
                one += v == OTHER
                tie += v == "tie"
        n = umo + one
        lo, hi = wilson(umo, n)
        p = binom_p(umo, n)
        total = umo + one + tie
        pooled[q] = {"tie": tie, "umo": umo, "one_shot": one, "n_nontie": n,
                     "tie_rate": round(tie / total, 4) if total else None,
                     "umo_win_rate": round(umo / n, 4) if n else None,
                     "ci95": [round(lo, 4), round(hi, 4)], "binom_p": round(p, 4)}
        print(f"  {qname:20} 平票={tie:3}/{total} ({tie/total:.0%})  "
              f"UMO={umo} one_shot={one}  胜率={umo/n:.1%} CI[{lo:.2f},{hi:.2f}] p={p:.3f}"
              if n else f"  {qname:20} 全部平票")
    report["pooled"] = pooled

    # ---- inter-labeler agreement ----
    print("\n" + "=" * 74)
    print("两位标注员的一致性")
    print("=" * 74)
    kappas = {}
    if len(ballots) == 2:
        (n1, _), (n2, _) = ballots
        for q, qname in QUESTIONS.items():
            shared = sorted(set(per_q[q][n1]) & set(per_q[q][n2]))
            a = [per_q[q][n1][p] for p in shared]
            b = [per_q[q][n2][p] for p in shared]
            agree = sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan")
            k = cohen_kappa(a, b)
            kappas[q] = {"n": len(a), "raw_agreement": round(agree, 4),
                         "cohen_kappa": round(k, 4)}
            print(f"  {qname:20} n={len(a)}  逐项一致={agree:.1%}  Cohen κ={k:+.3f}")
    report["inter_labeler"] = kappas

    # ---- cross-check against the automatic metrics on the same tasks ----
    if args.records:
        print("\n" + "=" * 74)
        print("同样这批任务上的自动指标（UMO − one_shot）")
        print("=" * 74)
        root = Path(args.records)
        umo_r = read_records(root / f"{MAIN}" / "records.jsonl")
        one_r = read_records(root / f"{OTHER}" / "records.jsonl")
        tids = [key[p]["task_id"] for p in key]
        auto = {}
        for metric in ("scr", "dino_mean"):
            d = [umo_r[t][metric] - one_r[t][metric] for t in tids
                 if t in umo_r and t in one_r
                 and umo_r[t].get(metric) is not None and one_r[t].get(metric) is not None]
            if not d:
                continue
            mean = statistics.mean(d)
            wins = sum(1 for x in d if (x < 0 if metric == "scr" else x > 0))
            ties = sum(1 for x in d if x == 0)
            auto[metric] = {"n": len(d), "mean_diff": round(mean, 5),
                            "umo_better": wins, "tie": ties,
                            "one_shot_better": len(d) - wins - ties}
            better = "UMO 更好" if (mean < 0 if metric == "scr" else mean > 0) else "one_shot 更好"
            print(f"  {metric:10} n={len(d)}  平均差={mean:+.4f} ({better})  "
                  f"UMO赢/平/输 = {wins}/{ties}/{len(d)-wins-ties}")
        report["automatic_metrics_same_tasks"] = auto

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
