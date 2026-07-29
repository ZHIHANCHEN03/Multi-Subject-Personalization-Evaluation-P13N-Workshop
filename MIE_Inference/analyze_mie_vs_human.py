"""How well does MIE agree with human annotators on the PP1 reference-extension study?

Two annotation batches, each a head-to-head over the same 216 prompts x 3 reference
groups:

    pp1   nano_banana  vs  mosaic     (a lopsided pair)
    pp2   flux2        vs  gpt15      (a close pair -- the informative one)

Both carry more than a preference vote: annotators marked existence / appearance /
interaction as 0/1 *for each model separately*. That allows two independent tests
rather than one:

  1. Preference agreement -- does MIE rank the pair the same way humans do?
     Cheap to pass on a lopsided pair, so it is reported but not leaned on.
  2. Per-facet alignment -- does MIE's existence/appearance/interaction probability
     track the fraction of annotators who marked that facet satisfied? This is an
     *absolute* comparison on the same axes MIE was trained to emit, and a scalar
     preference model cannot fake it.

    python analyze_mie_vs_human.py --scores results/mie_scores.json \
        --pp1 anno_pp1.csv --pp2 anno_pp2.csv --out results/mie_vs_human.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_MAP = {"a": "a_real", "b": "b_gpt_image_1", "c": "c_qwen"}
FACETS = ("existence", "appearance", "interaction")

# batch -> (model A, model B) as named in both the CSV columns and the MIE records
BATCHES = {"pp1": ("nb", "mosaic"), "pp2": ("flux2", "gpt15")}
# CSV model prefix -> MIE generator label
GEN_MAP = {"nb": "nano_banana", "mosaic": "mosaic", "flux2": "flux2", "gpt15": "gpt15"}


def load_votes(path: Path, models: tuple[str, str]) -> list[dict]:
    """Rows with the quality flags respected and facet marks coerced to int."""
    out, dropped = [], 0
    for r in csv.DictReader(path.open()):
        if r.get("exclude", "").strip().lower() == "true":
            dropped += 1
            continue
        if r.get("prompt_ilogical", "").strip().lower() == "true":
            dropped += 1
            continue
        rec = {
            "annotator": r["annotator_code"],
            "group": GROUP_MAP[r["ref_group"]],
            "id": int(r["base_id"]),
            "level": int(r["level"]),
            "class_tag": r["class_tag"],
            "ratio_type": r["ratio_type"],
            "preferred": r["preferred_model"],
        }
        good = True
        for m in models:
            for f in FACETS:
                v = r.get(f"{m}_{f}", "")
                if v == "":
                    good = False
                    break
                rec[f"{m}_{f}"] = int(v)
        if good:
            out.append(rec)
        else:
            dropped += 1
    print(f"  {path.name}: kept {len(out)} votes, dropped {dropped} "
          f"(exclude / prompt_ilogical / missing marks)")
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial test against p=0.5."""
    if n == 0:
        return float("nan")
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fleiss_kappa(items: list[list[int]]) -> float:
    """items[i] = [count of category 0, count of category 1] for item i."""
    items = [c for c in items if sum(c) >= 2]
    if not items:
        return float("nan")
    n = sum(items[0])
    if any(sum(c) != n for c in items):          # ragged rater counts
        n = min(sum(c) for c in items)
        items = [c for c in items if sum(c) == n] or items
        if not items:
            return float("nan")
    N, k = len(items), len(items[0])
    if n < 2:
        return float("nan")
    p_j = [sum(c[j] for c in items) / (N * n) for j in range(k)]
    P_i = [(sum(x * x for x in c) - n) / (n * (n - 1)) for c in items]
    P_bar = statistics.mean(P_i)
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) else float("nan")


def auc(scores: list[float], labels: list[int]) -> float:
    """P(score of a positive > score of a negative), ties counted as 0.5.

    Chosen over raw agreement because these head-to-heads are ~90-100% one-sided:
    a constant predictor gets near-perfect agreement but AUC exactly 0.5.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_ci(scores: list[float], labels: list[int], n_boot=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(labels))
    boots = []
    for _ in range(n_boot):
        smp = [rng.randrange(len(labels)) for _ in idx]
        v = auc([scores[i] for i in smp], [labels[i] for i in smp])
        if v == v:
            boots.append(v)
    if not boots:
        return (float("nan"), float("nan"))
    boots.sort()
    return (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])


def boot_ci_diff(pairs: list[tuple[float, float]], n_boot=10000, seed=0):
    """Bootstrap CI for mean(a-b) over paired observations."""
    if not pairs:
        return (float("nan"),) * 4
    rng = random.Random(seed)
    d = [a - b for a, b in pairs]
    md = statistics.mean(d)
    boots = sorted(statistics.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    p = min(2 * min(sum(b >= 0 for b in boots), sum(b <= 0 for b in boots)) / n_boot, 1.0)
    return md, lo, hi, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--pp1", required=True)
    ap.add_argument("--pp2", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop_split", action="store_true",
                    help="drop items whose annotators disagree on preferred_model. "
                         "Raises every agreement number because it removes exactly "
                         "the hard cases -- always report it next to the unfiltered "
                         "run, never instead of it.")
    args = ap.parse_args()

    payload = json.loads(Path(args.scores).read_text())
    mie = {(r["group"], r["generator"], r["id"]): r["mie"]
           for r in payload["records"] if "mie" in r}
    print(f"MIE records with scores: {len(mie)}")

    print("\nloading annotations")
    votes = {b: load_votes(Path(getattr(args, b)), BATCHES[b]) for b in BATCHES}

    report: dict = {"meta": {
        "mie_checkpoint": payload["meta"]["weight_load_report"].get("checkpoint"),
        "mie_base_model": payload["meta"].get("base_model"),
        "batches": {b: list(BATCHES[b]) for b in BATCHES},
    }}

    for batch, (mA, mB) in BATCHES.items():
        rows = votes[batch]
        gA, gB = GEN_MAP[mA], GEN_MAP[mB]
        print(f"\n{'='*78}\n{batch}:  {gA}  vs  {gB}\n{'='*78}")
        sec: dict = {}

        # ---------- 1. preference agreement (per item, majority vote) ----------
        by_item = defaultdict(list)
        for r in rows:
            by_item[(r["group"], r["id"])].append(r)

        if args.drop_split:
            before = len(by_item)
            by_item = {
                k: vs for k, vs in by_item.items()
                if len({v["preferred"] for v in vs}) == 1
            }
            sec_drop = {"items_before": before, "items_after": len(by_item),
                        "dropped": before - len(by_item)}
            print(f"[drop_split] 丢弃标注员偏好不一致的 item: "
                  f"{before} -> {len(by_item)} (丢 {before - len(by_item)})")
        else:
            sec_drop = {"items_before": len(by_item), "items_after": len(by_item),
                        "dropped": 0}
        agree = total = 0
        human_a_wins = 0
        items_used = []
        for key, vs in by_item.items():
            g, pid = key
            if (g, gA, pid) not in mie or (g, gB, pid) not in mie:
                continue
            nA = sum(1 for v in vs if v["preferred"] == mA)
            nB = sum(1 for v in vs if v["preferred"] == mB)
            if nA == nB:                      # tied annotators -> no majority
                continue
            human = mA if nA > nB else mB
            pick = mA if mie[(g, gA, pid)]["total"] > mie[(g, gB, pid)]["total"] else mB
            agree += (human == pick)
            human_a_wins += (human == mA)
            total += 1
            items_used.append(key)
        lo, hi = wilson(agree, total)
        sec["preference_agreement"] = {
            "n_items": total, "agreement": round(agree / total, 4) if total else None,
            "ci95": [round(lo, 4), round(hi, 4)],
            f"human_{mA}_win_rate": round(human_a_wins / total, 4) if total else None,
        }
        print(f"[1] 偏好一致率  MIE vs 人类多数票: {agree}/{total} = "
              f"{agree/total:.1%}  95%CI [{lo:.1%}, {hi:.1%}]")
        print(f"    人类 {mA} 胜率: {human_a_wins/total:.1%}")

        # MIE's own win rate on the same items, for side-by-side
        mie_a = sum(1 for (g, pid) in items_used
                    if mie[(g, gA, pid)]["total"] > mie[(g, gB, pid)]["total"])
        sec["preference_agreement"][f"mie_{mA}_win_rate"] = round(mie_a / total, 4) if total else None
        print(f"    MIE  {mA} 胜率: {mie_a/total:.1%}")

        # These head-to-heads are heavily one-sided, so raw agreement is a trap:
        # a predictor that always names the majority winner scores very high while
        # carrying no information. Report that baseline next to MIE, and use AUC
        # (which pins any constant predictor at exactly 0.5) as the real metric.
        humans = [mA if sum(1 for v in by_item[k] if v["preferred"] == mA)
                  > sum(1 for v in by_item[k] if v["preferred"] == mB) else mB
                  for k in items_used]
        maj = mA if humans.count(mA) > total / 2 else mB
        base = humans.count(maj) / total if total else float("nan")
        margins = [mie[(g, gA, pid)]["total"] - mie[(g, gB, pid)]["total"]
                   for (g, pid) in items_used]
        labels = [1 if h == mA else 0 for h in humans]
        a = auc(margins, labels)
        lo_a, hi_a = auc_ci(margins, labels)
        sec["preference_agreement"].update({
            "majority_class_baseline": round(base, 4),
            "mie_minus_baseline": round(agree / total - base, 4) if total else None,
            "auc": round(a, 4), "auc_ci95": [round(lo_a, 4), round(hi_a, 4)],
            "positive_rate": round(sum(labels) / total, 4) if total else None,
        })
        print(f"    常数基线 (always {maj}): {base:.1%}   "
              f"→ MIE {'优于' if agree/total > base else '不如'}基线 "
              f"{(agree/total - base)*100:+.1f} pp")
        print(f"    AUC(Δtotal→人类偏好) = {a:.3f}  95%CI [{lo_a:.3f}, {hi_a:.3f}]  "
              f"(常数预测器恒为 0.500)")

        # Absolute calibration: is MIE systematically more generous than humans?
        bias = {}
        for m, gen in ((mA, gA), (mB, gB)):
            for f in FACETS:
                ds = [mie[(g, gen, pid)][f]
                      - statistics.mean(v[f"{m}_{f}"] for v in vs)
                      for (g, pid), vs in by_item.items() if (g, gen, pid) in mie]
                bias[f"{gen}.{f}"] = round(statistics.mean(ds), 4) if ds else None
        sec["systematic_bias_mie_minus_human"] = bias

        # ---------- 2. per-facet alignment (the real test) ----------
        print(f"[2] 逐维对齐（人类标注比例 vs MIE 概率，逐 item 聚合）")
        facet_out = {}
        for m, gen in ((mA, gA), (mB, gB)):
            for f in FACETS:
                xs, ys = [], []
                for (g, pid), vs in by_item.items():
                    if (g, gen, pid) not in mie:
                        continue
                    human = statistics.mean(v[f"{m}_{f}"] for v in vs)
                    xs.append(human)
                    ys.append(mie[(g, gen, pid)][f])
                r = pearson(xs, ys)
                # do MIE and humans agree on which side of the midpoint?
                both = [(x, y) for x, y in zip(xs, ys) if x != 0.5]
                acc = (sum(1 for x, y in both if (x > 0.5) == (y > 0.5)) / len(both)
                       if both else float("nan"))
                facet_out[f"{gen}.{f}"] = {
                    "n": len(xs), "pearson_r": round(r, 4),
                    "human_mean": round(statistics.mean(xs), 4),
                    "mie_mean": round(statistics.mean(ys), 4),
                    "binary_agreement": round(acc, 4),
                }
                print(f"    {gen:12} {f:11} r={r:+.3f}  人类均值={statistics.mean(xs):.3f} "
                      f"MIE均值={statistics.mean(ys):.3f}  同侧率={acc:.1%}")
        sec["facet_alignment"] = facet_out

        # ---------- 3. does MIE's margin predict human agreement? ----------
        # If the score magnitude is meaningful, items where MIE is confident should
        # be items where annotators agree more strongly.
        buckets = defaultdict(lambda: [0, 0])
        for (g, pid), vs in by_item.items():
            if (g, gA, pid) not in mie or (g, gB, pid) not in mie:
                continue
            margin = abs(mie[(g, gA, pid)]["total"] - mie[(g, gB, pid)]["total"])
            nA = sum(1 for v in vs if v["preferred"] == mA)
            nB = sum(1 for v in vs if v["preferred"] == mB)
            if nA == nB:
                continue
            human = mA if nA > nB else mB
            pick = mA if mie[(g, gA, pid)]["total"] > mie[(g, gB, pid)]["total"] else mB
            q = "low" if margin < 0.5 else ("mid" if margin < 1.5 else "high")
            buckets[q][0] += (human == pick)
            buckets[q][1] += 1
        print(f"[3] MIE 分差 vs 一致率（分差越大是否越可靠）")
        margin_out = {}
        for q in ("low", "mid", "high"):
            k, n = buckets[q]
            if n:
                margin_out[q] = {"n": n, "agreement": round(k / n, 4)}
                print(f"    |Δtotal| {q:4}: {k}/{n} = {k/n:.1%}")
        sec["margin_vs_agreement"] = margin_out

        # ---------- 4. inter-annotator agreement ----------
        print(f"[4] 标注员一致性 (Fleiss kappa, 二分类)")
        kap = {}
        for m, gen in ((mA, gA), (mB, gB)):
            for f in FACETS:
                counts = []
                for vs in by_item.values():
                    if len(vs) < 2:
                        continue
                    ones = sum(v[f"{m}_{f}"] for v in vs)
                    counts.append([len(vs) - ones, ones])
                k = fleiss_kappa(counts)
                kap[f"{gen}.{f}"] = round(k, 4)
                print(f"    {gen:12} {f:11} kappa={k:+.3f}")
        pref_counts = []
        for vs in by_item.values():
            if len(vs) < 2:
                continue
            a = sum(1 for v in vs if v["preferred"] == mA)
            pref_counts.append([len(vs) - a, a])
        kp = fleiss_kappa(pref_counts)
        kap["preference"] = round(kp, 4)
        print(f"    {'preference':12} {'':11} kappa={kp:+.3f}")
        sec["fleiss_kappa"] = kap

        # ---------- 5. by subject count ----------
        print(f"[5] 按主体数分解（偏好一致率）")
        lvl = defaultdict(lambda: [0, 0])
        for (g, pid), vs in by_item.items():
            if (g, gA, pid) not in mie or (g, gB, pid) not in mie:
                continue
            nA = sum(1 for v in vs if v["preferred"] == mA)
            nB = sum(1 for v in vs if v["preferred"] == mB)
            if nA == nB:
                continue
            human = mA if nA > nB else mB
            pick = mA if mie[(g, gA, pid)]["total"] > mie[(g, gB, pid)]["total"] else mB
            lvl[vs[0]["level"]][0] += (human == pick)
            lvl[vs[0]["level"]][1] += 1
        by_level = {}
        for L in sorted(lvl):
            k, n = lvl[L]
            by_level[L] = {"n": n, "agreement": round(k / n, 4)}
            print(f"    level {L}: {k}/{n} = {k/n:.1%}")
        sec["by_level"] = by_level

        sec["drop_split"] = sec_drop
        report[batch] = sec

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
