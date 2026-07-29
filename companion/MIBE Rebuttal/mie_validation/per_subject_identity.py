"""Does identity preservation hold up on subjects MIB does not contain?

Reviewer ws3J: "The number of reference subjects is also fairly limited, given that
identity-preservation is a key element the benchmark. It might be useful to see how
well identity preserving extends for more obscure, long-tailed classes."

The reference-extension study adds 22 subjects with zero overlap against MIB's 80.
Crucially they are not uniformly "new" in the same sense: the 10 humans are new
*identities* drawn from demographic descriptions adjacent to MIB's, while the 12
animal / object / food subjects are new *categories* -- MIB contains no camelid, no
wading bird, no stringed instrument, no basket. That split is exactly the reviewer's
question, so it is the contrast this script measures.

Attribution problem and how it is handled: a score is produced per *image*, and each
image contains 2-8 subjects, so no score belongs to one subject alone. We therefore
work with residuals -- each image's score minus the mean of its
(subject-count, generator, reference-source) cell -- and attribute the residual
equally to every subject present. Averaged over the ~hundreds of images a subject
appears in, that gives a per-subject index whose nuisance structure (harder prompts
have more subjects; generators differ wildly) has been removed. It is a first-order
attribution, not a causal decomposition, and the permutation test below is what says
whether the spread it produces is larger than chance.

Two independent measurements are reported for every subject:
    human   -- the fraction of annotator marks satisfied, from the two ballot batches
    MIE     -- the evaluator's own facet probability
If the human index shows no penalty for the new categories, the answer to the
reviewer holds regardless of what MIE says.

    python3 per_subject_identity.py            # -> results/per_subject_identity.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP_MAP = {"a": "a_real", "b": "b_gpt_image_1", "c": "c_qwen"}
BATCHES = {"pp1": ("nb", "mosaic"), "pp2": ("flux2", "gpt15")}
GEN_MAP = {"nb": "nano_banana", "mosaic": "mosaic", "flux2": "flux2", "gpt15": "gpt15"}

# MIB's own inventory, read off the released manifests, so "new category" is checked
# rather than asserted.
ANIMAL = {"alpaca", "dairy_cow", "flamingo", "hamster", "turtle"}
OBJECT = {"desk_globe", "electric_kettle", "picnic_basket", "shopping_cart", "violin"}
FOOD = {"croissant", "cupcake"}


def category(slug: str) -> str:
    if slug in ANIMAL:
        return "animal"
    if slug in OBJECT:
        return "object"
    if slug in FOOD:
        return "food"
    return "human"


def kind(slug: str) -> str:
    """The distinction the reviewer is actually asking about."""
    return "new identity" if category(slug) == "human" else "new category"


def residualise(rows: list[dict], cell_keys: tuple, value_key) -> list[tuple[list[str], float]]:
    """Return (subjects, residual) after removing each (cell) mean."""
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in cell_keys)].append(value_key(r))
    means = {c: statistics.mean(v) for c, v in cells.items()}
    out = []
    for r in rows:
        c = tuple(r[k] for k in cell_keys)
        out.append((r["ref_slugs"], value_key(r) - means[c]))
    return out


def per_subject(pairs: list[tuple[list[str], float]]) -> dict[str, list[float]]:
    acc = defaultdict(list)
    for slugs, resid in pairs:
        for s in slugs:
            acc[s].append(resid)
    return acc


def perm_test(pairs, n_perm: int = 2000, seed: int = 0) -> float:
    """Is the spread across subjects larger than random relabelling produces?

    Statistic: the standard deviation of per-subject mean residuals. Under the null
    (subject identity carries no information) shuffling the residuals across images
    should reproduce it.
    """
    obs = statistics.pstdev([statistics.mean(v) for v in per_subject(pairs).values()])
    rng = random.Random(seed)
    vals = [r for _, r in pairs]
    slug_lists = [s for s, _ in pairs]
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(vals)
        sd = statistics.pstdev(
            [statistics.mean(v) for v in per_subject(list(zip(slug_lists, vals))).values()])
        ge += sd >= obs
    return (ge + 1) / (n_perm + 1)


def boot_mean_ci(xs: list[float], n_boot=4000, seed=0) -> tuple[float, float]:
    rng = random.Random(seed)
    b = sorted(statistics.mean(xs[rng.randrange(len(xs))] for _ in xs) for _ in range(n_boot))
    return b[int(0.025 * n_boot)], b[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(HERE / "mie_scores.json"))
    ap.add_argument("--out", default=str(HERE / "per_subject_identity.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.scores).read_text())
    recs = [r for r in payload["records"] if "mie" in r]
    # (group, generator, id) -> record, so the annotation CSVs can be joined on it
    by_key = {(r["group"], r["generator"], r["id"]): r for r in recs}
    slugs_of = {(r["group"], r["id"]): r["ref_slugs"] for r in recs}

    report: dict = {"n_subjects": len({s for r in recs for s in r["ref_slugs"]})}
    print(f"subjects: {report['n_subjects']}   scored cells: {len(recs)}")

    # ---------------- human side ----------------
    # Each ballot row carries per-model 0/1 marks; join to the subject list by
    # (reference group, prompt id). Appearance is the identity-preservation facet.
    human_rows = []
    for batch, (mA, mB) in BATCHES.items():
        for r in csv.DictReader((HERE / f"anno_{batch}.csv").open()):
            if r["exclude"] == "True" or r["prompt_ilogical"] == "True":
                continue
            key = (GROUP_MAP[r["ref_group"]], int(r["base_id"]))
            if key not in slugs_of:
                continue
            for m in (mA, mB):
                human_rows.append({
                    "ref_slugs": slugs_of[key], "level": int(r["level"]),
                    "generator": GEN_MAP[m], "group": key[0],
                    "appearance": int(r[f"{m}_appearance"]),
                    "existence": int(r[f"{m}_existence"]),
                })
    print(f"human marks joined: {len(human_rows)}")

    out_facets = {}
    for facet in ("appearance", "existence"):
        print(f"\n{'='*76}\n人类标注 · {facet}（已扣除 主体数×生成器×参考源 的单元均值）\n{'='*76}")
        pairs = residualise(human_rows, ("level", "generator", "group"),
                            lambda r, f=facet: r[f])
        acc = per_subject(pairs)
        rows = sorted(((statistics.mean(v), s, len(v)) for s, v in acc.items()))
        for m, s, n in rows:
            lo, hi = boot_mean_ci(acc[s])
            flag = "" if lo <= 0 <= hi else "  *"
            print(f"  {s:32} {kind(s):13} {m:+.4f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={n}{flag}")
        p = perm_test(pairs)
        print(f"  逐主体差异的置换检验 p = {p:.4f}"
              f"  ({'主体身份确实带信息' if p < 0.05 else '主体之间无显著差异'})")

        by_kind = defaultdict(list)
        for s, v in acc.items():
            by_kind[kind(s)].extend(v)
        print("  按类型:")
        kind_out = {}
        for k, v in sorted(by_kind.items()):
            lo, hi = boot_mean_ci(v)
            kind_out[k] = {"mean_residual": round(statistics.mean(v), 4),
                           "ci95": [round(lo, 4), round(hi, 4)], "n": len(v)}
            print(f"    {k:13} {statistics.mean(v):+.4f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(v)}")
        # the contrast the reviewer is asking for
        a = by_kind["new category"]
        b = by_kind["new identity"]
        rng = random.Random(0)
        diffs = sorted(statistics.mean(a[rng.randrange(len(a))] for _ in a)
                       - statistics.mean(b[rng.randrange(len(b))] for _ in b)
                       for _ in range(4000))
        d, lo, hi = statistics.mean(a) - statistics.mean(b), diffs[100], diffs[3900]
        print(f"    对比 (new category − new identity): {d:+.4f}  95%CI[{lo:+.3f},{hi:+.3f}]")
        out_facets[f"human.{facet}"] = {
            "per_subject": {s: {"mean_residual": round(statistics.mean(v), 4), "n": len(v)}
                            for s, v in acc.items()},
            "by_kind": kind_out, "permutation_p": round(p, 4),
            "category_minus_identity": {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]},
        }

    # ---------------- MIE side, same treatment ----------------
    mie_rows = [{"ref_slugs": r["ref_slugs"], "level": r["level"],
                 "generator": r["generator"], "group": r["group"],
                 "appearance": r["mie"]["appearance"], "existence": r["mie"]["existence"]}
                for r in recs]
    for facet in ("appearance", "existence"):
        print(f"\n{'='*76}\nMIE · {facet}（同样扣除单元均值）\n{'='*76}")
        pairs = residualise(mie_rows, ("level", "generator", "group"),
                            lambda r, f=facet: r[f])
        acc = per_subject(pairs)
        by_kind = defaultdict(list)
        for s, v in acc.items():
            by_kind[kind(s)].extend(v)
        kind_out = {}
        for k, v in sorted(by_kind.items()):
            lo, hi = boot_mean_ci(v)
            kind_out[k] = {"mean_residual": round(statistics.mean(v), 4),
                           "ci95": [round(lo, 4), round(hi, 4)], "n": len(v)}
            print(f"  {k:13} {statistics.mean(v):+.4f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(v)}")
        worst = sorted(((statistics.mean(v), s) for s, v in acc.items()))[:5]
        best = sorted(((statistics.mean(v), s) for s, v in acc.items()))[-3:]
        print("  最难的 5 个:", ", ".join(f"{s} ({m:+.3f})" for m, s in worst))
        print("  最容易的 3 个:", ", ".join(f"{s} ({m:+.3f})" for m, s in reversed(best)))
        out_facets[f"mie.{facet}"] = {
            "per_subject": {s: {"mean_residual": round(statistics.mean(v), 4), "n": len(v)}
                            for s, v in acc.items()},
            "by_kind": kind_out,
        }

    report["facets"] = out_facets
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
