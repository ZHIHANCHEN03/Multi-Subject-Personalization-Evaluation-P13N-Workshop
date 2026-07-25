"""B-1 re-analysis v3: POOLED (task,seed) rows to match paper's n=84/216 split."""
import json, os, statistics, random
from collections import defaultdict

BASE = "/workspace/misc/round2/results_flux2"

def load_rows(method_prefix, entity):
    rows = []
    for s in (0, 1, 2):
        p = f"{BASE}/flux2_{entity}_{method_prefix}_s{s}/records.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_seed"] = s
            rows.append(r)
    return rows

def mean(xs):
    return statistics.mean(xs) if xs else float("nan")

def bootstrap_ci(diffs, n_boot=10000):
    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "ci": [float("nan")]*2, "p": float("nan"), "n": 0}
    md = mean(diffs)
    boots = sorted(mean(random.choice(diffs) for _ in range(n)) for _ in range(n_boot))
    lo, hi = boots[int(0.025*n_boot)], boots[int(0.975*n_boot)]
    ge = sum(1 for b in boots if b >= 0) / n_boot
    le = sum(1 for b in boots if b <= 0) / n_boot
    p = min(2 * min(ge, le), 1.0)
    return {"mean_diff": round(md, 4), "ci": [round(lo, 4), round(hi, 4)], "p": round(p, 4), "n": n}

def by_key(rows, metric="scr"):
    return {(r["task_id"], r["_seed"]): r[metric] for r in rows}

out = {}
for entity in (6, 8):
    print(f"\n{'='*70}\n  {entity}-entity FLUX.2 (POOLED rows)\n{'='*70}")
    one_rows = load_rows("oneshot", entity)
    bon_rows = load_rows("bon", entity)
    our_rows = load_rows("ours", entity)
    print(f"  rows: oneshot={len(one_rows)} bon={len(bon_rows)} ours={len(our_rows)}")

    one_scr = by_key(one_rows, "scr")
    bon_scr = by_key(bon_rows, "scr")
    our_scr = by_key(our_rows, "scr")
    mask = {(r["task_id"], r["_seed"]): (r.get("accepted_steps", 0) > 0) for r in our_rows}

    n_trig = sum(mask.values())
    n_noop = len(mask) - n_trig
    pct = 100.0 * n_trig / len(mask)

    def sm(d, m, val):
        vals = [v for k, v in d.items() if m.get(k) == val]
        return mean(vals), len(vals)

    o_t, _ = sm(one_scr, mask, True); o_n, _ = sm(one_scr, mask, False)
    b_t, _ = sm(bon_scr, mask, True); b_n, _ = sm(bon_scr, mask, False)
    m_t, _ = sm(our_scr, mask, True); m_n, _ = sm(our_scr, mask, False)

    print(f"\n  Gating table (POOLED, SCR lower better):")
    print(f"  {'split':<10} {'subset':<10} {'n (%)':<14} {'one-shot':<10} {'best-of-8':<10} {'MIDC':<10}")
    print(f"  {'-'*64}")
    print(f"  {entity}-entity  triggered  {n_trig} ({pct:.0f}%)     {o_t:.3f}    {b_t:.3f}    {m_t:.3f}")
    print(f"  {entity}-entity  no-op      {n_noop} ({100-pct:.0f}%)     {o_n:.3f}    {b_n:.3f}    {m_n:.3f}")
    all_o = mean(list(one_scr.values())); all_b = mean(list(bon_scr.values())); all_m = mean(list(our_scr.values()))
    print(f"  {entity}-entity  ALL        {len(mask)} (100%)   {all_o:.3f}    {all_b:.3f}    {all_m:.3f}")

    out[f"{entity}_gating_pooled"] = {
        "triggered": {"n": n_trig, "pct": round(pct, 1), "oneshot": round(o_t, 3), "bon": round(b_t, 3), "midc": round(m_t, 3)},
        "noop":      {"n": n_noop, "pct": round(100 - pct, 1), "oneshot": round(o_n, 3), "bon": round(b_n, 3), "midc": round(m_n, 3)},
        "all":       {"n": len(mask), "oneshot": round(all_o, 3), "bon": round(all_b, 3), "midc": round(all_m, 3)},
    }

    for subset_name, keys in [("TRIGGERED", [k for k, v in mask.items() if v]),
                              ("NO-OP", [k for k, v in mask.items() if not v])]:
        print(f"\n  Paired bootstrap on {subset_name} subset (n={len(keys)}):")
        for name, a, b in [("MIDC vs oneshot", our_scr, one_scr),
                          ("MIDC vs bon", our_scr, bon_scr),
                          ("bon vs oneshot", bon_scr, one_scr)]:
            d = [a[k] - b[k] for k in keys if k in a and k in b]
            bs = bootstrap_ci(d)
            win = sum(1 for x in d if x < 0)
            loss = sum(1 for x in d if x > 0)
            tie = len(d) - win - loss
            print(f"    {name:<18} diff={bs['mean_diff']:+.4f} CI={bs['ci']} p={bs['p']} win/tie/loss={win}/{tie}/{loss}")
            out[f"{entity}_{subset_name}_{name.replace(' ','_')}"] = bs

    print(f"\n  Best-of-2 (no-op subset, POOLED):")
    print(f"    MIDC no-op SCR={m_n:.3f}, one-shot SCR={o_n:.3f}, bon SCR={b_n:.3f}")
    print(f"    MIDC-no-op vs one-shot: {m_n - o_n:+.3f}")
    print(f"    bon vs one-shot on no-op: {b_n - o_n:+.3f}")
    out[f"{entity}_bestof2_pooled"] = {"midc_noop": round(m_n, 3), "oneshot": round(o_n, 3), "bon": round(b_n, 3)}

json.dump(out, open("/tmp/b1_pooled.json", "w"), indent=2)
print(f"\nWrote /tmp/b1_pooled.json")
