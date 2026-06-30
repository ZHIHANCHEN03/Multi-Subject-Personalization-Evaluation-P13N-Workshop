"""DPO pair selection / data preparation.

Data sources
------------
1. prompt/train_60k_v13_2.jsonl   -> subject count N (total_entities) per task
2. 60k_LLM_Result/2_5_merged_sorted.jsonl   -> Gemini-2.5-flash judgements
3. 60k_LLM_Result/3_1_merged_sorted.jsonl   -> Gemini-3.1 (second source) judgements

Selection logic
---------------
Layer 0  consensus + signal
    - keep only pairs where the two sources agree on `winner` (A/B)
    - average the two sources' per-dimension {existence, appearance, interaction}
      pass/fail scores, build delta = winner_triplet - loser_triplet
    - drop weak signal where ||delta||_1 ~ 0

Layer 1  structural stratification
    - bucket by N in {2,4,6,8}, even 25% quota per bucket
    - within an N bucket, balance across the 3 primary dimensions (1/3 each)
      -> the (N, primary_dim) grid gives the 4 x 3 = 12 sampling cells
        (the second-level balance the design refers to as "36 buckets" is the
         (N x primary_dim x winner) cross; sampling is done on the 12 (N,dim)
         cells, which is the operative quota)

Layer 2  significance filter (per cell)
    - S = max_k(delta_k) - mean_k(delta_k)
    - keep a pair if S >= cell median(S)  OR  ||delta|| >= cell median(||delta||)

Layer 3  primary quota + sampling
    - primary = argmax_k(delta_k); e/a/i each get 1/3 of the N bucket
    - within a cell, random.sample(seed) down to the per-cell target
      (fallback: take all if the cell has fewer than the target)

Output
------
    selected_pairs_2k.jsonl + sampling_stats_2k.json
    selected_pairs_5k.jsonl + sampling_stats_5k.json
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

DIMS = ["existence", "appearance", "interaction"]
SUBJECT_COUNTS = [2, 4, 6, 8]


def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["task_id"])] = r
    return rows


def load_subject_count(path):
    """Subject count N per task, sourced from prompt/train_60k_v13_2.jsonl.

    The prompt file is keyed by `id` and stores N as `total_entities`.
    """
    n = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n[str(r["id"])] = r.get("total_entities")
    return n


def triplet(row, side):
    return [float(row[f"{side}_{d}"]) for d in DIMS]


def precompute(s25, s31, n_map, min_norm, stats):
    pairs = []
    for tid in sorted(set(s25) & set(s31)):
        stats["total"] += 1
        r25, r31 = s25[tid], s31[tid]

        # Layer 0: winner 两源一致
        if r25.get("winner") != r31.get("winner"):
            stats["drop_disagree"] += 1
            continue
        winner = r25.get("winner")
        if winner not in ("A", "B"):
            stats["drop_no_winner"] += 1
            continue

        a = [(x + y) / 2.0 for x, y in zip(triplet(r25, "a"), triplet(r31, "a"))]
        b = [(x + y) / 2.0 for x, y in zip(triplet(r25, "b"), triplet(r31, "b"))]
        plus, minus = (a, b) if winner == "A" else (b, a)
        delta = [p - m for p, m in zip(plus, minus)]
        # 幅度只取正贡献(与 design 对齐): ||delta|| = sum_d ReLU(delta_d)
        norm = sum(max(x, 0.0) for x in delta)

        # Layer 0: 弱信号 ||delta|| ~ 0
        # ReLU 和=0 表示 winner 在所有维度上都没赢(只输或打平)->无方向信息,丢弃
        if norm <= min_norm:
            stats["drop_weak"] += 1
            continue

        n = n_map.get(tid)
        if n not in SUBJECT_COUNTS:
            stats["drop_no_N"] += 1
            continue

        pos = [max(x, 0.0) for x in delta]
        ssum = sum(pos)
        alpha = [x / ssum for x in pos] if ssum > 0 else [0.0, 0.0, 0.0]
        primary = DIMS[max(range(3), key=lambda i: delta[i])]
        sig = max(delta) - (sum(delta) / 3.0)

        pairs.append({
            "task_id": tid,
            "winner": winner,
            "N": n,
            "primary_dim": primary,
            "delta": dict(zip(DIMS, delta)),
            "delta_norm": norm,
            "S": sig,
            "alpha": dict(zip(DIMS, alpha)),
        })
        stats["kept"] += 1
    return pairs


def significance_filter(cell):
    # Layer 2: 保留 <=> S>=cell中位数 或 ||delta||>=cell中位数
    if len(cell) <= 1:
        return cell
    s_med = statistics.median(p["S"] for p in cell)
    n_med = statistics.median(p["delta_norm"] for p in cell)
    return [p for p in cell if p["S"] >= s_med or p["delta_norm"] >= n_med]


def sample_cells(pairs, per_cell, seed, stats_cell):
    rng = random.Random(seed)
    cells = defaultdict(list)
    for p in pairs:
        cells[(p["N"], p["primary_dim"])].append(p)

    selected = []
    for n in SUBJECT_COUNTS:
        for dim in DIMS:
            raw = cells.get((n, dim), [])
            filtered = significance_filter(raw)
            if len(filtered) > per_cell:
                picked = rng.sample(filtered, per_cell)
            else:
                picked = filtered  # 兜底:不足取全部
            selected.extend(picked)
            stats_cell[f"N{n}_{dim}"] = {
                "in_bucket": len(raw),
                "after_filter": len(filtered),
                "selected": len(picked),
            }
    rng.shuffle(selected)
    return selected


def size_label(total):
    if total % 1000 == 0:
        return f"{total // 1000}k"
    return str(total)


def run_one(pairs, total, seed, out_tmpl, stats_tmpl, layer0_stats):
    per_cell = total // (len(SUBJECT_COUNTS) * len(DIMS))
    label = size_label(total)
    out_path = out_tmpl.format(label=label)
    stats_path = stats_tmpl.format(label=label)

    stats_cell = {}
    selected = sample_cells(pairs, per_cell, seed, stats_cell)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "target_total": total,
        "per_cell_target": per_cell,
        "total_selected": len(selected),
        "total_paired": layer0_stats["total"],
        "drop_disagree": layer0_stats["drop_disagree"],
        "drop_no_winner": layer0_stats["drop_no_winner"],
        "drop_weak": layer0_stats["drop_weak"],
        "drop_no_N": layer0_stats["drop_no_N"],
        "kept_after_layer0": layer0_stats["kept"],
        "cells": stats_cell,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== sampling: {label} (target {total}) ===")
    print(f"per-cell target      : {per_cell}")
    print(f"total selected       : {len(selected)}")
    for n in SUBJECT_COUNTS:
        row = "  ".join(
            f"{dim[:3]}={stats_cell[f'N{n}_{dim}']['selected']:>4}"
            f"/{stats_cell[f'N{n}_{dim}']['after_filter']:>5}"
            for dim in DIMS
        )
        print(f"  N={n}:  {row}")
    print(f"output -> {out_path}")
    print(f"stats  -> {stats_path}")
    return summary


def find_data_base():
    """Locate mibe/Model_Training/data_v2 by walking up from this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        cand = os.path.join(d, "mibe", "Model_Training", "data_v2")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    # fall back to the original relative default
    return "../mibe/Model_Training/data_v2"


def main():
    ap = argparse.ArgumentParser()
    base = find_data_base()
    ap.add_argument("--gemini25", default=f"{base}/60k_LLM_Result/2_5_merged_sorted.jsonl")
    ap.add_argument("--gemini31", default=f"{base}/60k_LLM_Result/3_1_merged_sorted.jsonl")
    ap.add_argument("--prompt", default=f"{base}/prompt/train_60k_v13_2.jsonl",
                    help="prompt jsonl providing subject count N (total_entities) per task id")
    ap.add_argument("--out_tmpl", default="selected_pairs_{label}.jsonl")
    ap.add_argument("--stats_tmpl", default="sampling_stats_{label}.json")
    ap.add_argument("--sizes", default="2000,5000",
                    help="comma-separated target totals; one output per size")
    ap.add_argument("--min_norm", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s25 = load_jsonl(args.gemini25)
    s31 = load_jsonl(args.gemini31)
    n_map = load_subject_count(args.prompt)

    stats = defaultdict(int)
    pairs = precompute(s25, s31, n_map, args.min_norm, stats)

    print("=== Layer 0 preprocessing ===")
    print(f"total paired         : {stats['total']}")
    print(f"drop winner disagree : {stats['drop_disagree']}")
    print(f"drop no winner       : {stats['drop_no_winner']}")
    print(f"drop weak (||d||~0)  : {stats['drop_weak']}")
    print(f"drop no subject_count: {stats['drop_no_N']}")
    print(f"kept after layer0    : {stats['kept']}")

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    for total in sizes:
        run_one(pairs, total, args.seed, args.out_tmpl, args.stats_tmpl, stats)


if __name__ == "__main__":
    main()
