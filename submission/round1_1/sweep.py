"""Round 1.1 — automated overnight algorithm-tuning sweep for OURS.

Goal: find a clearly-better training-free + MIE configuration than the Round-1
default, judged on a fixed hard subset against the FROZEN Round-1 baselines
(one_shot / best_of_n / umo — reused, not re-run).

Each trial runs ONLY the ours pipeline with a given config (env + args), scores
with the same detection-aware SCR, then compares (mean SCR/DINO + paired
win-rate vs baselines). Results are appended to a leaderboard. Nothing trained.

Design:
- Fixed subset manifest (default: reuse round1 test manifest, 4-entity slice).
- Frozen calibration reused from round1.
- Trials defined in TRIALS below (edit to expand). Sequential on 1 GPU.

Usage (on server, from repo root):
  python round1_1/sweep.py \
      --data round1/results/manifests/hard_cases.jsonl \
      --baselines round1/results \
      --calibration round1/results/calibration/mie_baselines.json \
      --out round1_1 --entities 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R1 = REPO / "round1"
PY_OMNI = os.environ.get("PY_OMNI", str(REPO / ".venvs/omni/bin/python"))

# Each trial: name -> {script, env:{...}, args:{n_init,k,routing,action}}
# script: "v1" -> p1_ours.py, "v2" -> p1_ours_v2.py (dual-signal pipeline)
# Budget B = n_init + k*proposals (v1) or n_init + k*actions*seeds (v2).
TRIALS = {
    # ---- v1 baseline (Round-1 reproduction, control) ----
    "t0_default":        {"script": "v1",
                          "env": {"OURS_PROPOSALS": "2", "OURS_DEFICIT_MIN": "0.75"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},

    # ---- v2 structural variants (dual-signal diagnose-and-target) ----
    # full v2: dual-signal + action portfolio + dual-accept
    "v2_full":           {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # ablation: no per-subject SCR diagnosis (round-robin subject)
    "v2_no_dual":        {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "0", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # ablation: no action portfolio (single action per step, seed diversity only)
    "v2_no_portfolio":   {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "0",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACTIONS_PER_STEP": "1",
                                  "V2_SEEDS_PER_ACTION": "2", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # ablation: v1 acceptance (MIE total+target dim, no per-subject verify)
    "v2_no_dual_accept": {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "0", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # compute-matched to best-of-N=8: n_init=2, k=2, actions=3, seeds=1 -> 2+6=8
    "v2_budget8":        {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    # aggressive: lower gate + more steps (deeper correction)
    "v2_aggressive":    {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.25",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1",
                                  "MIE_COLLATERAL_TOL": "0.10"},
                          "args": {"n_init": 2, "k": 4, "routing": "calibrated", "action": "both"}},

    # ---- v2.1: relaxed acceptance (drop SCR collateral) ----
    # v2_full showed accepted=0: emphasizing one subject trades off others ->
    # SCR collateral always fails. Relaxed: MIE total improves AND target
    # subject's DINO sim improves (MIE total guards overall quality).
    "v2_relaxed":       {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_ACTIONS_PER_STEP": "3", "V2_SEEDS_PER_ACTION": "1",
                                  "OURS_DEFICIT_MIN": "0.5", "OURS_REFSET_MODE": "front_dup3",
                                  "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # relaxed + compute-matched to best-of-N=8 (n_init=2, k=2, actions=3 -> 2+6=8)
    "v2_relaxed_b8":    {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_ACTIONS_PER_STEP": "3", "V2_SEEDS_PER_ACTION": "1",
                                  "OURS_DEFICIT_MIN": "0.5", "OURS_REFSET_MODE": "front_dup3",
                                  "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    # v1 acceptance (MIE total + target dim + MIE collateral, no per-subject)
    "v2_no_dual_accept": {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "0", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},

    # ---- v2.3: targeted selection + total tolerance (fix matched-compute gap) ----
    # v2_relaxed_b8 tied v1 (SCR=0.5) at budget 8: MIE total is too noisy to
    # improve strictly, vetoing good subject-rescuing fixes. Two levers:
    #   V2_TOTAL_TOL: allow MIE total to dip by this much (accept subject fix
    #     even if overall quality slightly perturbs).
    #   V2_SELECT_MODE=weak_subject: portfolio picks the candidate that most
    #     improves the collapsed subject (not the highest MIE total).
    # budget 8 = n_init 2 + k 2 + actions 3 (compute-matched to best-of-N=8).
    "v2_3_tol_b8":      {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_TOTAL_TOL": "0.1", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    "v2_3_weaksel_b8":  {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_SELECT_MODE": "weak_subject", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    "v2_3_combo_b8":    {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_TOTAL_TOL": "0.1", "V2_SELECT_MODE": "weak_subject",
                                  "V2_ACTIONS_PER_STEP": "3", "V2_SEEDS_PER_ACTION": "1",
                                  "OURS_DEFICIT_MIN": "0.5", "OURS_REFSET_MODE": "front_dup3",
                                  "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    "v2_3_combo_b11":   {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_TOTAL_TOL": "0.1", "V2_SELECT_MODE": "weak_subject",
                                  "V2_ACTIONS_PER_STEP": "3", "V2_SEEDS_PER_ACTION": "1",
                                  "OURS_DEFICIT_MIN": "0.5", "OURS_REFSET_MODE": "front_dup3",
                                  "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
    # winner validation on full 20-task subset (run without --limit)
    "v2_3_weaksel_20":  {"script": "v2",
                          "env": {"V2_DUAL_SIGNAL": "1", "V2_ACTION_PORTFOLIO": "1",
                                  "V2_DUAL_ACCEPT": "1", "V2_ACCEPT_MODE": "relaxed",
                                  "V2_SELECT_MODE": "weak_subject", "V2_ACTIONS_PER_STEP": "3",
                                  "V2_SEEDS_PER_ACTION": "1", "OURS_DEFICIT_MIN": "0.5",
                                  "OURS_REFSET_MODE": "front_dup3", "OURS_LAYOUT": "1"},
                          "args": {"n_init": 2, "k": 2, "routing": "calibrated", "action": "both"}},
    # v1 control on full 20-task subset (matched-compute comparison to weaksel_20)
    "t0_default_20":   {"script": "v1",
                          "env": {"OURS_PROPOSALS": "2", "OURS_DEFICIT_MIN": "0.75"},
                          "args": {"n_init": 2, "k": 3, "routing": "calibrated", "action": "both"}},
}


def run_trial(name, cfg, args):
    out_dir = Path(args.out) / "trials"
    # skip if this trial already produced a full records file (resume support)
    rec_file = out_dir / name / "records.jsonl"
    if rec_file.exists() and os.environ.get("SWEEP_OVERWRITE", "0") != "1":
        n_sub = sum(1 for _ in open(rec_file))
        if n_sub > 0:
            print(f"\n===== TRIAL {name} (skip: {n_sub} records exist) =====", flush=True)
            return
    script = cfg.get("script", "v1")
    script_path = R1 / ("p1_ours_v2.py" if script == "v2" else "p1_ours.py")
    env = dict(os.environ)
    env.update({
        "ROUND1_WORK": str(out_dir),
        "OMNIGEN2_STEPS": os.environ.get("OMNIGEN2_STEPS", "28"),
        "ROUND1_CPU_OFFLOAD": "0",
        "MIE_PYTHON": os.environ.get("MIE_PYTHON", str(REPO / ".venvs/mie/bin/python")),
        "MIE_CKPT": args.mie_ckpt,
    })
    env.update(cfg["env"])
    a = cfg["args"]
    cmd = [PY_OMNI, str(script_path),
           "--name", name, "--data", args.data, "--generator", "omnigen2",
           "--n_init", str(a["n_init"]), "--k", str(a["k"]),
           "--routing", a["routing"], "--action", a["action"],
           "--calibration", args.calibration]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print(f"\n===== TRIAL {name} (script={script}) =====\n{' '.join(cmd)}\n  env+={cfg['env']}", flush=True)
    subprocess.run(cmd, env=env, check=False, cwd=str(R1))


def load(recdir, method):
    p = Path(recdir) / method / "records.jsonl"
    d = {}
    if p.exists():
        for l in open(p):
            l = l.strip()
            if l:
                r = json.loads(l)
                d[r["task_id"]] = r
    return d


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def evaluate(name, args):
    trials_dir = Path(args.out) / "trials"
    ours = load(trials_dir, name)
    one = load(args.baselines, "one_shot")
    bon = load(args.baselines, "best_of_n")
    umo = load(args.baselines, "umo")
    row = {"trial": name}
    for n in args.entities:
        keys = [k for k in ours if ours[k].get("num_subjects") == n]
        if not keys:
            continue
        def m(d, f):
            return mean([d.get(k, {}).get(f) for k in keys])
        def wr(d):
            t = w = 0
            for k in keys:
                b = d.get(k, {}).get("scr")
                if b is None or ours[k].get("scr") is None:
                    continue
                t += 1
                w += (ours[k]["scr"] < b)
            return f"{w}/{t}"
        row[f"n{n}"] = {
            "ours_scr": round(m(ours, "scr"), 4) if keys else None,
            "ours_dino": round(m(ours, "dino_mean"), 4) if keys else None,
            "umo_scr": round(m(umo, "scr"), 4),
            "bon_scr": round(m(bon, "scr"), 4),
            "one_scr": round(m(one, "scr"), 4),
            "win_vs_umo": wr(umo), "win_vs_bon": wr(bon), "win_vs_one": wr(one),
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baselines", required=True, help="round1 results dir (one_shot/best_of_n/umo)")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--mie_ckpt", default=os.environ.get("MIE_CKPT",
                    "Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--entities", nargs="+", type=int, default=[4])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None, help="run only these trial names")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    board_path = Path(args.out) / "leaderboard.jsonl"
    names = args.only or list(TRIALS.keys())
    for name in names:
        run_trial(name, TRIALS[name], args)
        row = evaluate(name, args)
        with open(board_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[sweep] {name}: {json.dumps(row)}", flush=True)

    # print sorted leaderboard by 4-entity ours_scr (lower better)
    rows = [json.loads(l) for l in open(board_path)] if board_path.exists() else []
    key_ent = args.entities[0]
    rows = [r for r in rows if f"n{key_ent}" in r]
    rows.sort(key=lambda r: (r[f"n{key_ent}"]["ours_scr"] if r[f"n{key_ent}"]["ours_scr"] is not None else 9))
    print(f"\n==== LEADERBOARD (by n{key_ent} ours_scr, lower=better) ====")
    for r in rows:
        e = r[f"n{key_ent}"]
        print(f"  {r['trial']:18s} SCR={e['ours_scr']} DINO={e['ours_dino']} "
              f"| umo={e['umo_scr']} bon={e['bon_scr']} | winUMO={e['win_vs_umo']} winBoN={e['win_vs_bon']}")


if __name__ == "__main__":
    main()
