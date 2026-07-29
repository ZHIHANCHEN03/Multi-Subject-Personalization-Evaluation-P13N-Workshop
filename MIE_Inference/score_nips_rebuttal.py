"""Score the Parallel Plays PP1 reference-extension study with MIE.

The study is a 3 x 4 grid: three reference sources (A real photos, B GPT-Image,
C Qwen-Image) crossed with four generators (Nano Banana, MOSAIC, Flux.2,
GPT-Image-1.5), each covering the same 216 prompts. This scores every cell with
one model load and emits a single JSON keyed so that any pivot (by generator, by
reference source, by subject count, by class tag) needs no further joins.

    MIE_CKPT=... python score_nips_rebuttal.py --root /workspace/nips_rebuttal \
        --out results/mie_scores.json

Design notes
------------
* Reference order is fixed as `people_names + object_names` for every cell. The
  heads were trained with references first and the candidate last, and scores are
  only comparable across cells if the ordering is identical -- so this is not a
  detail to vary.
* References always come from `refs_512/`. Group A only has 512, so 512 is the
  only resolution where all three groups line up.
* Results stream to a JSONL sidecar and are folded into the final JSON at the
  end. An interrupted run resumes from the sidecar instead of restarting.
* A missing candidate is recorded as an `error` row, never silently dropped:
  `experiment_a_real/flux2_out` is known to be missing ids 211-216, and that gap
  has to stay visible in the output.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from mie_loader import load_runtime, score

# (directory name, short label used in record ids)
GROUPS = [
    ("experiment_a_real", "a_real"),
    ("experiment_b_gpt_image_1", "b_gpt_image_1"),
    ("experiment_c_qwen", "c_qwen"),
]

# (subdirectory, short label, filename template)  -- padding differs per generator
GENERATORS = [
    ("nano_banana_out", "nano_banana", "{id:05d}.png"),
    ("mosaic_out/full", "mosaic", "{id}.jpg"),
    ("flux2_out", "flux2", "{id}.jpg"),
    ("gpt15_out", "gpt15", "{id:05d}.png"),
]

REF_DIR = "refs_512"
PROBE = "shared_data/probe_216_novel_v13.jsonl"

log = logging.getLogger("mie_nips")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    # Unsloth installs a root handler; without this every line is emitted twice.
    log.propagate = False
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def load_probe(root: Path) -> dict[int, dict]:
    path = root / PROBE
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            row = json.loads(line)
            tasks[int(row["id"])] = row
    log.info("loaded %d prompts from %s", len(tasks), path.name)
    return tasks


def build_plan(root: Path, probe: dict[int, dict]) -> list[dict]:
    """One entry per (group, generator, prompt id). Paths resolved, not yet checked."""
    plan = []
    for group_dir, group in GROUPS:
        for gen_dir, gen, template in GENERATORS:
            for pid in sorted(probe):
                row = probe[pid]
                slugs = list(row.get("people_names", [])) + list(row.get("object_names", []))
                plan.append({
                    "record_id": f"{group}__{gen}__{pid:06d}",
                    "group": group,
                    "group_dir": group_dir,
                    "generator": gen,
                    "id": pid,
                    "level": row["level"],
                    "class_tag": row["class_tag"],
                    "ratio_type": row["ratio_type"],
                    "n_humans": row["n_humans"],
                    "n_objects": row["n_objects"],
                    "prompt": row["prompt_en"],
                    "ref_slugs": slugs,
                    "ref_paths": [
                        str(root / group_dir / REF_DIR / f"{s}.png") for s in slugs
                    ],
                    "candidate_path": str(root / group_dir / gen_dir / template.format(id=pid)),
                    "n_refs": len(slugs),
                })
    return plan


def resume_from(sidecar: Path) -> dict[str, dict]:
    """Records already written by an earlier run, keyed by record_id."""
    if not sidecar.exists():
        return {}
    done = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated tail from a killed run
        done[row["record_id"]] = row
    if done:
        log.info("resuming: %d records already scored in %s", len(done), sidecar.name)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted nips_rebuttal_ref_extension_exp dir")
    ap.add_argument("--out", required=True, help="final JSON path")
    ap.add_argument("--checkpoint", default=None, help="defaults to $MIE_CKPT")
    ap.add_argument("--log", default=None, help="log file (default: alongside --out)")
    ap.add_argument("--limit", type=int, default=None, help="first N plan entries (smoke run)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the plan across N processes (GPU is idle at ~0%% with "
                         "one process, so parallel shards scale nearly linearly)")
    ap.add_argument("--shard", type=int, default=0, help="which shard this process runs (0-based)")
    ap.add_argument("--merge", action="store_true",
                    help="skip scoring; fold every shard sidecar into the final JSON")
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        ap.error(f"--shard must be in [0, {args.num_shards})")

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    # Each shard owns its own sidecar so parallel writers never interleave.
    # Built by string, not Path.with_suffix: with_suffix would treat the
    # ".shard0" we just appended as the suffix and replace it.
    sidecar = (out_path.parent / f"{out_path.stem}.shard{args.shard}.jsonl"
               if args.num_shards > 1 else out_path.with_suffix(".jsonl"))
    log_path = Path(args.log).expanduser().resolve() if args.log else \
        out_path.parent / "logs" / f"mie_scoring_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    log.info("root=%s", root)
    log.info("out=%s  sidecar=%s  log=%s", out_path, sidecar, log_path)

    probe = load_probe(root)
    full_plan = build_plan(root, probe)
    if args.limit:
        full_plan = full_plan[:args.limit]

    if args.merge:
        # Fold every shard's sidecar together, ordered by the full plan.
        done = {}
        pattern = f"{out_path.stem}.shard*.jsonl"
        shards = sorted(out_path.parent.glob(pattern)) or [out_path.with_suffix(".jsonl")]
        for s in shards:
            rows = resume_from(s)
            log.info("  %s: %d records", s.name, len(rows))
            done.update(rows)
        plan, todo = full_plan, []
        report = {"note": f"merged from {len(shards)} shard sidecar(s)"}
        # No model is loaded in merge mode, so recover the provenance the shards
        # logged. Without this the output JSON would not say which checkpoint
        # produced it, which is the one thing it must never lose.
        for log_file in sorted((out_path.parent / "logs").glob("*.log"), reverse=True):
            for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
                marker = "model ready: "
                if marker in line:
                    try:
                        report = {**json.loads(line.split(marker, 1)[1]), **report}
                    except json.JSONDecodeError:
                        continue
                    break
            if "checkpoint" in report:
                log.info("provenance recovered from %s", log_file.name)
                break
    else:
        # Cost per task scales with subject count, so shards must get a balanced
        # mix of levels or the run is held up by its slowest shard. A plain
        # stride does *not* balance here: levels cycle with period 4 and
        # gcd(4, 6 shards) = 2, so half the shards would draw only the heavy
        # levels. Shuffle with a fixed seed first, which is balanced for any
        # shard count and still fully deterministic.
        if args.num_shards > 1:
            shuffled = list(full_plan)
            random.Random(0).shuffle(shuffled)
            plan = shuffled[args.shard::args.num_shards]
        else:
            plan = full_plan
        done = resume_from(sidecar)
        todo = [p for p in plan if p["record_id"] not in done]
        log.info("shard %d/%d | plan=%d done=%d todo=%d  (full plan %d = %d groups x %d gen x %d prompts)",
                 args.shard, args.num_shards, len(plan), len(done), len(todo),
                 len(full_plan), len(GROUPS), len(GENERATORS), len(probe))

    if todo:
        runtime = load_runtime(args.checkpoint)
        report = runtime["report"]
        log.info("model ready: %s", json.dumps(report))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Rate is measured against the start of scoring, not per cell: the plan is
        # shuffled for shard balance, so consecutive tasks almost always belong to
        # different cells and a per-cell clock would reset every iteration and
        # report nonsense rates.
        run_started = time.time()
        ok = failed = 0
        with sidecar.open("a", encoding="utf-8") as sink:
            for i, entry in enumerate(todo, 1):
                record = {k: v for k, v in entry.items() if k != "group_dir"}
                try:
                    missing = [p for p in entry["ref_paths"] if not Path(p).exists()]
                    if missing:
                        raise FileNotFoundError(f"{len(missing)} reference(s) missing, "
                                                f"first: {missing[0]}")
                    if not Path(entry["candidate_path"]).exists():
                        raise FileNotFoundError(
                            f"no candidate image at {entry['candidate_path']}")
                    record["mie"] = score(
                        runtime,
                        image_path=entry["candidate_path"],
                        ref_paths=entry["ref_paths"],
                        prompt=entry["prompt"],
                    )
                    ok += 1
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    failed += 1
                    log.warning("%s -> %s", entry["record_id"], record["error"])

                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()  # keeps the run resumable if it dies here
                done[record["record_id"]] = record

                if i % 50 == 0 or i == len(todo):
                    rate = i / max(time.time() - run_started, 1e-9)
                    eta = (len(todo) - i) / max(rate, 1e-9) / 60
                    log.info("progress %d/%d  ok=%d failed=%d  %.2f task/s  eta~%.1f min",
                             i, len(todo), ok, failed, rate, eta)
        log.info("scoring finished: ok=%d failed=%d", ok, failed)
    elif not args.merge:
        log.info("nothing to score; folding existing sidecar into JSON")
        report = {"note": "no model loaded; all records resumed from sidecar"}

    # A shard writes only its own slice; --merge assembles the whole grid.
    order = full_plan if args.merge else plan
    records = [done[p["record_id"]] for p in order if p["record_id"] in done]
    n_ok = sum(1 for r in records if "mie" in r)
    n_failed = len(records) - n_ok

    payload = {
        "meta": {
            "checkpoint": report.get("checkpoint"),
            "base_model": report.get("base_model"),
            "mode": report.get("mode"),
            "device": report.get("device"),
            "weight_load_report": report,
            "root": str(root),
            "ref_source": REF_DIR,
            "ref_order": "people_names + object_names (fixed across all cells)",
            "groups": [g for _, g in GROUPS],
            "generators": [g for _, g, _ in GENERATORS],
            "n_prompts": len(probe),
            "n_records": len(records),
            "n_ok": n_ok,
            "n_failed": n_failed,
            "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "log": str(log_path),
        },
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("wrote %s  (%d records: %d ok, %d failed)",
             out_path, len(records), n_ok, n_failed)

    # Per-cell coverage, so a partially-generated cell is obvious at a glance.
    log.info("per-cell coverage:")
    for _, group in GROUPS:
        for _, gen, _ in GENERATORS:
            rows = [r for r in records
                    if r["group"] == group and r["generator"] == gen]
            good = sum(1 for r in rows if "mie" in r)
            flag = "" if good == len(probe) else f"  <-- {len(probe) - good} missing"
            log.info("  %-16s %-12s %3d/%d%s", group, gen, good, len(probe), flag)


if __name__ == "__main__":
    main()
