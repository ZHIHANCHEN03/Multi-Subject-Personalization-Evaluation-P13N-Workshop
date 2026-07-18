"""Split a manifest JSONL into K disjoint shards (round-robin, deterministic)."""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--shards", type=int, required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
lines = [l for l in open(args.data) if l.strip()]
buckets = [[] for _ in range(args.shards)]
for i, l in enumerate(lines):
    buckets[i % args.shards].append(l)
for i, b in enumerate(buckets):
    p = os.path.join(args.out_dir, f"shard_{i}.jsonl")
    open(p, "w").writelines(b)
    print(f"{p}: {len(b)} tasks")
