#!/usr/bin/env python3
"""Build a submit-ready MIDC folder inside this repo: copy only needed files,
scrub all server/local paths, drop internal docs and reference PDFs.

Run from the repo root:
    python3 build_submission.py
Produces ./submission/ (gitignored). Zip that to upload to OpenReview.

Source : <repo root>
Dest   : <repo root>/submission
"""
import os, shutil, json, re

SRC = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(SRC, "submission")

# 1. what to copy (whitelist of top-level paths)
COPY = ["paper", "round1", "round1_1", "round2", "README.md", "ABSTRACT.md"]
# 2. within round2, exclude big/generated/local stuff
EXCLUDE_DIRS = {
    "results_flux2",  # will copy records but scrub paths (handled below)
}
# 3. exclude these patterns anywhere under copied dirs
EXCLUDE_PATTERNS = [
    "__pycache__", ".cache", "*.pyc", ".DS_Store",
    "human_eval_umo_vs_oneshot*", "human_eval.zip", "human_eval_key.json",
    "human_eval/pairs", "human_eval/manifest.js", "human_eval/ballot.csv",
]
# files that are internal-only (drop from submission)
DROP_FILES = {
    "PLAN.md",  # internal planning
    "round2/b1_reanalysis.py",  # has /workspace paths, reproducibility-only
}

SERVER_PREFIXES = [
    "/workspace/misc/",
    "/workspace/",
]

def should_exclude(path):
    base = os.path.basename(path)
    for pat in EXCLUDE_PATTERNS:
        if pat.endswith("*") and base.startswith(pat[:-1]):
            return True
        if "*" in pat:
            # glob-style
            import fnmatch
            if fnmatch.fnmatch(base, pat):
                return True
        elif base == pat:
            return True
    return False

def scrub_text(text):
    """Scrub server paths from text. Replace /workspace/misc/round1/../round2/
    and /workspace/misc/... with relative placeholders."""
    # Replace /workspace/misc/round1/../round2/ -> round2/
    text = text.replace("/workspace/misc/round1/../round2/", "round2/")
    text = text.replace("/workspace/misc/round1/../", "")
    text = text.replace("/workspace/misc/", "")
    text = text.replace("/workspace/", "")
    return text

def scrub_jsonl(path):
    """Rewrite records.jsonl: drop image_path (server path, images not bundled)."""
    tmp = path + ".tmp"
    n = 0
    with open(path) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                fout.write(line + "\n")
                continue
            if "image_path" in r:
                # rewrite to relative basename-only
                base = os.path.basename(r["image_path"])
                r["image_path"] = f"images/{base}"
            fout.write(json.dumps(r) + "\n")
            n += 1
    os.replace(tmp, path)
    return n

def copy_tree(src, dst):
    """Copy src -> dst, excluding patterns, scrubbing .sh/.py/.tex/.md/.jsonl."""
    for root, dirs, files in os.walk(src):
        # prune excluded dirs in-place
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        for f in files:
            sp = os.path.join(root, f)
            if should_exclude(sp):
                continue
            rel = os.path.relpath(sp, src)
            dp = os.path.join(dst, rel)
            # drop internal-only files
            if rel in DROP_FILES:
                continue
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            ext = os.path.splitext(f)[1].lower()
            if ext in (".sh", ".py", ".tex", ".md", ".json", ".jsonl", ".csv", ".html", ".js"):
                with open(sp) as fin, open(dp, "w") as fout:
                    fout.write(scrub_text(fin.read()))
                if f == "records.jsonl":
                    scrub_jsonl(dp)
            else:
                shutil.copy2(sp, dp)

def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    for item in COPY:
        sp = os.path.join(SRC, item)
        dp = os.path.join(DST, item)
        if os.path.isdir(sp):
            copy_tree(sp, dp)
        else:
            if item in DROP_FILES:
                continue
            with open(sp) as fin, open(dp, "w") as fout:
                fout.write(scrub_text(fin.read()))
    print(f"Built {DST}")

if __name__ == "__main__":
    main()
