import json
import collections
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent
FILE_25 = RESULTS_DIR / "2_5_merged_sorted.jsonl"
FILE_31 = RESULTS_DIR / "3_1_merged_sorted.jsonl"

def load_jsonl(path):
    data = {}
    if not path.exists(): return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "task_id" in obj:
                data[str(obj["task_id"])] = obj
    return data

res_25 = load_jsonl(FILE_25)
res_31 = load_jsonl(FILE_31)

# Only consider intersection tasks
common_tids = set(res_25.keys()).intersection(set(res_31.keys()))

stats = {
    "total_common": len(common_tids),
    "pref_agree": 0,
    "exist_agree": 0,
    "app_agree": 0,
    "int_agree": 0,
    "strict_agree": 0,
    
    # Subgroups
    "by_subj_count": collections.defaultdict(lambda: {"total": 0, "pref_agree": 0}),
    "by_class_tag": collections.defaultdict(lambda: {"total": 0, "pref_agree": 0})
}

for tid in common_tids:
    r25 = res_25[tid]
    r31 = res_31[tid]
    
    # Preference
    w25 = r25.get("winner")
    w31 = r31.get("winner")
    pref_match = (w25 == w31 and w25 in ["A", "B"])
    if pref_match:
        stats["pref_agree"] += 1
        
    # Sub-scores (combine A and B for simpler metric, or treat each separately. Here we treat "both A and B match" as agreement for that dimension)
    exist_match = (r25.get("a_existence") == r31.get("a_existence")) and (r25.get("b_existence") == r31.get("b_existence"))
    app_match = (r25.get("a_appearance") == r31.get("a_appearance")) and (r25.get("b_appearance") == r31.get("b_appearance"))
    int_match = (r25.get("a_interaction") == r31.get("a_interaction")) and (r25.get("b_interaction") == r31.get("b_interaction"))
    
    if exist_match: stats["exist_agree"] += 1
    if app_match: stats["app_agree"] += 1
    if int_match: stats["int_agree"] += 1
    
    if pref_match and exist_match and app_match and int_match:
        stats["strict_agree"] += 1
        
    # Metadata groups
    subj_count = r25.get("subject_count", "Unknown")
    class_tag = r25.get("metadata", {}).get("class_tag", "Unknown")
    
    stats["by_subj_count"][subj_count]["total"] += 1
    if pref_match: stats["by_subj_count"][subj_count]["pref_agree"] += 1
        
    stats["by_class_tag"][class_tag]["total"] += 1
    if pref_match: stats["by_class_tag"][class_tag]["pref_agree"] += 1

print("=== OVERALL AGREEMENT ===")
print(f"Total Valid Overlap: {stats['total_common']}")
print(f"Preference Agreement: {stats['pref_agree']} ({(stats['pref_agree']/stats['total_common']*100):.2f}%)")
print(f"Strict Agreement (Pref + all sub-scores): {stats['strict_agree']} ({(stats['strict_agree']/stats['total_common']*100):.2f}%)")
print("\n=== SUB-SCORE AGREEMENT (Both A & B match) ===")
print(f"Existence: {stats['exist_agree']} ({(stats['exist_agree']/stats['total_common']*100):.2f}%)")
print(f"Appearance: {stats['app_agree']} ({(stats['app_agree']/stats['total_common']*100):.2f}%)")
print(f"Interaction: {stats['int_agree']} ({(stats['int_agree']/stats['total_common']*100):.2f}%)")

print("\n=== AGREEMENT BY SUBJECT COUNT ===")
for k in sorted(stats["by_subj_count"].keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 999):
    v = stats["by_subj_count"][k]
    print(f"Count {k}: {v['pref_agree']} / {v['total']} ({(v['pref_agree']/v['total']*100):.2f}%)")

print("\n=== AGREEMENT BY CLASS TAG ===")
for k in sorted(stats["by_class_tag"].keys()):
    v = stats["by_class_tag"][k]
    print(f"Tag {k}: {v['pref_agree']} / {v['total']} ({(v['pref_agree']/v['total']*100):.2f}%)")
