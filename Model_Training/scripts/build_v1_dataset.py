import os
import sys

# Fix Python Path for 'lens' module if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
from collections import defaultdict

# 动态获取相对于当前脚本的路径 (Dynamic relative paths)
# 这样无论代码被 clone 到什么操作系统的什么目录下（比如 Mac 或 Linux 服务器），都不会报错。
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "..", "data_v1"))

csv_path = os.path.join(base_dir, "annotations_20260409.csv")
prompts_json_path = os.path.join(base_dir, "contact-bench-assets", "prompts.json")
output_path = os.path.join(base_dir, "train_ready.json")

# read prompts
with open(prompts_json_path, 'r', encoding='utf-8') as f:
    prompts_data = json.load(f)

# mapping combo_id to prompt info
prompts_dict = {}
for p in prompts_data:
    prompts_dict[p['id']] = p

# read annotations
annotations_by_combo = defaultdict(list)
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        combo_id = row['combo_id']
        annotations_by_combo[combo_id].append(row)

result = []


def canonicalize_annotation(row, canonical_model_A, canonical_model_B):
    model_a = row["model_a"]
    model_b = row["model_b"]
    if {model_a, model_b} != {canonical_model_A, canonical_model_B}:
        raise ValueError(
            f"Inconsistent model pair for combo_id={row['combo_id']}: "
            f"({model_a}, {model_b}) vs canonical ({canonical_model_A}, {canonical_model_B})"
        )

    preference = row["preference"]
    if preference not in {"A", "B"}:
        raise ValueError(f"Unexpected preference={preference} for combo_id={row['combo_id']}")

    winner_model = model_a if preference == "A" else model_b

    scores_for_model_a_slot = {
        "existence": int(row["a_existence"]),
        "appearance": int(row["a_appearance"]),
        "interaction": int(row["a_interaction"]),
    }
    scores_for_model_b_slot = {
        "existence": int(row["b_existence"]),
        "appearance": int(row["b_appearance"]),
        "interaction": int(row["b_interaction"]),
    }

    if model_a == canonical_model_A and model_b == canonical_model_B:
        category_scores_A = scores_for_model_a_slot
        category_scores_B = scores_for_model_b_slot
    else:
        category_scores_A = scores_for_model_b_slot
        category_scores_B = scores_for_model_a_slot

    canonical_preference = "A" if winner_model == canonical_model_A else "B"

    return {
        "annotator_id": row["email"],
        "preference": canonical_preference,
        "category_scores_A": category_scores_A,
        "category_scores_B": category_scores_B,
        "winner_model": winner_model,
    }

for combo_id, anns in annotations_by_combo.items():
    if combo_id not in prompts_dict:
        print(f"Warning: {combo_id} not found in prompts.json")
        continue
        
    p_info = prompts_dict[combo_id]
    
    # Canonicalize the pair by REAL model identities instead of raw A/B slots.
    # Different annotators may see swapped presentation orders:
    # - row1: model_a=nano_banana, model_b=mosaic, preference=A
    # - row2: model_a=mosaic, model_b=nano_banana, preference=B
    # These are semantically identical and must NOT be dropped.
    canonical_models = sorted({anns[0]["model_a"], anns[0]["model_b"]})
    if len(canonical_models) != 2:
        print(f"Drop {combo_id}: expected exactly 2 distinct models, got {canonical_models}")
        continue
    canonical_model_A, canonical_model_B = canonical_models

    normalized_annotations = []
    try:
        for ann in anns:
            normalized_annotations.append(
                canonicalize_annotation(ann, canonical_model_A, canonical_model_B)
            )
    except ValueError as exc:
        print(f"Drop {combo_id}: {exc}")
        continue

    # Check preference consistency AFTER canonicalization by real model identity.
    preferences = [ann["preference"] for ann in normalized_annotations]
    if len(set(preferences)) > 1:
        print(f"Drop {combo_id}: Inconsistent canonical preference {preferences}")
        continue
    
    # image paths
    img_a_path = f"./data_v1/round2/{canonical_model_A}/{combo_id}.jpg"
    img_b_path = f"./data_v1/round2/{canonical_model_B}/{combo_id}.jpg"
    
    # subject refs
    subject_refs = []
    for h in p_info.get('humans', []):
        subject_refs.append({
            "id": h.split('.')[0],
            "image_path": f"./data_v1/contact-bench-assets/references/humans/{h}"
        })
    for o in p_info.get('objects', []):
        subject_refs.append({
            "id": o.split('.')[0],
            "image_path": f"./data_v1/contact-bench-assets/references/objects/{o}"
        })
        
    task_data = {
        "task_id": combo_id,
        "prompt": p_info['prompt'],
        "subject_count": int(anns[0]['n_total']),
        "subject_refs": subject_refs,
        "image_A_path": img_a_path,
        "image_B_path": img_b_path,
        "annotator_results": [
            {
                "annotator_id": ann["annotator_id"],
                "preference": ann["preference"],
                "category_scores_A": ann["category_scores_A"],
                "category_scores_B": ann["category_scores_B"],
            }
            for ann in normalized_annotations
        ],
        "metadata": {
            "source": "Human Annotation V1",
            "ratio_type": anns[0]['ratio_type'],
            "model_A_name": canonical_model_A,
            "model_B_name": canonical_model_B,
        }
    }
    result.append(task_data)

import random
random.seed(42)
random.shuffle(result)

total_len = len(result)
train_len = int(total_len * 0.6)
val_len = int(total_len * 0.2)

train_data = result[:train_len]
val_data = result[train_len:train_len + val_len]
test_data = result[train_len + val_len:]

train_path = os.path.join(base_dir, "train_v1.json")
val_path = os.path.join(base_dir, "val_v1.json")
test_path = os.path.join(base_dir, "test_v1.json")

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)
with open(val_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)
with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"Processed {total_len} valid tasks.")
print(f"Split data: Train ({len(train_data)}) -> {train_path}")
print(f"Split data: Val ({len(val_data)}) -> {val_path}")
print(f"Split data: Test ({len(test_data)}) -> {test_path}")
