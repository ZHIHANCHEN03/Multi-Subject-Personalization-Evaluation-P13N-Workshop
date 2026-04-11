import os
import json
import csv
from collections import defaultdict

# paths
base_dir = "/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/data_v1"
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

for combo_id, anns in annotations_by_combo.items():
    if combo_id not in prompts_dict:
        print(f"Warning: {combo_id} not found in prompts.json")
        continue
        
    p_info = prompts_dict[combo_id]
    
    # Check preference consistency
    preferences = [ann['preference'] for ann in anns]
    if len(set(preferences)) > 1:
        print(f"Drop {combo_id}: Inconsistent preference {preferences}")
        continue
        
    # We assume model_a and model_b are consistent across annotations for the same combo_id
    model_a = anns[0]['model_a']
    model_b = anns[0]['model_b']
    
    # image paths
    img_a_path = f"./data_v1/round2/{model_a}/{combo_id}.jpg"
    img_b_path = f"./data_v1/round2/{model_b}/{combo_id}.jpg"
    
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
        
    annotator_results = []
    for ann in anns:
        annotator_results.append({
            "annotator_id": ann['email'],
            "preference": ann['preference'],
            "category_scores_A": {
                "existence": int(ann['a_existence']),
                "appearance": int(ann['a_appearance']),
                "interaction": int(ann['a_interaction'])
            },
            "category_scores_B": {
                "existence": int(ann['b_existence']),
                "appearance": int(ann['b_appearance']),
                "interaction": int(ann['b_interaction'])
            }
        })
        
    task_data = {
        "task_id": combo_id,
        "prompt": p_info['prompt'],
        "subject_count": int(anns[0]['n_total']),
        "subject_refs": subject_refs,
        "image_A_path": img_a_path,
        "image_B_path": img_b_path,
        "annotator_results": annotator_results,
        "metadata": {
            "source": "Human Annotation V1",
            "ratio_type": anns[0]['ratio_type']
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
