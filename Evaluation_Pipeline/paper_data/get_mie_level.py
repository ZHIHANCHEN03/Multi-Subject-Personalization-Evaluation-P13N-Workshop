import json

MODEL_NAME_MAP = {
    "mosaic_v10": "mosaic",
    "nano_banana_v10": "nano_banana",
    "flux2_klein_9b_kv": "flux",
    "gpt_image_1_5": "gpt-image-1.5",
    "seedream45": "seedream4.5",
    "seedream4.5": "seedream4.5",
    "seedream_4_5": "seedream4.5",
    "glm": "glm",
    "flux": "flux",
    "mosaic": "mosaic",
    "nano_banana": "nano_banana",
}

def normalize_model_name(name: str) -> str:
    key = (name or "").strip()
    return MODEL_NAME_MAP.get(key, key)

human_path = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/paper_data/section_4_1_2_mib_gold/gold_human_annotation_groups.jsonl'
mie_path = '/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/paper_data/section_4_2_2_mie_alignment/source_jsonl/auto_manifest_lens_scores_all6__qwen35_4b_lora_layer.jsonl'

humans = {}
with open(human_path) as f:
    for line in f:
        row = json.loads(line)
        if row['preference_consistent']:
            base_id = str(row['base_id'])
            humans[base_id] = {
                'level': row['level'],
                'winner': normalize_model_name(row['human_winner_model']),
                'dataset': row['dataset'],
                'pair_models': {normalize_model_name(m) for m in row['pair_models']}
            }

mie_pairs = {}
with open(mie_path) as f:
    for line in f:
        row = json.loads(line)
        pid = row['pair_id']
        letter = row['id'].split('::')[1]
        if pid not in mie_pairs:
            mie_pairs[pid] = {}
        
        base_id = pid.split('_')[-1]
        dataset = pid.split('_')[0]
        mie_pairs[pid][letter] = {
            'score': row['preference_raw_score'],
            'model_name': normalize_model_name(row['gen_image_model_name']),
            'base_id': base_id,
            'dataset': dataset
        }

correct_by_level_v10 = {2: 0, 4: 0, 6: 0, 8: 0}
total_by_level_v10 = {2: 0, 4: 0, 6: 0, 8: 0}

for pid, pair in mie_pairs.items():
    if 'A' in pair and 'B' in pair:
        base_id = pair['A']['base_id']
        if base_id in humans and humans[base_id]['dataset'] == 'v10' and pair['A']['dataset'] == 'v10':
            h = humans[base_id]
            level = h['level']
            
            models_in_pair = {pair['A']['model_name'], pair['B']['model_name']}
            if models_in_pair != h['pair_models']:
                continue

            pred_letter = 'A' if pair['A']['score'] >= pair['B']['score'] else 'B'
            pred_model = pair[pred_letter]['model_name']
            
            total_by_level_v10[level] += 1
            if pred_model == h['winner']:
                correct_by_level_v10[level] += 1

print("\nv10 only:")
for lvl in [2, 4, 6, 8]:
    acc = correct_by_level_v10[lvl] / total_by_level_v10[lvl] if total_by_level_v10[lvl] else 0
    print(f"L{lvl}: {acc:.3f} ({correct_by_level_v10[lvl]}/{total_by_level_v10[lvl]})")

