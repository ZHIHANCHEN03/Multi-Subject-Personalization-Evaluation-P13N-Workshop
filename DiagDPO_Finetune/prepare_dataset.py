import json
import os
import argparse

def filter_and_format_data(input_json_path, output_json_path, min_delta=0.1):
    """
    Filter the dataset based on delta scores and format it for fine-tuning.
    """
    print(f"Reading from {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    filtered_data = []
    
    for item in data:
        # 1. Align images based on preference
        if item.get("preference") == "A":
            winner_scores = item.get("category_scores_A", {})
            loser_scores = item.get("category_scores_B", {})
            image_w = item.get("image_A_path")
            image_l = item.get("image_B_path")
        elif item.get("preference") == "B":
            winner_scores = item.get("category_scores_B", {})
            loser_scores = item.get("category_scores_A", {})
            image_w = item.get("image_B_path")
            image_l = item.get("image_A_path")
        else:
            continue
            
        # 2. Calculate delta scores
        delta_E = winner_scores.get("existence", 0) - loser_scores.get("existence", 0)
        delta_A = winner_scores.get("appearance", 0) - loser_scores.get("appearance", 0)
        delta_I = winner_scores.get("interaction", 0) - loser_scores.get("interaction", 0)
        
        # 3. Hard Negative Mining
        winner_sum = winner_scores.get("existence", 0) + winner_scores.get("appearance", 0) + winner_scores.get("interaction", 0)
        
        if winner_sum > 0 and (delta_E > min_delta or delta_A > min_delta or delta_I > min_delta):
            formatted_item = {
                "task_id": item.get("task_id"),
                "prompt": item.get("prompt"),
                "image_w": image_w,
                "image_l": image_l,
                "delta_E": delta_E,
                "delta_A": delta_A,
                "delta_I": delta_I,
                "winner_scores": winner_scores,
                "loser_scores": loser_scores,
                "subject_refs": item.get("subject_refs", [])
            }
            filtered_data.append(formatted_item)
            
    print(f"Original pairs: {len(data)}")
    print(f"Filtered pairs (min_delta={min_delta}): {len(filtered_data)}")
    
    # Save the formatted dataset
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    print(f"Saved formatted data to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and format DPO dataset")
    parser.add_argument("--input", type=str, default="/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/data_v2/train_v2.json")
    parser.add_argument("--output", type=str, default="/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/DiagDPO_Finetune/data/dpo_train_filtered.json")
    parser.add_argument("--min_delta", type=float, default=0.1)
    
    args = parser.parse_args()
    filter_and_format_data(args.input, args.output, args.min_delta)
