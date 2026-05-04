import os
import json
import pandas as pd
from glob import glob

def build_evaluation_manifest(dataset_root, output_dir):
    """
    Scans the Dataset_Eval directory and builds TWO JSONL manifests:
    - v10_manifest.jsonl
    - v13_manifest.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    v10_manifest = []
    v13_manifest = []
    
    # 1. Parse v10 Test Set (Nano & Mosaic)
    v10_dir = os.path.join(dataset_root, "v10_test", "v10_test")
    if os.path.exists(v10_dir):
        jsonl_path = os.path.join(v10_dir, "test_1.5k_v10.jsonl")
        ref_dir = os.path.join(v10_dir, "inference_images_v10")
        
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                v10_prompts = [json.loads(line) for line in f]
                
            for p in v10_prompts:
                prompt_id = p.get('id', p.get('idx', 'unknown'))
                prompt_text = p.get('prompt', '')
                # Find all corresponding refs
                refs = glob(os.path.join(ref_dir, "*.jpg")) # Simplified, needs exact matching logic based on your jsonl
                
                # 核心逻辑变更：按 PAIR (对) 来组织数据
                # v10 的 Pair 是 Mosaic vs Nano
                mosaic_img = os.path.join(v10_dir, "mosaic_images", f"{prompt_id}.jpg")
                nano_img = os.path.join(v10_dir, "nano_banana_v10_full1500_512", f"{prompt_id}.png")
                
                if os.path.exists(mosaic_img) and os.path.exists(nano_img):
                    v10_manifest.append({
                        "task_id": f"v10_pair_{prompt_id}",
                        "prompt": prompt_text,
                        "reference_images": refs,
                        "pair": {
                            "model_A_name": "mosaic",
                            "model_A_image": mosaic_img,
                            "model_B_name": "nano",
                            "model_B_image": nano_img
                        }
                    })

    # 2. Parse v13.2 Test Set (GLM, Flux, GPT, SeeDream)
    v13_dir = os.path.join(dataset_root, "v13_2_1.26k_evl", "v13_2_1.26k_evl")
    if os.path.exists(v13_dir):
        jsonl_path = os.path.join(v13_dir, "sampled_prompts.jsonl")
        ref_dir = os.path.join(v13_dir, "all_refs_noindex_v13.2")
        
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                v13_prompts = [json.loads(line) for line in f]
                
            for p in v13_prompts:
                prompt_id = str(p.get('id', p.get('idx', 'unknown')))
                prompt_text = p.get('prompt', '')
                # Note: Exact ref matching logic will depend on prompt metadata or filenames.
                # Assuming here we grab all for the specific ID or a generic list if mapping is complex.
                refs = glob(os.path.join(ref_dir, "*.jpg"))[:2] # Placeholder: Should map to actual used refs
                
                # 核心逻辑变更：v13 往往是两两比较 (比如 GLM vs Flux, 或 GPT vs SeeDream)
                # 这里为了完全遵循 Pair 结构，我们将同一个 prompt_id 下找到的多个模型的图，组合成 Pair。
                # 你可以根据实际的人类评测标注逻辑 (比如是 GLM 碰 Flux，还是 4 个模型互相碰) 来调整组合方式。
                # 假设这里是把 4 个模型都找到，然后组成 2 个独立的 Pair 放入 manifest。
                
                glm_img = os.path.join(v13_dir, "GLM", f"{prompt_id}.jpg")
                flux_img = os.path.join(v13_dir, "flux", "flux2_klein_9b_kv_1260_20260423", f"{prompt_id}.jpg")
                gpt_imgs = glob(os.path.join(v13_dir, "gpt-image-1.5_high", f"id_{prompt_id}_*.jpg"))
                seedream_imgs = glob(os.path.join(v13_dir, "seedream4.5", "ark_seedream45_full1260_20260424_jpeg_only", f"*_id_{prompt_id}.jpg"))
                
                # Pair 1: GLM vs Flux
                if os.path.exists(glm_img) and os.path.exists(flux_img):
                    v13_manifest.append({
                        "task_id": f"v13_pair_glm_vs_flux_{prompt_id}",
                        "prompt": prompt_text,
                        "reference_images": refs,
                        "pair": {
                            "model_A_name": "glm",
                            "model_A_image": glm_img,
                            "model_B_name": "flux",
                            "model_B_image": flux_img
                        }
                    })
                    
                # Pair 2: GPT vs SeeDream
                if gpt_imgs and seedream_imgs:
                    v13_manifest.append({
                        "task_id": f"v13_pair_gpt_vs_seedream_{prompt_id}",
                        "prompt": prompt_text,
                        "reference_images": refs,
                        "pair": {
                            "model_A_name": "gpt-image-1.5",
                            "model_A_image": gpt_imgs[0],
                            "model_B_name": "seedream",
                            "model_B_image": seedream_imgs[0]
                        }
                    })

    # Save to master manifests
    v10_out = os.path.join(output_dir, "v10_manifest.jsonl")
    with open(v10_out, 'w') as f:
        for item in v10_manifest:
            f.write(json.dumps(item) + '\n')
            
    v13_out = os.path.join(output_dir, "v13_manifest.jsonl")
    with open(v13_out, 'w') as f:
        for item in v13_manifest:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved {len(v10_manifest)} tasks to {v10_out}")
    print(f"Saved {len(v13_manifest)} tasks to {v13_out}")
    return v10_manifest, v13_manifest

if __name__ == '__main__':
    # 动态获取路径，兼容不同的部署位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # 因为 Dataset_Eval 可能在 repo 根目录下，也可能在 /workspace 下，这里优先找 repo 下的
    dataset_eval_dir = os.path.join(repo_root, "Dataset_Eval")
    
    # Fallback to /workspace/Dataset_Eval if not found in repo
    if not os.path.exists(dataset_eval_dir):
        dataset_eval_dir = "/workspace/Dataset_Eval"
        
    build_evaluation_manifest(dataset_eval_dir, repo_root)
