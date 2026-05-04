import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def load_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_manifest(manifest, manifest_path):
    with open(manifest_path, 'w') as f:
        for item in manifest:
            f.write(json.dumps(item) + '\n')

def init_metrics_dict(item, metric_name):
    if 'classic_metrics' not in item:
        item['classic_metrics'] = {}
    if metric_name not in item['classic_metrics']:
        item['classic_metrics'][metric_name] = {}
    return item

# ==========================================
# 1. CLIP-T (Text-to-Image Alignment)
# ==========================================
def run_clip_t(manifest):
    print("Loading Real CLIP Model for CLIP-T...")
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "clip_t")
        if 'model_A_score' in item['classic_metrics']['clip_t']: continue
        print(f"[{idx}/{len(manifest)}] CLIP-T: {item['task_id']}")
        
        text = item['prompt']
        img_a = Image.open(item['pair']['model_A_image']).convert("RGB")
        img_b = Image.open(item['pair']['model_B_image']).convert("RGB")
        
        with torch.no_grad():
            inputs_a = processor(text=[text], images=img_a, return_tensors="pt", padding=True).to("cuda")
            inputs_b = processor(text=[text], images=img_b, return_tensors="pt", padding=True).to("cuda")
            
            # CLIP logits_per_image is cosine similarity * 100. We normalize it back.
            score_a = model(**inputs_a).logits_per_image.item() / 100.0
            score_b = model(**inputs_b).logits_per_image.item() / 100.0
            
        item['classic_metrics']['clip_t'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 2. CLIP-I (Image-to-Image Identity)
# ==========================================
def run_clip_i(manifest):
    print("Loading Real CLIP Model for CLIP-I...")
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "clip_i")
        if 'model_A_score' in item['classic_metrics']['clip_i']: continue
        print(f"[{idx}/{len(manifest)}] CLIP-I: {item['task_id']}")
        
        refs = [Image.open(r).convert("RGB") for r in item['reference_images']]
        img_a = Image.open(item['pair']['model_A_image']).convert("RGB")
        img_b = Image.open(item['pair']['model_B_image']).convert("RGB")
        
        with torch.no_grad():
            ref_inputs = processor(images=refs, return_tensors="pt").to("cuda")
            ref_feats = model.get_image_features(**ref_inputs)
            ref_feats = ref_feats / ref_feats.norm(p=2, dim=-1, keepdim=True)
            ref_mean = ref_feats.mean(dim=0, keepdim=True) # Average reference embedding
            
            feat_a = model.get_image_features(**processor(images=img_a, return_tensors="pt").to("cuda"))
            feat_b = model.get_image_features(**processor(images=img_b, return_tensors="pt").to("cuda"))
            feat_a = feat_a / feat_a.norm(p=2, dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(p=2, dim=-1, keepdim=True)
            
            score_a = torch.cosine_similarity(ref_mean, feat_a).item()
            score_b = torch.cosine_similarity(ref_mean, feat_b).item()
            
        item['classic_metrics']['clip_i'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 3. DINOv2 (Structural Alignment)
# ==========================================
def run_dinov2(manifest):
    print("Loading Real DINOv2 Model...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to("cuda")
    
    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "dinov2")
        if 'model_A_score' in item['classic_metrics']['dinov2']: continue
        print(f"[{idx}/{len(manifest)}] DINOv2: {item['task_id']}")
        
        refs = [Image.open(r).convert("RGB") for r in item['reference_images']]
        img_a = Image.open(item['pair']['model_A_image']).convert("RGB")
        img_b = Image.open(item['pair']['model_B_image']).convert("RGB")
        
        with torch.no_grad():
            ref_inputs = processor(images=refs, return_tensors="pt").to("cuda")
            ref_cls = model(**ref_inputs).last_hidden_state[:, 0, :] # CLS token
            ref_mean = ref_cls.mean(dim=0, keepdim=True)
            
            cls_a = model(**processor(images=img_a, return_tensors="pt").to("cuda")).last_hidden_state[:, 0, :]
            cls_b = model(**processor(images=img_b, return_tensors="pt").to("cuda")).last_hidden_state[:, 0, :]
            
            score_a = F.cosine_similarity(ref_mean, cls_a).item()
            score_b = F.cosine_similarity(ref_mean, cls_b).item()
            
        item['classic_metrics']['dinov2'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 4. ArcFace (Face Identity Preservation)
# ==========================================
def run_arcface(manifest):
    print("Loading Real ArcFace (InsightFace) Model...")
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640)) # 0 = CUDA
    
    def get_face_emb(img_path):
        img = cv2.imread(img_path)
        faces = app.get(img)
        if len(faces) == 0: return None
        # Return the largest face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        return torch.tensor(faces[0].embedding).cuda()

    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "arcface")
        if 'model_A_score' in item['classic_metrics']['arcface']: continue
        print(f"[{idx}/{len(manifest)}] ArcFace: {item['task_id']}")
        
        ref_embs = [get_face_emb(r) for r in item['reference_images']]
        ref_embs = [e for e in ref_embs if e is not None]
        
        if len(ref_embs) == 0:
            # No faces in references, skip
            item['classic_metrics']['arcface'] = {"model_A_score": 0.0, "model_B_score": 0.0}
            continue
            
        ref_mean = torch.stack(ref_embs).mean(dim=0, keepdim=True)
        ref_mean = ref_mean / ref_mean.norm(p=2, dim=-1, keepdim=True)
        
        emb_a = get_face_emb(item['pair']['model_A_image'])
        emb_b = get_face_emb(item['pair']['model_B_image'])
        
        score_a = F.cosine_similarity(ref_mean, emb_a.unsqueeze(0)).item() if emb_a is not None else 0.0
        score_b = F.cosine_similarity(ref_mean, emb_b.unsqueeze(0)).item() if emb_b is not None else 0.0
        
        item['classic_metrics']['arcface'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 5. ImageReward (Human Preference)
# ==========================================
def run_image_reward(manifest):
    print("Loading Real ImageReward Model...")
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0").to("cuda")
    
    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "image_reward")
        if 'model_A_score' in item['classic_metrics']['image_reward']: continue
        print(f"[{idx}/{len(manifest)}] ImageReward: {item['task_id']}")
        
        prompt = item['prompt']
        with torch.no_grad():
            score_a = model.score(prompt, item['pair']['model_A_image'])
            score_b = model.score(prompt, item['pair']['model_B_image'])
            
        item['classic_metrics']['image_reward'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 6. SCR (Subject Collapse Rate via GroundingDINO)
# ==========================================
def run_scr(manifest):
    print("Loading Real DINOv2 Model for SCR...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to("cuda")
    
    tau = 0.4  # as per paper

    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "scr")
        if 'model_A_score' in item['classic_metrics']['scr']: continue
        print(f"[{idx}/{len(manifest)}] SCR: {item['task_id']}")
        
        refs = [Image.open(r).convert("RGB") for r in item['reference_images']]
        img_a = Image.open(item['pair']['model_A_image']).convert("RGB")
        img_b = Image.open(item['pair']['model_B_image']).convert("RGB")
        
        with torch.no_grad():
            ref_inputs = processor(images=refs, return_tensors="pt").to("cuda")
            ref_cls = model(**ref_inputs).last_hidden_state[:, 0, :] # [N, D]
            
            cls_a = model(**processor(images=img_a, return_tensors="pt").to("cuda")).last_hidden_state[:, 0, :] # [1, D]
            cls_b = model(**processor(images=img_b, return_tensors="pt").to("cuda")).last_hidden_state[:, 0, :] # [1, D]
            
            sim_a = F.cosine_similarity(ref_cls, cls_a) # [N]
            sim_b = F.cosine_similarity(ref_cls, cls_b) # [N]
            
            # Collapse if sim < tau
            score_a = (sim_a < tau).float().mean().item()
            score_b = (sim_b < tau).float().mean().item()
            
        item['classic_metrics']['scr'] = {"model_A_score": round(score_a, 4), "model_B_score": round(score_b, 4)}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

# ==========================================
# 7. REFVNLI (Reference Visual NLI Proxy via Qwen2-VL 2B)
# ==========================================
def run_refvnli(manifest):
    print("Loading Qwen2-VL for REFVNLI Proxy...")
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(
        "unsloth/Qwen2-VL-2B-Instruct", load_in_4bit=True, use_gradient_checkpointing="unsloth"
    )
    FastVisionModel.for_inference(model)
    
    def get_entailment(refs, gen_img_path, prompt):
        images = [Image.open(r).convert("RGB") for r in refs] + [Image.open(gen_img_path).convert("RGB")]
        instruction = f"Given the reference images and the text prompt '{prompt}', does the generated image align with the text and preserve the visual identity of all reference subjects? Output exactly 'Yes' or 'No'."
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
        res = processor.batch_decode(outputs, skip_special_tokens=True)[0].lower()
        return 1.0 if 'yes' in res else 0.0

    for idx, item in enumerate(manifest):
        item = init_metrics_dict(item, "refvnli")
        if 'model_A_score' in item['classic_metrics']['refvnli']: continue
        print(f"[{idx}/{len(manifest)}] REFVNLI: {item['task_id']}")
        
        refs = item['reference_images']
        prompt = item['prompt']
        score_a = get_entailment(refs, item['pair']['model_A_image'], prompt)
        score_b = get_entailment(refs, item['pair']['model_B_image'], prompt)
        
        item['classic_metrics']['refvnli'] = {"model_A_score": score_a, "model_B_score": score_b}
        if idx % 10 == 0: save_manifest(manifest, args.manifest)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--metric", type=str, choices=[
        "clip_t", "clip_i", "dinov2", "arcface", "image_reward", "scr", "refvnli"
    ])
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    if args.metric == "clip_t": manifest = run_clip_t(manifest)
    elif args.metric == "clip_i": manifest = run_clip_i(manifest)
    elif args.metric == "dinov2": manifest = run_dinov2(manifest)
    elif args.metric == "arcface": manifest = run_arcface(manifest)
    elif args.metric == "image_reward": manifest = run_image_reward(manifest)
    elif args.metric == "scr": manifest = run_scr(manifest)
    elif args.metric == "refvnli": manifest = run_refvnli(manifest)

    save_manifest(manifest, args.manifest)
    print(f"✅ Finished evaluating {args.metric} and updated manifest.")