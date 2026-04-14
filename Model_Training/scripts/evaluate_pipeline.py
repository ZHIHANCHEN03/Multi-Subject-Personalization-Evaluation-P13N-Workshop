import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import unsloth
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lens.dataset import PrismBenchDataset
from lens.model import LENS
from lens.utils.image_processing import resize_and_pad_image


DIAGNOSTIC_DIMENSIONS = ["Existence", "Appearance", "Interaction"]


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return (correct / len(predictions)) * 100 if predictions else 0.0


def resolve_ground_truth(item: Dict) -> str:
    annotator_results = item.get("annotator_results", [])
    if not annotator_results:
        raise ValueError(f"Missing annotator_results for task_id={item.get('task_id', 'unknown')}")
    votes = [ann.get("preference", "A") for ann in annotator_results]
    return "A" if votes.count("A") >= votes.count("B") else "B"


def sigmoid_probs(logits: torch.Tensor) -> List[float]:
    return torch.sigmoid(logits).detach().cpu().tolist()


def generate_lens_explanation(pred_choice: str, logits_A: torch.Tensor, logits_B: torch.Tensor) -> str:
    rejected = "B" if pred_choice == "A" else "A"
    rejected_probs = sigmoid_probs(logits_B if rejected == "B" else logits_A)
    weakest_dim = min(range(len(rejected_probs)), key=lambda i: rejected_probs[i])
    weakest_name = DIAGNOSTIC_DIMENSIONS[weakest_dim]
    weakest_score = rejected_probs[weakest_dim]
    return f"  └─ LENS Diagnostic: Chose {pred_choice}. Reason: [{rejected}] {weakest_name} weak (score={weakest_score:.3f})"


def find_latest_checkpoint(outputs_dir: str, model_name: str, mode: str) -> str:
    safe_model_name = model_name.replace("/", "_")
    
    # First, check if a "-best" checkpoint exists
    best_path = os.path.join(outputs_dir, f"{safe_model_name}-{mode}-best")
    if os.path.isdir(best_path):
        print(f"Detected '-best' checkpoint: {best_path}")
        return best_path

    prefix = f"{safe_model_name}-{mode}-epoch"
    candidates = []
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    for name in os.listdir(outputs_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        try:
            epoch_num = int(suffix)
        except ValueError:
            continue
        candidates.append((epoch_num, os.path.join(outputs_dir, name)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for mode='{mode}' under {outputs_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_lens_checkpoint(checkpoint_dir: str, device: torch.device) -> Tuple[LENS, AutoProcessor, Dict]:
    config_path = os.path.join(checkpoint_dir, "lens_config.json")
    heads_path = os.path.join(checkpoint_dir, "lens_heads.pt")
    if not os.path.exists(config_path) or not os.path.exists(heads_path):
        raise FileNotFoundError(f"Invalid checkpoint dir: {checkpoint_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = LENS(
        model_name=cfg["base_model_name"],
        num_error_classes=cfg["num_error_classes"],
        mode=cfg["mode"],
        unfreeze_layers=cfg.get("unfreeze_layers", 4),
    )

    heads_state = torch.load(heads_path, map_location="cpu")
    model.score_head.load_state_dict(heads_state["score_head"])
    model.classification_head.load_state_dict(heads_state["classification_head"])

    lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if cfg["mode"] in {"lora", "lora_layer"}:
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"Missing LoRA adapter directory: {lora_dir}")
        model.backbone = PeftModel.from_pretrained(model.backbone, lora_dir, is_trainable=False)

    backbone_updates_path = os.path.join(checkpoint_dir, "trainable_backbone.pt")
    if cfg["mode"] in {"layer_only", "partial", "lora_layer", "full"}:
        if not os.path.exists(backbone_updates_path):
            raise FileNotFoundError(
                "Missing `trainable_backbone.pt` in checkpoint. "
                "This checkpoint was saved with the old logic and cannot support real eval. "
                "Please rerun training with the updated `save_pretrained()`."
            )
        backbone_updates = torch.load(backbone_updates_path, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(backbone_updates, strict=False)
        if unexpected:
            print(f"Warning: Unexpected backbone keys while loading checkpoint: {unexpected[:5]}")
        if missing:
            print(f"Info: Missing backbone keys during partial load: {len(missing)} entries (expected for frozen layers).")

    model.eval()
    if torch.cuda.is_available():
        model.to(device)

    processor = AutoProcessor.from_pretrained(cfg["base_model_name"], trust_remote_code=True)
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor, cfg


def build_single_batch(sample: Dict, device: torch.device) -> Tuple[Dict, Dict]:
    kwargs_A = {
        "input_ids": sample["input_ids_A"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask_A"].unsqueeze(0).to(device),
        "pixel_values": sample["pixel_values_A"].to(device),
    }
    kwargs_B = {
        "input_ids": sample["input_ids_B"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask_B"].unsqueeze(0).to(device),
        "pixel_values": sample["pixel_values_B"].to(device),
    }
    if "image_grid_thw_A" in sample:
        kwargs_A["image_grid_thw"] = sample["image_grid_thw_A"].to(device)
        kwargs_B["image_grid_thw"] = sample["image_grid_thw_B"].to(device)
    if "mm_token_type_ids_A" in sample:
        kwargs_A["mm_token_type_ids"] = sample["mm_token_type_ids_A"].unsqueeze(0).to(device)
        kwargs_B["mm_token_type_ids"] = sample["mm_token_type_ids_B"].unsqueeze(0).to(device)
    return kwargs_A, kwargs_B


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def get_clip_text_embedding(clip_model: CLIPModel, clip_text: Dict[str, torch.Tensor]) -> torch.Tensor:
    text_outputs = clip_model.text_model(**clip_text)
    pooled = text_outputs.pooler_output
    return clip_model.text_projection(pooled)


def get_clip_image_embedding(clip_model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    return clip_model.visual_projection(pooled)


def load_pil_image(base_dir: str, rel_path: str, image_size: int = 512):
    abs_path = os.path.join(base_dir, rel_path)
    return resize_and_pad_image(abs_path, target_size=(image_size, image_size))


def main(args):
    print("--- Starting REAL Pipeline Evaluation (LENS vs Baselines) ---")
    print(f"Running script: {os.path.abspath(__file__)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_path = os.path.join(os.path.dirname(__file__), "../data_v1/test_v1.json")
    outputs_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found at {test_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    checkpoint_dir = args.checkpoint_dir or find_latest_checkpoint(outputs_dir, args.model_name, args.mode)
    print(f"Loading LENS checkpoint from: {checkpoint_dir}")
    lens_model, lens_processor, lens_cfg = load_lens_checkpoint(checkpoint_dir, device)
    lens_dataset = PrismBenchDataset(json_path=test_path, processor=lens_processor, image_size=args.image_size)

    print(f"Loading CLIP baseline: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_max_length = int(clip_model.config.text_config.max_position_embeddings)
    print(f"CLIP text truncation enabled with max_length={clip_max_length}")

    print(f"Loading DINO baseline: {args.dino_model}")
    dino_model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)

    ground_truths = []
    lens_preds = []
    clip_preds = []
    dino_preds = []

    print("\n--- Example Interpretability Logs ---")
    for idx, item in enumerate(test_data):
        gt_pref = resolve_ground_truth(item)
        ground_truths.append(gt_pref)

        sample = lens_dataset[idx]
        kwargs_A, kwargs_B = build_single_batch(sample, device)
        with torch.no_grad():
            score_A, logits_A = lens_model(**kwargs_A)
            score_B, logits_B = lens_model(**kwargs_B)

        score_A_val = float(score_A.squeeze().item())
        score_B_val = float(score_B.squeeze().item())
        lens_pred = "A" if score_A_val > score_B_val else "B"
        lens_preds.append(lens_pred)

        if idx % 5 == 0:
            print(f"Task ID: {item.get('task_id', 'unknown')}")
            print(generate_lens_explanation(lens_pred, logits_A.squeeze(0), logits_B.squeeze(0)))

        prompt = item["prompt"]
        img_A = load_pil_image(base_dir, item["image_A_path"], args.image_size)
        img_B = load_pil_image(base_dir, item["image_B_path"], args.image_size)

        with torch.no_grad():
            # Use the tokenizer directly so max_length is applied explicitly.
            clip_text = clip_processor.tokenizer(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=clip_max_length,
            ).to(device)
            clip_imgs = clip_processor(images=[img_A, img_B], return_tensors="pt").to(device)
            text_feat = get_clip_text_embedding(clip_model, clip_text)
            image_feat = get_clip_image_embedding(clip_model, clip_imgs["pixel_values"])
            clip_scores = cosine_similarity(image_feat, text_feat.expand_as(image_feat))
        clip_preds.append("A" if float(clip_scores[0]) > float(clip_scores[1]) else "B")

        ref_images = [load_pil_image(base_dir, ref["image_path"], args.image_size) for ref in item["subject_refs"]]
        with torch.no_grad():
            ref_inputs = dino_processor(images=ref_images, return_tensors="pt").to(device)
            gen_inputs = dino_processor(images=[img_A, img_B], return_tensors="pt").to(device)
            ref_outputs = dino_model(**ref_inputs).last_hidden_state[:, 0]
            gen_outputs = dino_model(**gen_inputs).last_hidden_state[:, 0]
            ref_mean = F.normalize(ref_outputs, dim=-1).mean(dim=0, keepdim=True)
            ref_mean = F.normalize(ref_mean, dim=-1)
            dino_scores = cosine_similarity(gen_outputs, ref_mean.expand_as(gen_outputs))
        dino_preds.append("A" if float(dino_scores[0]) > float(dino_scores[1]) else "B")

    lens_acc = calculate_accuracy(lens_preds, ground_truths)
    clip_acc = calculate_accuracy(clip_preds, ground_truths)
    dino_acc = calculate_accuracy(dino_preds, ground_truths)

    print("\n==========================================")
    print("        PIPELINE EVALUATION RESULTS       ")
    print("==========================================")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Mode: {lens_cfg['mode']}")
    print(f"Test Set Size: {len(test_data)} pairs")
    print(f"1. CLIP Accuracy:   {clip_acc:.2f}% (Real Baseline - Prompt/Image Similarity)")
    print(f"2. DINO Accuracy:   {dino_acc:.2f}% (Real Baseline - Ref/Image Similarity)")
    print(f"3. LENS Accuracy:   {lens_acc:.2f}% (Real Checkpoint - Diagnostic Metric)")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real evaluation for LENS, CLIP, and DINO baselines.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-0.8B", help="Backbone model name used during training")
    parser.add_argument("--mode", type=str, default="layer_only", help="Checkpoint mode prefix used under outputs/.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Optional explicit checkpoint directory.")
    parser.add_argument("--image_size", type=int, default=512, help="Resize-and-pad size used during evaluation.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-base")
    args = parser.parse_args()
    main(args)
