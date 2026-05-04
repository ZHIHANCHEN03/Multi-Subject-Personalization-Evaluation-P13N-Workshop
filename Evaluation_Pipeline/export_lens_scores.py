import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoProcessor
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_TRAINING_ROOT = REPO_ROOT / "Model_Training"
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.append(str(MODEL_TRAINING_ROOT))

from lens.model import LENS  # noqa: E402
from lens.utils.image_processing import load_image_with_safety  # noqa: E402


READY_METRIC_MODELS = {
    "qwen35_08b_layer_only": {
        "model_name": "unsloth/Qwen3.5-0.8B",
        "mode": "layer_only",
        "checkpoint_relpath": (
            "v2/unsloth_Qwen3.5-0.8B/20260501_200302/outputs/"
            "unsloth_Qwen3.5-0.8B-layer_only-best"
        ),
    },
    "qwen35_08b_lora_layer": {
        "model_name": "unsloth/Qwen3.5-0.8B",
        "mode": "lora_layer",
        "checkpoint_relpath": (
            "v2/unsloth_Qwen3.5-0.8B/20260501_202531/outputs/"
            "unsloth_Qwen3.5-0.8B-lora_layer-best"
        ),
    },
    "qwen35_2b_layer_only": {
        "model_name": "unsloth/Qwen3.5-2B",
        "mode": "layer_only",
        "checkpoint_relpath": (
            "v2/unsloth_Qwen3.5-2B/20260501_203049/outputs/"
            "unsloth_Qwen3.5-2B-layer_only-best"
        ),
    },
    "qwen35_2b_lora_layer": {
        "model_name": "unsloth/Qwen3.5-2B",
        "mode": "lora_layer",
        "checkpoint_relpath": (
            "v2/unsloth_Qwen3.5-2B/20260503_033216/outputs/"
            "unsloth_Qwen3.5-2B-lora_layer-best"
        ),
    },
    "qwen35_4b_lora_layer": {
        "model_name": "unsloth/Qwen3.5-4B",
        "mode": "lora_layer",
        "checkpoint_relpath": (
            "v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/"
            "unsloth_Qwen3.5-4B-lora_layer-best"
        ),
    },
}

DIAGNOSTIC_DIMENSIONS = ("existence", "appearance", "interaction")


def log(message: str) -> None:
    print(f"[export_lens_scores] {message}", flush=True)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_manifest(manifest_path: Path) -> List[Dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_images(reference_images: Sequence[str], generated_image: str) -> List:
    images = [load_image_with_safety(str(path)) for path in reference_images]
    images.append(load_image_with_safety(str(generated_image)))
    return images


def build_prompt_text(processor, prompt: str, num_refs: int) -> str:
    content = [{"type": "image"} for _ in range(num_refs)]
    content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                "You are evaluating a multi-subject personalization result. "
                "The first images are subject references. "
                "The last image is the generated candidate. "
                f"Prompt: {prompt}"
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_inputs(processor, prompt: str, reference_images: Sequence[str], generated_image: str, device: torch.device) -> Dict:
    images = load_images(reference_images, generated_image)
    text_prompt = build_prompt_text(processor, prompt, num_refs=len(reference_images))
    inputs = processor(
        text=[text_prompt],
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
    }
    if "mm_token_type_ids" in inputs:
        model_inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(device)
    if "image_grid_thw" in inputs:
        model_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(device)
    return model_inputs


def load_lens_checkpoint(checkpoint_dir: Path, device: torch.device) -> Tuple[LENS, AutoProcessor, Dict]:
    config_path = checkpoint_dir / "lens_config.json"
    heads_path = checkpoint_dir / "lens_heads.pt"
    if not config_path.exists() or not heads_path.exists():
        raise FileNotFoundError(f"Invalid checkpoint directory: {checkpoint_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    init_mode = "head_only" if cfg["mode"] in {"lora", "lora_layer"} else cfg["mode"]
    model = LENS(
        model_name=cfg["base_model_name"],
        num_error_classes=cfg["num_error_classes"],
        mode=init_mode,
        unfreeze_layers=cfg.get("unfreeze_layers", 4),
    )

    heads_state = torch.load(heads_path, map_location="cpu")
    model.score_head.load_state_dict(heads_state["score_head"])
    model.classification_head.load_state_dict(heads_state["classification_head"])

    lora_dir = checkpoint_dir / "lora_adapter"
    if cfg["mode"] in {"lora", "lora_layer"}:
        if not lora_dir.is_dir():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {lora_dir}")
        model.backbone = PeftModel.from_pretrained(model.base_model, str(lora_dir), is_trainable=False)

    backbone_updates_path = checkpoint_dir / "trainable_backbone.pt"
    if cfg["mode"] in {"layer_only", "partial", "lora_layer", "full"}:
        if not backbone_updates_path.exists():
            raise FileNotFoundError(f"Missing trainable backbone weights: {backbone_updates_path}")
        backbone_updates = torch.load(backbone_updates_path, map_location="cpu")
        model.backbone.load_state_dict(backbone_updates, strict=False)

    model.eval()
    model.to(device)

    processor = AutoProcessor.from_pretrained(cfg["base_model_name"], trust_remote_code=True)
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor, cfg


def score_single_image(
    model: LENS,
    processor,
    prompt: str,
    reference_images: Sequence[str],
    generated_image: str,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model_inputs = prepare_inputs(processor, prompt, reference_images, generated_image, device)
    with torch.no_grad():
        score, logits = model(**model_inputs)
        raw_score = float(score.squeeze().item())
        probs = torch.sigmoid(logits.squeeze(0)).detach().cpu().tolist()
    category_scores = {
        dim: float(prob)
        for dim, prob in zip(DIAGNOSTIC_DIMENSIONS, probs)
    }
    return raw_score, category_scores


def infer_dataset_name(task_id: str) -> str:
    if task_id.startswith("v10_"):
        return "v10"
    if task_id.startswith("v13_"):
        return "v13"
    return "unknown"


def expand_pair_records(
    pair_item: Dict,
    metrics_alias: str,
    checkpoint_dir: Path,
    score_a: float,
    cats_a: Dict[str, float],
    score_b: float,
    cats_b: Dict[str, float],
) -> List[Dict]:
    pair = pair_item["pair"]
    win_prob_a = sigmoid(score_a - score_b)
    win_prob_b = 1.0 - win_prob_a
    preferred_side = "A" if score_a >= score_b else "B"
    dataset_name = pair_item.get("dataset", infer_dataset_name(pair_item.get("task_id", "")))

    base_fields = {
        "pair_id": pair_item["task_id"],
        "dataset": dataset_name,
        "prompt": pair_item.get("prompt", ""),
        "metrics_model_name": metrics_alias,
        "metrics_checkpoint_dir": str(checkpoint_dir),
        "predicted_preference": preferred_side,
    }

    return [
        {
            "id": f"{pair_item['task_id']}::A::{metrics_alias}",
            **base_fields,
            "gen_image_model_name": pair["model_A_name"],
            "gen_image_path": pair["model_A_image"],
            "preference_raw_score": score_a,
            "pairwise_win_prob": win_prob_a,
            "existence_score": cats_a["existence"],
            "appearance_score": cats_a["appearance"],
            "interaction_score": cats_a["interaction"],
        },
        {
            "id": f"{pair_item['task_id']}::B::{metrics_alias}",
            **base_fields,
            "gen_image_model_name": pair["model_B_name"],
            "gen_image_path": pair["model_B_image"],
            "preference_raw_score": score_b,
            "pairwise_win_prob": win_prob_b,
            "existence_score": cats_b["existence"],
            "appearance_score": cats_b["appearance"],
            "interaction_score": cats_b["interaction"],
        },
    ]


def validate_manifest_item(item: Dict) -> None:
    if "pair" not in item:
        raise ValueError(f"Manifest item missing `pair`: {item.get('task_id', 'unknown')}")
    if "reference_images" not in item or not item["reference_images"]:
        raise ValueError(
            f"Manifest item missing `reference_images`: {item.get('task_id', 'unknown')} "
            "Please build a manifest with correct refs first."
        )


def run_export_for_metric_model(
    manifest_items: Sequence[Dict],
    metrics_alias: str,
    checkpoint_dir: Path,
    device: torch.device,
    dataset_base_dir: Path,
    log_every: int,
) -> List[Dict]:
    log(f"Loading metric model: {metrics_alias}")
    log(f"Checkpoint directory: {checkpoint_dir}")
    model, processor, _ = load_lens_checkpoint(checkpoint_dir, device)
    results: List[Dict] = []

    for index, item in enumerate(
        tqdm(manifest_items, desc=f"Scoring {metrics_alias}", unit=" pair", mininterval=1.0),
        start=1,
    ):
        validate_manifest_item(item)
        pair = item["pair"]
        refs = [str(normalize_path(path, dataset_base_dir)) for path in item["reference_images"]]
        gen_a = str(normalize_path(pair["model_A_image"], dataset_base_dir))
        gen_b = str(normalize_path(pair["model_B_image"], dataset_base_dir))

        score_a, cats_a = score_single_image(
            model=model,
            processor=processor,
            prompt=item.get("prompt", ""),
            reference_images=refs,
            generated_image=gen_a,
            device=device,
        )
        score_b, cats_b = score_single_image(
            model=model,
            processor=processor,
            prompt=item.get("prompt", ""),
            reference_images=refs,
            generated_image=gen_b,
            device=device,
        )
        results.extend(
            expand_pair_records(
                pair_item=item,
                metrics_alias=metrics_alias,
                checkpoint_dir=checkpoint_dir,
                score_a=score_a,
                cats_a=cats_a,
                score_b=score_b,
                cats_b=cats_b,
            )
        )
        if log_every > 0 and (index % log_every == 0 or index == len(manifest_items)):
            log(
                f"{metrics_alias} progress {index}/{len(manifest_items)} | "
                f"task_id={item.get('task_id', 'unknown')} | "
                f"A={pair['model_A_name']} score={score_a:.4f} | "
                f"B={pair['model_B_name']} score={score_b:.4f}"
            )

    return results


def resolve_metric_model_specs(runs_root: Path, selected_aliases: Sequence[str]) -> List[Tuple[str, Path]]:
    specs = []
    for alias in selected_aliases:
        if alias not in READY_METRIC_MODELS:
            raise ValueError(
                f"Unknown metric model alias: {alias}. "
                f"Available: {', '.join(sorted(READY_METRIC_MODELS))}"
            )
        relpath = READY_METRIC_MODELS[alias]["checkpoint_relpath"]
        checkpoint_dir = (runs_root / relpath).resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")
        specs.append((alias, checkpoint_dir))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LENS preference/category scores for all generated images in a pair manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to a pair-level manifest JSONL, e.g. v10_manifest.jsonl or v13_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path for image-level LENS scores",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="/workspace/Model_Training_runs",
        help="Root directory containing Model_Training_runs",
    )
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        default=str(REPO_ROOT),
        help="Base directory used to resolve relative image paths from the manifest",
    )
    parser.add_argument(
        "--metric_models",
        nargs="+",
        default=list(READY_METRIC_MODELS.keys()),
        help="Metric model aliases to run. Default: all 5 ready models.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Emit explicit progress logs every N pairs for each metric model. Set <=0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    runs_root = Path(args.runs_root).resolve()
    dataset_base_dir = Path(args.dataset_base_dir).resolve()

    manifest_items = load_manifest(manifest_path)
    log(f"Loaded {len(manifest_items)} pair items from {manifest_path}")
    metric_specs = resolve_metric_model_specs(runs_root, args.metric_models)
    log(f"Will run {len(metric_specs)} metric models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    all_records: List[Dict] = []
    for metrics_alias, checkpoint_dir in metric_specs:
        metric_records = run_export_for_metric_model(
            manifest_items=manifest_items,
            metrics_alias=metrics_alias,
            checkpoint_dir=checkpoint_dir,
            device=device,
            dataset_base_dir=dataset_base_dir,
            log_every=args.log_every,
        )
        all_records.extend(metric_records)
        log(f"Finished {metrics_alias}: {len(metric_records)} image-level records")

    write_jsonl(all_records, output_path)
    log(f"Wrote {len(all_records)} records to {output_path}")


if __name__ == "__main__":
    main()
