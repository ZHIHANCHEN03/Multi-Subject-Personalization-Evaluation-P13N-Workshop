import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image


PROVIDER_TO_MODEL = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.1-flash",
}


JUDGE_INSTRUCTIONS = """
You are an expert judge for multi-subject personalized image generation.

You will receive:
1. The original generation prompt.
2. Reference images for each subject mentioned in the prompt.
3. Candidate image A.
4. Candidate image B.

Your job is to compare candidate A and B and decide which image better satisfies:
- Existence: are the required subjects/objects present?
- Appearance: do the generated subjects match the provided references?
- Interaction: are the relationships / contact / pose bindings in the prompt correct?

Rules:
- Use only the provided prompt and images.
- Judge image A and image B independently on the three dimensions.
- Then choose the better overall candidate.
- Return JSON only. Do not wrap the JSON in markdown.
- Output exactly 7 classification fields plus 1 reason field.
- Every classification field must be either 0 or 1, except `better_candidate`, which must be "A" or "B".
- `better_candidate` must never be "Tie".
- `reason` must be 25 words or fewer.

Scoring Notes & Specific Edge Cases:
- General Strictness: A score is 1 only if that dimension is fully correct. Any issue must be scored 0, subject to the exceptions below.
- Mangled/Melted Subjects: If the model attempted to generate a subject but it is severely deformed (e.g., merged subjects, missing head but body is present), score Existence as 1 (since it was attempted), but score Appearance and Interaction as 0.
- Missing Subjects in Interactions: Evaluate interactions based ONLY on the successfully generated subjects. However, if an interaction inherently requires multiple subjects (e.g., dual interaction) and one is missing, Interaction must be scored 0.
- Physics & Gravity: Be generally lenient on minor physical errors. However, obvious physical violations (e.g., floating objects, unexplained supporting structures) must be scored 0 for Interaction.
- Gaze & Movement Direction: Interactions like "looking at" or "walking towards" only require the general direction to be correct. Score 0 for Interaction only if the subject is clearly facing/moving away from the target.
- Occlusion & Spatial Relations: Strictly judge terms like "behind", "in front of", or "occluded by". As long as an object can logically be considered "behind" (spatially or by planar layers), it is acceptable. However, if the foreground/background or explicit occlusion relationship is reversed compared to the prompt, score 0 for Interaction.
- `reason` should mention the main deciding factor only.

Return this exact schema:
{
  "a_subject_existence": 0,
  "a_subject_appearance": 0,
  "a_interaction_alignment": 0,
  "b_subject_existence": 0,
  "b_subject_appearance": 0,
  "b_interaction_alignment": 0,
  "better_candidate": "A",
  "reason": "short reason"
}

Scoring notes:
- A score is 1 only if that dimension is fully correct with no visible problem.
- Any issue, even minor, must be scored 0.
- `reason` should mention the main deciding factor only.
""".strip()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    model_training_dir = script_dir.parent
    default_dataset = model_training_dir / "data_v1" / "test_v1.json"

    parser = argparse.ArgumentParser(
        description="Evaluate multi-subject generation pairs with Gemini or OpenAI VLMs."
    )
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to pairwise evaluation JSON")
    parser.add_argument(
        "--base-dir",
        default=str(model_training_dir),
        help="Base directory used to resolve relative image paths from the dataset",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSONL path. Defaults to results/<provider>_<model>_<timestamp>.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--only-task-id", default=None, help="Evaluate a single task_id")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between API calls")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    return parser.parse_args()


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")


def ensure_output_path(output_arg: Optional[str], provider: str, model: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_arg:
        output_path = Path(output_arg).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{provider}_{safe_model_name(model)}_{timestamp}.jsonl"


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset, got: {type(data)}")
    return data


def resolve_ground_truth(item: Dict[str, Any]) -> Optional[str]:
    annotator_results = item.get("annotator_results") or []
    if not annotator_results:
        return None
    votes = [str(ann.get("preference", "")).strip().upper() for ann in annotator_results]
    votes = [vote for vote in votes if vote in {"A", "B", "TIE"}]
    if not votes:
        return None
    count_a = votes.count("A")
    count_b = votes.count("B")
    if count_a == count_b:
        return "Tie"
    return "A" if count_a > count_b else "B"


def build_text_payload(item: Dict[str, Any]) -> str:
    subject_lines = []
    for ref in item.get("subject_refs", []):
        subject_lines.append(f'- subject_id="{ref.get("id", "unknown")}"')

    metadata = item.get("metadata", {})
    metadata_lines = [
        f'- task_id="{item.get("task_id", "unknown")}"',
        f'- subject_count={item.get("subject_count", "unknown")}',
        f'- ratio_type="{metadata.get("ratio_type", "unknown")}"',
        f'- model_A_name="{metadata.get("model_A_name", "A")}"',
        f'- model_B_name="{metadata.get("model_B_name", "B")}"',
    ]

    return "\n".join(
        [
            JUDGE_INSTRUCTIONS,
            "",
            "Prompt:",
            item["prompt"],
            "",
            "Subject references in order:",
            "\n".join(subject_lines) if subject_lines else "- none",
            "",
            "Metadata:",
            "\n".join(metadata_lines),
            "",
            "Image order:",
            "1. Reference images in the listed order",
            "2. Candidate image A",
            "3. Candidate image B",
        ]
    )


def preprocess_image_bytes(image_path: Path, target_size: int = 512) -> bytes:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (target_size, target_size), "white")
        offset_x = (target_size - image.width) // 2
        offset_y = (target_size - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))

        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=95)
        return output.getvalue()


def to_data_url(image_path: Path, target_size: int = 512) -> str:
    raw = preprocess_image_bytes(image_path, target_size=target_size)
    mime_type = "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates.extend(fenced)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Failed to parse JSON from model response: {text[:500]}")


def normalize_preference(value: Any) -> str:
    pref = str(value).strip().upper()
    if pref not in {"A", "B"}:
        raise ValueError(f"Invalid preference: {value}")
    return pref


def normalize_binary_score(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "pass"}:
        return 1
    if text in {"0", "false", "no", "fail"}:
        return 0
    raise ValueError(f"Invalid binary score: {value}")


def trim_reason(reason: Any, max_words: int = 25) -> str:
    words = str(reason or "").strip().split()
    return " ".join(words[:max_words])


def normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "a_subject_existence": normalize_binary_score(raw.get("a_subject_existence")),
        "a_subject_appearance": normalize_binary_score(raw.get("a_subject_appearance")),
        "a_interaction_alignment": normalize_binary_score(raw.get("a_interaction_alignment")),
        "b_subject_existence": normalize_binary_score(raw.get("b_subject_existence")),
        "b_subject_appearance": normalize_binary_score(raw.get("b_subject_appearance")),
        "b_interaction_alignment": normalize_binary_score(raw.get("b_interaction_alignment")),
        "better_candidate": normalize_preference(raw.get("better_candidate")),
        "reason": trim_reason(raw.get("reason")),
    }


def load_existing_task_ids(output_path: Path) -> Sequence[str]:
    if not output_path.exists():
        return []
    task_ids: List[str] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id:
                task_ids.append(str(task_id))
    return task_ids


def build_record(
    item: Dict[str, Any],
    normalized_result: Dict[str, Any],
    provider: str,
    model: str,
    raw_response_text: str,
) -> Dict[str, Any]:
    ground_truth = resolve_ground_truth(item)
    prediction = normalized_result["better_candidate"]

    return {
        "task_id": item.get("task_id"),
        "provider": provider,
        "model": model,
        "a_subject_existence": normalized_result["a_subject_existence"],
        "a_subject_appearance": normalized_result["a_subject_appearance"],
        "a_interaction_alignment": normalized_result["a_interaction_alignment"],
        "b_subject_existence": normalized_result["b_subject_existence"],
        "b_subject_appearance": normalized_result["b_subject_appearance"],
        "b_interaction_alignment": normalized_result["b_interaction_alignment"],
        "better_candidate": prediction,
        "reason": normalized_result["reason"],
        "ground_truth": ground_truth,
        "correct": None if ground_truth not in {"A", "B"} else prediction == ground_truth,
        "metadata": item.get("metadata", {}),
        "subject_count": item.get("subject_count"),
        "prompt": item.get("prompt"),
        "raw_response_text": raw_response_text,
    }


def write_summary(records: Sequence[Dict[str, Any]], summary_path: Path) -> None:
    scored = [record for record in records if isinstance(record.get("correct"), bool)]
    correct = sum(1 for record in scored if record["correct"])

    by_subject_count: Dict[str, Dict[str, int]] = {}
    for record in scored:
        key = str(record.get("subject_count", "unknown"))
        by_subject_count.setdefault(key, {"total": 0, "correct": 0})
        by_subject_count[key]["total"] += 1
        by_subject_count[key]["correct"] += int(bool(record["correct"]))

    summary = {
        "total_records": len(records),
        "scored_records": len(scored),
        "accuracy": (correct / len(scored)) if scored else None,
        "by_subject_count": {
            key: {
                "total": value["total"],
                "correct": value["correct"],
                "accuracy": (value["correct"] / value["total"]) if value["total"] else None,
            }
            for key, value in sorted(by_subject_count.items(), key=lambda kv: kv[0])
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def call_openai(
    api_key: str,
    model: str,
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": text_payload}]
    for idx, ref_path in enumerate(ref_paths, start=1):
        content.append({"type": "input_text", "text": f"Reference image {idx}"})
        content.append({"type": "input_image", "image_url": to_data_url(ref_path)})

    content.append({"type": "input_text", "text": "Candidate image A"})
    content.append({"type": "input_image", "image_url": to_data_url(image_a_path)})
    content.append({"type": "input_text", "text": "Candidate image B"})
    content.append({"type": "input_image", "image_url": to_data_url(image_b_path)})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        temperature=0,
    )
    return response.output_text


def call_gemini(
    api_key: str,
    model: str,
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    parts: List[Any] = [types.Part.from_text(text=text_payload)]
    for idx, ref_path in enumerate(ref_paths, start=1):
        parts.append(types.Part.from_text(text=f"Reference image {idx}"))
        parts.append(
            types.Part.from_bytes(
                data=preprocess_image_bytes(ref_path),
                mime_type="image/jpeg",
            )
        )

    parts.append(types.Part.from_text(text="Candidate image A"))
    parts.append(
        types.Part.from_bytes(
            data=preprocess_image_bytes(image_a_path),
            mime_type="image/jpeg",
        )
    )
    parts.append(types.Part.from_text(text="Candidate image B"))
    parts.append(
        types.Part.from_bytes(
            data=preprocess_image_bytes(image_b_path),
            mime_type="image/jpeg",
        )
    )

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0),
    )
    if not response.text:
        raise ValueError("Gemini response did not contain text")
    return response.text


def dispatch_api_call(
    provider: str,
    model: str,
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> str:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        return call_openai(api_key, model, text_payload, ref_paths, image_a_path, image_b_path)

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        return call_gemini(api_key, model, text_payload, ref_paths, image_a_path, image_b_path)

    raise ValueError(f"Unsupported provider: {provider}")


def resolve_image_path(base_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def select_items(items: Sequence[Dict[str, Any]], only_task_id: Optional[str], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    selected = list(items)
    if only_task_id:
        selected = [item for item in selected if str(item.get("task_id")) == only_task_id]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def main() -> None:
    args = parse_args()
    model = PROVIDER_TO_MODEL[args.provider]

    script_dir = Path(__file__).resolve().parent
    output_path = ensure_output_path(args.output, args.provider, model, script_dir / "results")
    summary_path = output_path.with_suffix(".summary.json")

    dataset_path = Path(args.dataset).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    items = load_dataset(dataset_path)
    items = select_items(items, args.only_task_id, args.max_samples)

    if not items:
        raise ValueError("No dataset items selected")

    completed_task_ids = set()
    existing_records: List[Dict[str, Any]] = []
    if output_path.exists() and not args.overwrite:
        completed_task_ids = set(load_existing_task_ids(output_path))
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))
    elif args.overwrite and output_path.exists():
        output_path.unlink()

    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset_path}")
    print(f"Base dir: {base_dir}")
    print(f"Output: {output_path}")
    print(f"Selected samples: {len(items)}")

    written_records: List[Dict[str, Any]] = list(existing_records)
    with output_path.open("a", encoding="utf-8") as f:
        for index, item in enumerate(items, start=1):
            task_id = str(item.get("task_id", f"index_{index}"))
            if task_id in completed_task_ids:
                print(f"[{index}/{len(items)}] Skip {task_id} (already exists)")
                continue

            ref_paths = [
                resolve_image_path(base_dir, ref["image_path"])
                for ref in item.get("subject_refs", [])
            ]
            image_a_path = resolve_image_path(base_dir, item["image_A_path"])
            image_b_path = resolve_image_path(base_dir, item["image_B_path"])
            text_payload = build_text_payload(item)

            print(f"[{index}/{len(items)}] Evaluating {task_id}")
            raw_response_text = dispatch_api_call(
                provider=args.provider,
                model=model,
                text_payload=text_payload,
                ref_paths=ref_paths,
                image_a_path=image_a_path,
                image_b_path=image_b_path,
            )

            parsed = extract_json_object(raw_response_text)
            normalized = normalize_result(parsed)
            record = build_record(item, normalized, args.provider, model, raw_response_text)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written_records.append(record)

            print(
                f'  better_candidate={record["better_candidate"]} '
                f'ground_truth={record["ground_truth"]} '
                f'correct={record["correct"]}'
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    write_summary(written_records, summary_path)
    print(f"Saved JSONL to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
