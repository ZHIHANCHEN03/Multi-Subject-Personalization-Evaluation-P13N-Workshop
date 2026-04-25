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
    "gemini": "gemini-3.1-flash-lite-preview",
}
OPENAI_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
GEMINI_BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
GEMINI_BATCH_MAX_INLINE_BYTES = 18_000_000


JUDGE_INSTRUCTIONS = """
You are an expert judge for multi-subject personalized image generation.

You will receive:
1. The original generation prompt.
2. Reference images for each subject mentioned in the prompt.
3. Candidate image A.
4. Candidate image B.

Your job consists of two parts:
Part 1: Independent Evaluation
Evaluate candidate A and candidate B independently against the prompt and references. For each image, assess the following three dimensions:
- Existence: Are ALL the required subjects in the references present in the generated image?
- Appearance: Do the generated subjects match the provided references in identity and style?
- Interaction: Does the generated image accurately reflect the prompt's described actions, maintain correct relative proportions between subjects, and follow realistic spatial/physical laws?
Part 2: Overall Comparison
Compare Candidate A and B and decide the winner.

OUTPUT RULES:
- Use ONLY the provided prompt and images to make your decision.
- Judge image A and image B independently on the three dimensions.
- Choose the overall winner. `winner` must be either "A" or "B" (Never "Tie").
- All dimensional scores (`*_existence`, `*_appearance`, `*_interaction`) must be strictly binary: 1 (fully correct) or 0 (any issue exists).
- Return valid JSON ONLY. Absolutely NO markdown formatting, NO markdown code blocks (do not use ```json), and NO extra text.
- `reason` must be 25 words or fewer, mentioning only the main deciding factor or the dimensions that scored 0.

SCORING NOTES & SPECIFIC EDGE CASES:
- Missing Subjects (Strict Existence Rule): If one or more subjects from the reference images are completely missing (i.e., not rendered at all) in the candidate image, you MUST score Existence as 0. 
- Mangled/Melted Subjects: If the model clearly attempted to generate a subject but it is severely deformed (e.g., missing a head but the body is present, or subjects are merged), score Existence as 1 (since it was attempted), but strictly score Appearance and Interaction as 0.
- Appearance Matching: Score Appearance as 1 ONLY if every successfully generated subject matches its reference in hairstyle, facial features, clothing style, and object style. Any mismatch in these identity-defining details must result in an Appearance score of 0.
- Interaction - Partial Generation: Evaluate interactions based ONLY on the subjects that were actually rendered. If the generated subjects perform their individual actions correctly, score Interaction as 1, even if other unrelated subjects are missing. HOWEVER, if an interaction inherently requires a missing subject (e.g., rendered subject A is supposed to hug subject B, but subject B is missing), the interaction becomes physically/logically impossible and MUST be scored 0.
- Interaction - Proportions & Spatial Logic: Subjects must maintain logical relative sizes (proportions) and realistic spatial positioning as dictated by the prompt and real-world physical laws. Scale distortions or impossible geometry must be scored 0.
- Interaction - Physics & Gravity: Obvious physical violations (e.g., floating objects, unexplained supporting structures) must be scored 0. Be lenient only on very minor physical inaccuracies.
- Interaction - Gaze & Movement: Terms like "looking at" or "walking towards" require the correct general direction. Score 0 only if the subject is clearly facing or moving away from the target.
- Interaction - Occlusion: Strictly judge terms like "behind" or "in front of". If an object can logically be considered "behind" spatially, it is acceptable. If the foreground/background relationship is explicitly reversed compared to the prompt, score 0.

EXPECTED JSON SCHEMA:
{
  "a_existence": 0,
  "a_appearance": 0,
  "a_interaction": 0,
  "b_existence": 0,
  "b_appearance": 0,
  "b_interaction": 0,
  "winner": "A",
  "reason": "short reason here"
}
""".strip()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    default_data_root = project_root / "data"
    default_dataset = default_data_root / "train_60k_v13_2.jsonl"

    parser = argparse.ArgumentParser(
        description="Evaluate multi-subject generation pairs with Gemini or OpenAI VLMs."
    )
    parser.add_argument("--provider", choices=["openai", "gemini"], required=True)
    parser.add_argument(
        "--api-mode",
        choices=["auto", "sync", "batch"],
        default="auto",
        help="Execution mode. `auto` uses batch for both OpenAI and Gemini.",
    )
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to pairwise evaluation JSON")
    parser.add_argument(
        "--base-dir",
        default=str(default_data_root),
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
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Existing batch ID(s) to poll / collect. For Gemini, pass comma-separated job names if needed.",
    )
    parser.add_argument(
        "--wait-for-batch",
        action="store_true",
        help="Wait for submitted batch job(s) to reach a terminal state and then collect results.",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval in seconds when waiting for batch completion.",
    )
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


def load_local_env_files(script_dir: Path) -> None:
    # Prefer real local config, but allow `.env.example` as a fallback for convenience.
    for env_path in [script_dir / ".env", script_dir / ".env.example"]:
        if not env_path.exists():
            continue
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def find_existing_file(base_dir: Path, stem: str, exts: Sequence[str]) -> Path:
    for ext in exts:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to find file for stem={stem!r} under {base_dir}")


def build_item_from_jsonl_record(record: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    task_id = str(record["id"])
    people_names = list(record.get("people_names") or [])
    object_names = list(record.get("object_names") or [])
    subject_names = people_names + object_names

    subject_refs = [
        {
            "id": subject_name,
            "image_path": str(find_existing_file(data_root / "refs", subject_name, [".jpg", ".jpeg", ".png", ".webp"])),
        }
        for subject_name in subject_names
    ]

    image_a_path = find_existing_file(data_root / "A", task_id, [".png", ".jpg", ".jpeg", ".webp"])
    image_b_path = find_existing_file(data_root / "B", task_id, [".png", ".jpg", ".jpeg", ".webp"])

    prompt = record.get("prompt_en") or record.get("prompt_zh")
    if not prompt:
        raise ValueError(f"Sample {task_id} does not contain `prompt_en` or `prompt_zh`")

    return {
        "task_id": task_id,
        "subject_count": record.get("total_entities", len(subject_refs)),
        "prompt": prompt,
        "subject_refs": subject_refs,
        "image_A_path": str(image_a_path),
        "image_B_path": str(image_b_path),
        "metadata": {
            "ratio_type": record.get("ratio_type", "unknown"),
            "model_A_name": "A",
            "model_B_name": "B",
            "level": record.get("level"),
            "class_tag": record.get("class_tag"),
            "seed_id": record.get("seed_id"),
            "n_humans": record.get("n_humans"),
            "n_objects": record.get("n_objects"),
            "people_names": people_names,
            "object_names": object_names,
        },
    }


def load_dataset(dataset_path: Path, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    if dataset_path.suffix.lower() == ".jsonl":
        data_root = base_dir or dataset_path.parent
        items: List[Dict[str, Any]] = []
        skipped_missing_assets = 0
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object at line {line_number}, got: {type(record)}")
                try:
                    items.append(build_item_from_jsonl_record(record, data_root))
                except FileNotFoundError:
                    skipped_missing_assets += 1
        if skipped_missing_assets:
            print(
                f"Skipped {skipped_missing_assets} JSONL samples with missing references or candidate images "
                f"under {data_root}"
            )
        return items

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset, got: {type(data)}")
    return data


def resolve_api_mode(_provider: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "batch"
    return requested_mode


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
        "a_existence": normalize_binary_score(raw.get("a_existence")),
        "a_appearance": normalize_binary_score(raw.get("a_appearance")),
        "a_interaction": normalize_binary_score(raw.get("a_interaction")),
        "b_existence": normalize_binary_score(raw.get("b_existence")),
        "b_appearance": normalize_binary_score(raw.get("b_appearance")),
        "b_interaction": normalize_binary_score(raw.get("b_interaction")),
        "winner": normalize_preference(raw.get("winner")),
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


def load_existing_records(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_record(
    item: Dict[str, Any],
    normalized_result: Dict[str, Any],
    provider: str,
    model: str,
    raw_response_text: str,
) -> Dict[str, Any]:
    return {
        "task_id": item.get("task_id"),
        "provider": provider,
        "model": model,
        "a_existence": normalized_result["a_existence"],
        "a_appearance": normalized_result["a_appearance"],
        "a_interaction": normalized_result["a_interaction"],
        "b_existence": normalized_result["b_existence"],
        "b_appearance": normalized_result["b_appearance"],
        "b_interaction": normalized_result["b_interaction"],
        "winner": normalized_result["winner"],
        "reason": normalized_result["reason"],
        "metadata": item.get("metadata", {}),
        "subject_count": item.get("subject_count"),
        "prompt": item.get("prompt"),
        "raw_response_text": raw_response_text,
    }


def build_openai_content(
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": text_payload}]
    for idx, ref_path in enumerate(ref_paths, start=1):
        content.append({"type": "input_text", "text": f"Reference image {idx}"})
        content.append({"type": "input_image", "image_url": to_data_url(ref_path)})

    content.append({"type": "input_text", "text": "Candidate image A"})
    content.append({"type": "input_image", "image_url": to_data_url(image_a_path)})
    content.append({"type": "input_text", "text": "Candidate image B"})
    content.append({"type": "input_image", "image_url": to_data_url(image_b_path)})
    return content


def build_openai_request_body(
    model: str,
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> Dict[str, Any]:
    return {
        "model": model,
        "input": [{"role": "user", "content": build_openai_content(text_payload, ref_paths, image_a_path, image_b_path)}],
        "temperature": 0,
    }


def write_summary(records: Sequence[Dict[str, Any]], summary_path: Path) -> None:
    by_subject_count: Dict[str, int] = {}
    for record in records:
        key = str(record.get("subject_count", "unknown"))
        by_subject_count[key] = by_subject_count.get(key, 0) + 1

    summary = {
        "total_records": len(records),
        "by_subject_count": dict(sorted(by_subject_count.items(), key=lambda kv: kv[0])),
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
    request_body = build_openai_request_body(model, text_payload, ref_paths, image_a_path, image_b_path)
    response = client.responses.create(**request_body)
    return response.output_text


def get_gemini_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    return api_key


def build_gemini_contents(
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> List[Any]:
    from google.genai import types

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
    return parts


def build_gemini_generate_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(temperature=0)


def extract_gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    chunks: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                chunks.append(part_text)
    if chunks:
        return "".join(chunks)
    raise ValueError("Gemini response did not contain text")


def call_gemini(
    api_key: str,
    model: str,
    text_payload: str,
    ref_paths: Sequence[Path],
    image_a_path: Path,
    image_b_path: Path,
) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=build_gemini_contents(text_payload, ref_paths, image_a_path, image_b_path),
        config=build_gemini_generate_config(),
    )
    return extract_gemini_response_text(response)


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
        api_key = get_gemini_api_key()
        return call_gemini(api_key, model, text_payload, ref_paths, image_a_path, image_b_path)

    raise ValueError(f"Unsupported provider: {provider}")


def serialize_batch_jsonl_line(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def extract_openai_output_text(response_body: Dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    output = response_body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)

    choices = response_body.get("choices")
    if isinstance(choices, list):
        fallback_chunks: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                fallback_chunks.append(content)
        if fallback_chunks:
            return "\n".join(fallback_chunks)

    raise ValueError("Unable to extract response text from OpenAI response body")


def response_content_to_text(content_obj: Any) -> str:
    text_attr = getattr(content_obj, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        value = text_attr()
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")

    content_attr = getattr(content_obj, "content", None)
    if isinstance(content_attr, (bytes, bytearray)):
        return bytes(content_attr).decode("utf-8")
    if isinstance(content_attr, str):
        return content_attr

    read_attr = getattr(content_obj, "read", None)
    if callable(read_attr):
        value = read_attr()
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8")

    raise ValueError("Unable to read text from OpenAI file content response")


def prepare_items(
    items: Sequence[Dict[str, Any]],
    base_dir: Path,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for item in items:
        prepared.append(
            {
                "item": item,
                "task_id": str(item.get("task_id")),
                "text_payload": build_text_payload(item),
                "ref_paths": [
                    resolve_image_path(base_dir, ref["image_path"])
                    for ref in item.get("subject_refs", [])
                ],
                "image_a_path": resolve_image_path(base_dir, item["image_A_path"]),
                "image_b_path": resolve_image_path(base_dir, item["image_B_path"]),
            }
        )
    return prepared


def write_openai_batch_input_file(
    prepared_items: Sequence[Dict[str, Any]],
    model: str,
    batch_input_path: Path,
) -> None:
    with batch_input_path.open("w", encoding="utf-8") as f:
        for prepared in prepared_items:
            request_body = build_openai_request_body(
                model=model,
                text_payload=prepared["text_payload"],
                ref_paths=prepared["ref_paths"],
                image_a_path=prepared["image_a_path"],
                image_b_path=prepared["image_b_path"],
            )
            line = {
                "custom_id": prepared["task_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body,
            }
            f.write(serialize_batch_jsonl_line(line) + "\n")


def write_batch_metadata(metadata_path: Path, payload: Dict[str, Any]) -> None:
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def poll_openai_batch_until_terminal(client: Any, batch_id: str, poll_seconds: float) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        counts = getattr(batch, "request_counts", None)
        print(f"Batch {batch_id} status={status} counts={counts}")
        if status in OPENAI_BATCH_TERMINAL_STATUSES:
            return batch
        time.sleep(max(poll_seconds, 1.0))


def collect_openai_batch_records(
    client: Any,
    batch: Any,
    prepared_by_task_id: Dict[str, Dict[str, Any]],
    provider: str,
    model: str,
    completed_task_ids: Sequence[str],
    output_raw_path: Path,
    error_raw_path: Path,
) -> List[Dict[str, Any]]:
    completed_task_id_set = set(completed_task_ids)
    records: List[Dict[str, Any]] = []

    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        output_text = response_content_to_text(client.files.content(output_file_id))
        output_raw_path.write_text(output_text, encoding="utf-8")
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            task_id = str(payload.get("custom_id"))
            if task_id in completed_task_id_set:
                continue
            prepared = prepared_by_task_id.get(task_id)
            if prepared is None:
                continue
            if payload.get("error"):
                continue
            response = payload.get("response") or {}
            body = response.get("body") or {}
            raw_response_text = extract_openai_output_text(body)
            parsed = extract_json_object(raw_response_text)
            normalized = normalize_result(parsed)
            record = build_record(prepared["item"], normalized, provider, model, raw_response_text)
            records.append(record)

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_text = response_content_to_text(client.files.content(error_file_id))
        error_raw_path.write_text(error_text, encoding="utf-8")

    return records


def run_sync_mode(
    args: argparse.Namespace,
    items: Sequence[Dict[str, Any]],
    base_dir: Path,
    output_path: Path,
    summary_path: Path,
    provider: str,
    model: str,
) -> None:
    completed_task_ids = set()
    existing_records = load_existing_records(output_path)
    if output_path.exists() and not args.overwrite:
        completed_task_ids = set(load_existing_task_ids(output_path))
    elif args.overwrite and output_path.exists():
        output_path.unlink()
        existing_records = []

    prepared_items = prepare_items(items, base_dir)

    written_records: List[Dict[str, Any]] = list(existing_records)
    with output_path.open("a", encoding="utf-8") as f:
        for index, prepared in enumerate(prepared_items, start=1):
            task_id = prepared["task_id"]
            if task_id in completed_task_ids:
                print(f"[{index}/{len(prepared_items)}] Skip {task_id} (already exists)")
                continue

            print(f"[{index}/{len(prepared_items)}] Evaluating {task_id}")
            raw_response_text = dispatch_api_call(
                provider=provider,
                model=model,
                text_payload=prepared["text_payload"],
                ref_paths=prepared["ref_paths"],
                image_a_path=prepared["image_a_path"],
                image_b_path=prepared["image_b_path"],
            )

            parsed = extract_json_object(raw_response_text)
            normalized = normalize_result(parsed)
            record = build_record(prepared["item"], normalized, provider, model, raw_response_text)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written_records.append(record)

            print(f'  winner={record["winner"]}')

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    write_summary(written_records, summary_path)
    print(f"Saved JSONL to: {output_path}")
    print(f"Saved summary to: {summary_path}")


def run_openai_batch_mode(
    args: argparse.Namespace,
    items: Sequence[Dict[str, Any]],
    base_dir: Path,
    output_path: Path,
    summary_path: Path,
    provider: str,
    model: str,
) -> None:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    prepared_items = prepare_items(items, base_dir)
    prepared_by_task_id = {prepared["task_id"]: prepared for prepared in prepared_items}

    existing_records = load_existing_records(output_path)
    completed_task_ids = set(record.get("task_id") for record in existing_records if record.get("task_id"))
    if args.overwrite and output_path.exists():
        output_path.unlink()
        existing_records = []
        completed_task_ids = set()

    pending_prepared_items = [
        prepared for prepared in prepared_items if prepared["task_id"] not in completed_task_ids
    ]

    batch_metadata_path = output_path.with_suffix(".batch_meta.json")
    batch_input_path = output_path.with_suffix(".batch_input.jsonl")
    batch_output_raw_path = output_path.with_suffix(".batch_output_raw.jsonl")
    batch_error_raw_path = output_path.with_suffix(".batch_error_raw.jsonl")

    if args.batch_id:
        batch = client.batches.retrieve(args.batch_id)
        print(f"Loaded existing batch: id={batch.id} status={batch.status}")
    else:
        if not pending_prepared_items:
            print("No pending items to submit.")
            if existing_records:
                write_summary(existing_records, summary_path)
                print(f"Saved summary to: {summary_path}")
            return

        write_openai_batch_input_file(pending_prepared_items, model, batch_input_path)
        uploaded_file = client.files.create(file=batch_input_path.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        metadata = {
            "provider": provider,
            "model": model,
            "api_mode": "batch",
            "batch_id": batch.id,
            "input_file_id": uploaded_file.id,
            "batch_input_path": str(batch_input_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "submitted_task_ids": [prepared["task_id"] for prepared in pending_prepared_items],
        }
        write_batch_metadata(batch_metadata_path, metadata)
        print(f"Created OpenAI batch: id={batch.id} status={batch.status}")
        print(f"Saved batch metadata to: {batch_metadata_path}")

    if args.wait_for_batch:
        batch = poll_openai_batch_until_terminal(client, batch.id, args.batch_poll_seconds)
    else:
        if batch.status not in OPENAI_BATCH_TERMINAL_STATUSES:
            print("Batch submitted but not yet complete. Re-run with --batch-id <id> --wait-for-batch to collect results.")
            return

    batch = client.batches.retrieve(batch.id)
    print(f"Final batch status: {batch.status}")

    new_records = collect_openai_batch_records(
        client=client,
        batch=batch,
        prepared_by_task_id=prepared_by_task_id,
        provider=provider,
        model=model,
        completed_task_ids=completed_task_ids,
        output_raw_path=batch_output_raw_path,
        error_raw_path=batch_error_raw_path,
    )

    if not new_records:
        print("No new completed records were collected from the batch.")
        return

    with output_path.open("a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_records = existing_records + new_records
    write_summary(all_records, summary_path)
    print(f"Collected {len(new_records)} new records from batch {batch.id}")
    print(f"Saved JSONL to: {output_path}")
    print(f"Saved summary to: {summary_path}")


def model_to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_to_jsonable(model_dump(mode="json", exclude_none=True))

    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return model_to_jsonable(to_json_dict())

    return str(value)


def parse_batch_ids(batch_id_arg: str) -> List[str]:
    return [part.strip() for part in str(batch_id_arg).split(",") if part.strip()]


def estimate_gemini_request_size_bytes(prepared: Dict[str, Any]) -> int:
    image_paths = list(prepared["ref_paths"]) + [prepared["image_a_path"], prepared["image_b_path"]]
    image_bytes = sum(len(preprocess_image_bytes(path)) for path in image_paths)
    text_bytes = len(prepared["text_payload"].encode("utf-8"))
    metadata_bytes = len(prepared["task_id"].encode("utf-8")) + 512
    return (image_bytes * 4) // 3 + text_bytes + metadata_bytes


def build_gemini_inline_request(prepared: Dict[str, Any], model: str) -> Any:
    from google.genai import types

    return types.InlinedRequest(
        model=model,
        contents=build_gemini_contents(
            prepared["text_payload"],
            prepared["ref_paths"],
            prepared["image_a_path"],
            prepared["image_b_path"],
        ),
        config=build_gemini_generate_config(),
        metadata={"task_id": prepared["task_id"]},
    )


def chunk_gemini_prepared_items(
    prepared_items: Sequence[Dict[str, Any]],
    max_inline_bytes: int = GEMINI_BATCH_MAX_INLINE_BYTES,
) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_size = 0

    for prepared in prepared_items:
        request_size = estimate_gemini_request_size_bytes(prepared)
        if current_chunk and current_size + request_size > max_inline_bytes:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(prepared)
        current_size += request_size

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def gemini_job_state_name(batch: Any) -> str:
    state = getattr(batch, "state", None)
    return getattr(state, "value", str(state))


def poll_gemini_batch_until_terminal(client: Any, batch_name: str, poll_seconds: float) -> Any:
    while True:
        batch = client.batches.get(name=batch_name)
        state_name = gemini_job_state_name(batch)
        print(f"Batch {batch_name} state={state_name}")
        if state_name in GEMINI_BATCH_TERMINAL_STATES:
            return batch
        time.sleep(max(poll_seconds, 1.0))


def collect_gemini_batch_records(
    batch: Any,
    prepared_by_task_id: Dict[str, Dict[str, Any]],
    provider: str,
    model: str,
    completed_task_ids: Sequence[str],
    output_raw_path: Path,
    error_raw_path: Path,
) -> List[Dict[str, Any]]:
    completed_task_id_set = set(completed_task_ids)
    records: List[Dict[str, Any]] = []
    raw_payload = {
        "name": getattr(batch, "name", None),
        "state": gemini_job_state_name(batch),
        "src": model_to_jsonable(getattr(batch, "src", None)),
        "dest": model_to_jsonable(getattr(batch, "dest", None)),
        "error": model_to_jsonable(getattr(batch, "error", None)),
    }
    output_raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    src = getattr(batch, "src", None)
    dest = getattr(batch, "dest", None)
    requests = list(getattr(src, "inlined_requests", None) or [])
    responses = list(getattr(dest, "inlined_responses", None) or [])
    errors: List[Dict[str, Any]] = []

    for request_obj, response_obj in zip(requests, responses):
        metadata = getattr(request_obj, "metadata", None) or {}
        task_id = str(metadata.get("task_id", ""))
        if not task_id or task_id in completed_task_id_set:
            continue
        prepared = prepared_by_task_id.get(task_id)
        if prepared is None:
            continue

        error = getattr(response_obj, "error", None)
        if error is not None:
            errors.append({"task_id": task_id, "error": model_to_jsonable(error)})
            continue

        response = getattr(response_obj, "response", None)
        if response is None:
            errors.append({"task_id": task_id, "error": "Missing response"})
            continue

        raw_response_text = extract_gemini_response_text(response)
        parsed = extract_json_object(raw_response_text)
        normalized = normalize_result(parsed)
        record = build_record(prepared["item"], normalized, provider, model, raw_response_text)
        records.append(record)

    if errors:
        error_raw_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    return records


def run_gemini_batch_mode(
    args: argparse.Namespace,
    items: Sequence[Dict[str, Any]],
    base_dir: Path,
    output_path: Path,
    summary_path: Path,
    provider: str,
    model: str,
) -> None:
    from google import genai

    client = genai.Client(api_key=get_gemini_api_key())
    prepared_items = prepare_items(items, base_dir)
    prepared_by_task_id = {prepared["task_id"]: prepared for prepared in prepared_items}

    existing_records = load_existing_records(output_path)
    completed_task_ids = set(record.get("task_id") for record in existing_records if record.get("task_id"))
    if args.overwrite and output_path.exists():
        output_path.unlink()
        existing_records = []
        completed_task_ids = set()

    pending_prepared_items = [
        prepared for prepared in prepared_items if prepared["task_id"] not in completed_task_ids
    ]

    batch_metadata_path = output_path.with_suffix(".batch_meta.json")
    submitted_batch_names: List[str] = []

    if args.batch_id:
        submitted_batch_names = parse_batch_ids(args.batch_id)
        batches = [client.batches.get(name=batch_name) for batch_name in submitted_batch_names]
        for batch in batches:
            print(f"Loaded existing Gemini batch: name={batch.name} state={gemini_job_state_name(batch)}")
    else:
        if not pending_prepared_items:
            print("No pending items to submit.")
            if existing_records:
                write_summary(existing_records, summary_path)
                print(f"Saved summary to: {summary_path}")
            return

        chunks = chunk_gemini_prepared_items(pending_prepared_items)
        batches = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            inline_requests = [build_gemini_inline_request(prepared, model) for prepared in chunk]
            batch = client.batches.create(
                model=model,
                src=inline_requests,
                config={"display_name": f"vlm-eval-{safe_model_name(model)}-{chunk_index}"},
            )
            batches.append(batch)
            submitted_batch_names.append(str(batch.name))
            print(
                f"Created Gemini batch {chunk_index}/{len(chunks)}: "
                f"name={batch.name} state={gemini_job_state_name(batch)} requests={len(chunk)}"
            )

        metadata = {
            "provider": provider,
            "model": model,
            "api_mode": "batch",
            "batch_names": submitted_batch_names,
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "submitted_task_ids": [prepared["task_id"] for prepared in pending_prepared_items],
            "chunk_sizes": [len(chunk) for chunk in chunks],
        }
        write_batch_metadata(batch_metadata_path, metadata)
        print(f"Saved batch metadata to: {batch_metadata_path}")

    terminal_batches: List[Any] = []
    for batch in batches:
        batch_name = str(getattr(batch, "name", ""))
        if args.wait_for_batch:
            terminal_batches.append(
                poll_gemini_batch_until_terminal(client, batch_name, args.batch_poll_seconds)
            )
            continue

        state_name = gemini_job_state_name(batch)
        if state_name not in GEMINI_BATCH_TERMINAL_STATES:
            print(
                "Gemini batch submitted but not yet complete. "
                "Re-run with --batch-id "
                f"{','.join(submitted_batch_names)} --wait-for-batch to collect results."
            )
            return
        terminal_batches.append(client.batches.get(name=batch_name))

    new_records: List[Dict[str, Any]] = []
    for index, batch in enumerate(terminal_batches, start=1):
        batch_name = str(getattr(batch, "name", f"batch_{index}"))
        print(f"Final Gemini batch state: name={batch_name} state={gemini_job_state_name(batch)}")
        batch_output_raw_path = output_path.with_suffix(f".gemini_batch_{index}.raw.json")
        batch_error_raw_path = output_path.with_suffix(f".gemini_batch_{index}.errors.json")
        new_records.extend(
            collect_gemini_batch_records(
                batch=batch,
                prepared_by_task_id=prepared_by_task_id,
                provider=provider,
                model=model,
                completed_task_ids=completed_task_ids,
                output_raw_path=batch_output_raw_path,
                error_raw_path=batch_error_raw_path,
            )
        )

    if not new_records:
        print("No new completed records were collected from the Gemini batch job(s).")
        return

    with output_path.open("a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_records = existing_records + new_records
    write_summary(all_records, summary_path)
    print(f"Collected {len(new_records)} new records from Gemini batch job(s)")
    print(f"Saved JSONL to: {output_path}")
    print(f"Saved summary to: {summary_path}")


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
    api_mode = resolve_api_mode(args.provider, args.api_mode)

    script_dir = Path(__file__).resolve().parent
    load_local_env_files(script_dir)
    output_path = ensure_output_path(args.output, args.provider, model, script_dir / "results")
    summary_path = output_path.with_suffix(".summary.json")

    dataset_path = Path(args.dataset).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    items = load_dataset(dataset_path, base_dir=base_dir)
    items = select_items(items, args.only_task_id, args.max_samples)

    if not items:
        raise ValueError("No dataset items selected")

    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"API mode: {api_mode}")
    print(f"Dataset: {dataset_path}")
    print(f"Base dir: {base_dir}")
    print(f"Output: {output_path}")
    print(f"Selected samples: {len(items)}")
    if api_mode == "sync":
        run_sync_mode(args, items, base_dir, output_path, summary_path, args.provider, model)
        return

    if args.provider == "openai":
        run_openai_batch_mode(args, items, base_dir, output_path, summary_path, args.provider, model)
        return

    if args.provider == "gemini":
        run_gemini_batch_mode(args, items, base_dir, output_path, summary_path, args.provider, model)
        return

    raise ValueError(f"Unsupported provider: {args.provider}")


if __name__ == "__main__":
    main()
