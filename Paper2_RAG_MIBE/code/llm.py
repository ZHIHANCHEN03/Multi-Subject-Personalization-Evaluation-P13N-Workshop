"""Thin, dependency-light client for an OpenAI-compatible chat API.

Used by (a) the vlm_judge critic backend and (b) the LLM-rewriter action
variant and (c) the independent GPT-4o judge. If no API is configured the
callers are expected to fall back to rule-based / mock paths.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Optional

import config


class LLMNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(config.LLM_API_BASE and config.LLM_API_KEY)


def _encode_image(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Send a chat completion request. `messages` follows the OpenAI schema."""
    if not is_configured():
        raise LLMNotConfigured(
            "LLM_API_BASE / LLM_API_KEY not set. Set them to use vlm_judge or "
            "the LLM-rewriter action, or use rule-based / mock backends."
        )
    import requests  # local import: only needed when an API is actually used

    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_with_images(
    text: str,
    images: list,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Multimodal chat: one text block + N images (PIL)."""
    content: list[dict] = [{"type": "text", "text": text}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(img)}"},
            }
        )
    return chat(
        [{"role": "user", "content": content}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def parse_json(raw: str) -> dict:
    """Best-effort extraction of the first JSON object from a model reply."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
