"""MIB-Gold data loader.

Expected JSONL schema (one task per line):
    {
      "task_id": "mib_000123",
      "prompt": "a photo of <S1> and <S2> shaking hands",
      "subjects": [
        {"name": "S1", "ref_images": ["/abs/path/a.png", ...]},
        {"name": "S2", "ref_images": ["/abs/path/b.png", ...]}
      ]
    }

You have not shipped the data yet, so `load_dataset` also supports a built-in
mock set (deterministic) so the pipeline runs end-to-end without it. Point
MISC_DATA at your real JSONL to switch over.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class Task:
    task_id: str
    prompt: str
    subject_names: list[str]
    subject_ref_paths: list[list[str]]  # per-subject list of ref image paths
    meta: dict = field(default_factory=dict)

    @property
    def num_subjects(self) -> int:
        return len(self.subject_names)

    def load_refs(self) -> list:
        """Flat list of PIL ref images (first ref per subject by default)."""
        refs = []
        for paths in self.subject_ref_paths:
            for p in paths:
                if os.path.exists(p):
                    refs.append(Image.open(p).convert("RGB"))
                    break
        return refs

    def refs_by_subject(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for name, paths in zip(self.subject_names, self.subject_ref_paths):
            imgs = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
            out[name] = imgs
        return out


def _mock_tasks(n: int = 8) -> list[Task]:
    subjects_pool = [
        ("a corgi", "a tabby cat"),
        ("a red robot", "a blue robot"),
        ("a woman in a yellow dress", "a man in a suit"),
        ("a teddy bear", "a wooden toy train"),
    ]
    interactions = ["shaking hands", "sitting side by side", "facing each other", "hugging"]
    tasks = []
    for i in range(n):
        s1, s2 = subjects_pool[i % len(subjects_pool)]
        inter = interactions[i % len(interactions)]
        tasks.append(
            Task(
                task_id=f"mock_{i:04d}",
                prompt=f"{s1} and {s2} {inter}, high quality photo",
                subject_names=["S1", "S2"],
                subject_ref_paths=[[], []],  # no real refs in mock mode
                meta={"mock": True},
            )
        )
    return tasks


def load_dataset(path: Optional[str] = None, limit: Optional[int] = None) -> list[Task]:
    path = path or os.environ.get("MISC_DATA", "")
    if not path:
        tasks = _mock_tasks()
    else:
        tasks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                subs = r.get("subjects", [])
                tasks.append(
                    Task(
                        task_id=str(r["task_id"]),
                        prompt=r["prompt"],
                        subject_names=[s.get("name", f"S{j+1}") for j, s in enumerate(subs)],
                        subject_ref_paths=[list(s.get("ref_images", [])) for s in subs],
                        meta={k: v for k, v in r.items() if k not in {"task_id", "prompt", "subjects"}},
                    )
                )
    if limit:
        tasks = tasks[:limit]
    return tasks
