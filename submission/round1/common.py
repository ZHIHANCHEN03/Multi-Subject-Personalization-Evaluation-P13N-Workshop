"""Shared utilities for Round-1: data, MIE critic (pluggable), SCR metric, driver.

Nothing here trains anything. MIE is a frozen verifier used *inside* the loop;
SCR (DINOv2) is the *independent* judge used only to score final images. Keep
those two separate (never grade with the same signal you optimize).
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
import atexit
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


@dataclass
class Task:
    task_id: str
    prompt: str
    subject_names: list[str]
    subject_ref_paths: list[list[str]]
    meta: dict = field(default_factory=dict)

    @property
    def num_subjects(self) -> int:
        return len(self.subject_names)

    def flat_ref_paths(self) -> list[str]:
        out = []
        for paths in self.subject_ref_paths:
            for p in paths:
                if os.path.exists(p):
                    out.append(p)
                    break
        return out

    def load_refs(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.flat_ref_paths()]


def load_tasks(path: str, limit: Optional[int] = None) -> list[Task]:
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
                    subject_names=[s.get("name", f"S{i+1}") for i, s in enumerate(subs)],
                    subject_ref_paths=[list(s.get("ref_images", [])) for s in subs],
                    meta=r.get("meta", {}),
                )
            )
    if limit:
        tasks = tasks[:limit]
    return tasks


def omnigen_prompt(task: Task, prompt: Optional[str] = None) -> str:
    """Make the reference-to-entity binding explicit for OmniGen2/UMO.

    OmniGen2's official usage guidance recommends naming image ordinals in the
    instruction. All OmniGen2-based methods use this exact formatter so their
    comparison is controlled.
    """
    bindings = "; ".join(
        f"image {i + 1} is the reference for {name.replace('_', ' ')}"
        for i, name in enumerate(task.subject_names)
    )
    return f"{bindings}. {prompt or task.prompt}"


# ----------------------------------------------------------------------------
# MIE critic (frozen verifier used inside the loop).
#   Set only MIE_CKPT. The Qwen/Unsloth model runs in an isolated persistent
#   subprocess because its runtime conflicts with OmniGen2's pinned stack.
#   Mock fallback exists only for dependency-free wiring tests.
# ----------------------------------------------------------------------------

DIMS = ["existence", "appearance", "interaction"]


class BaseCritic:
    def score(self, image: Image.Image, task: Task) -> dict:
        """Return {'total': float, 'existence':.., 'appearance':.., 'interaction':..}.
        total = MIE preference score (comparison-only, higher=better).
        dims  = sigmoid logits in [0,1] (higher=better)."""
        raise NotImplementedError


class MockCritic(BaseCritic):
    """Deterministic pseudo-scores from image stats. Only for wiring/dry-runs."""

    def score(self, image: Image.Image, task: Task) -> dict:
        import hashlib

        small = image.resize((16, 16)).convert("L")
        px = list(small.getdata())
        h = int(hashlib.md5(bytes(px)).hexdigest(), 16)
        rng = [(h >> (i * 8)) & 0xFF for i in range(4)]
        e = 0.55 + (rng[0] / 255) * 0.4
        a = 0.45 + (rng[1] / 255) * 0.4
        it = 0.35 + (rng[2] / 255) * 0.4
        return {
            "total": float(e + a + it),
            "existence": float(min(e, 1.0)),
            "appearance": float(min(a, 1.0)),
            "interaction": float(min(it, 1.0)),
        }


class MieSubprocessCritic(BaseCritic):
    """Persistent client for the real Qwen-based MIE in an isolated venv."""

    READY_PREFIX = "MIE_READY\t"
    RESULT_PREFIX = "MIE_RESULT\t"

    def __init__(self, ckpt: str):
        round1_dir = Path(__file__).resolve().parent
        repo_root = round1_dir.parent
        python = os.environ.get(
            "MIE_PYTHON", str(repo_root / ".venvs" / "mie" / "bin" / "python")
        )
        if not Path(python).exists():
            raise FileNotFoundError(
                f"MIE Python not found at {python}; run round1/setup_round1.sh"
            )
        print(
            f"[MIE START] runtime={python} checkpoint={Path(ckpt).resolve()} "
            "status=loading_qwen_backbone_and_heads",
            flush=True,
        )
        self.temp_dir = WORK_DIR / ".mie_tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                python,
                str(round1_dir / "mie_server.py"),
                "--checkpoint",
                str(Path(ckpt).expanduser().resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        ready = self._read_prefixed(self.READY_PREFIX)
        if not ready.get("ok"):
            self.close()
            raise RuntimeError(f"MIE failed to load: {ready.get('error')}")
        print(
            f"[critic] Qwen MIE ready: base={ready.get('base_model')} "
            f"checkpoint={ready.get('checkpoint')}"
        )
        atexit.register(self.close)

    def _read_prefixed(self, prefix: str) -> dict:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise RuntimeError(f"MIE server exited unexpectedly (code={code})")
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])

    def score(self, image: Image.Image, task: Task) -> dict:
        image_path = self.temp_dir / f"{task.task_id}-{uuid.uuid4().hex}.png"
        image.save(image_path)
        request = {
            "command": "score",
            "task_id": task.task_id,
            "prompt": task.prompt,  # raw training-template prompt, not Omni prompt
            "ref_paths": task.flat_ref_paths(),
            "image_path": str(image_path),
        }
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            response = self._read_prefixed(self.RESULT_PREFIX)
            if not response.get("ok"):
                raise RuntimeError(
                    f"MIE score failed for {task.task_id}: {response.get('error')}"
                )
            return {key: float(response[key]) for key in ["total", *DIMS]}
        finally:
            image_path.unlink(missing_ok=True)

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.terminate()
        finally:
            self.proc = None


def build_critic() -> BaseCritic:
    ckpt = os.environ.get("MIE_CKPT", "")
    if ckpt:
        return MieSubprocessCritic(ckpt)
    print("[critic] MIE_CKPT not set; using MockCritic for dry-run only")
    return MockCritic()


# ----------------------------------------------------------------------------
# SCR (DINOv2) — the INDEPENDENT judge. Never used to drive the loop.
#   Grounding-DINO locates each requested subject; DINOv2 compares that crop
#   against the corresponding reference. Missing detections count as collapse.
#   SCR = fraction of missing/low-similarity subjects (lower = better).
# ----------------------------------------------------------------------------


class DinoScorer:
    """Detection-aware subject identity scorer used independently of MIE."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        detector_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        local_detector = Path(__file__).resolve().parents[1] / "models" / "grounding-dino-tiny"
        detector_name = detector_name or os.environ.get(
            "GROUNDING_DINO",
            str(local_detector) if local_detector.exists()
            else "IDEA-Research/grounding-dino-tiny",
        )
        self.det_proc = AutoProcessor.from_pretrained(detector_name)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_name
        ).to(self.device).eval()

    def embed(self, image: Image.Image):
        import torch

        with torch.no_grad():
            inp = self.proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            out = self.model(**inp)
            cls = out.last_hidden_state[:, 0]  # CLS token
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls

    def sims(self, image: Image.Image, refs: list[Image.Image]) -> list[float]:
        if not refs:
            return []
        img_e = self.embed(image)
        out = []
        for r in refs:
            ref_e = self.embed(r)
            out.append(float((img_e * ref_e).sum().item()))
        return out

    def _subject_crop(self, image: Image.Image, subject_name: str) -> Optional[Image.Image]:
        import torch

        text = subject_name.replace("_", " ") + "."
        with torch.no_grad():
            inputs = self.det_proc(
                images=image.convert("RGB"), text=text, return_tensors="pt"
            ).to(self.device)
            outputs = self.detector(**inputs)
            result = self.det_proc.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=float(os.environ.get("DET_BOX_THRESH", "0.25")),
                text_threshold=float(os.environ.get("DET_TEXT_THRESH", "0.20")),
                target_sizes=[image.size[::-1]],
            )[0]
        if len(result["boxes"]) == 0:
            return None
        idx = int(result["scores"].argmax().item())
        x1, y1, x2, y2 = result["boxes"][idx].tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    def score_task(self, image: Image.Image, task: Task) -> tuple[list[float], int]:
        """Return per-subject ref↔detected-crop similarity and missing count."""
        refs = task.load_refs()
        sims, missing = [], 0
        for name, ref in zip(task.subject_names, refs):
            crop = self._subject_crop(image, name)
            if crop is None:
                sims.append(-1.0)  # missing subject always counts as collapse
                missing += 1
                continue
            img_e = self.embed(crop)
            ref_e = self.embed(ref)
            sims.append(float((img_e * ref_e).sum().item()))
        return sims, missing


def scr_from_sims(sims: list[float], thresh: float) -> float:
    if not sims:
        return 0.0
    collapsed = sum(1 for s in sims if s < thresh)
    return collapsed / len(sims)


# ----------------------------------------------------------------------------
# Driver: run one method over the dataset, score with SCR, dump records.
# ----------------------------------------------------------------------------

ROUND1_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("ROUND1_WORK", str(ROUND1_DIR / "results")))
SCR_THRESH = float(os.environ.get("SCR_THRESH", "0.5"))


def run_over_dataset(
    name: str,
    method_fn: Callable[[Task], tuple[Image.Image, dict]],
    tasks: list[Task],
    scorer: Optional[DinoScorer],
    save_images: bool = True,
    continue_on_error: bool = False,
):
    """method_fn(task) -> (final_image, info_dict). Scores each output with SCR."""
    out_dir = WORK_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if save_images:
        img_dir.mkdir(exist_ok=True)

    rec_path = out_dir / "records.jsonl"
    started = time.monotonic()
    print(
        f"[METHOD START] name={name} tasks={len(tasks)} "
        f"save_images={save_images} scr_enabled={scorer is not None}",
        flush=True,
    )

    def duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def show_progress(index: int, task_id: str, status: str, rec: dict) -> None:
        done = index + 1
        elapsed = time.monotonic() - started
        eta = elapsed / done * (len(tasks) - done) if done else 0
        details = []
        if rec.get("scr") is not None:
            details.append(f"SCR={rec['scr']:.3f}")
        if rec.get("dino_mean") is not None:
            details.append(f"DINO={rec['dino_mean']:.3f}")
        if rec.get("accepted_steps") is not None:
            details.append(f"accepted={rec['accepted_steps']}")
        suffix = " ".join(details)
        print(
            f"[PROGRESS][{name}] {done:03d}/{len(tasks):03d} "
            f"({100 * done / len(tasks):5.1f}%) task={task_id} status={status} "
            f"elapsed={duration(elapsed)} eta={duration(eta)} {suffix}".rstrip(),
            flush=True,
        )

    with open(rec_path, "w", encoding="utf-8") as fout:
        for i, task in enumerate(tasks):
            print(
                f"[TASK START][{name}] {i+1:03d}/{len(tasks):03d} "
                f"task={task.task_id} entities={task.num_subjects}",
                flush=True,
            )
            try:
                image, info = method_fn(task)
            except Exception as exc:
                if not continue_on_error:
                    raise
                rec = {
                    "task_id": task.task_id,
                    "method": name,
                    "num_subjects": task.num_subjects,
                    "meta": task.meta,
                    "generation_failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "dino_sims": [-1.0] * task.num_subjects,
                    "dino_mean": -1.0,
                    "scr": 1.0,
                    "missing_subjects": task.num_subjects,
                    "detection_recall": 0.0,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{name}] FAILED {task.task_id}: {rec['error']}")
                show_progress(i, task.task_id, "FAILED", rec)
                continue
            rec = {
                "task_id": task.task_id,
                "method": name,
                "num_subjects": task.num_subjects,
                "meta": task.meta,
                **info,
            }
            if scorer is not None:
                sims, missing = scorer.score_task(image, task)
                rec["dino_sims"] = sims
                rec["dino_mean"] = (sum(sims) / len(sims)) if sims else None
                rec["scr"] = scr_from_sims(sims, SCR_THRESH)
                rec["missing_subjects"] = missing
                rec["detection_recall"] = (
                    1.0 - missing / task.num_subjects if task.num_subjects else 1.0
                )
            if save_images:
                p = img_dir / f"{task.task_id}.png"
                image.save(p)
                rec["image_path"] = str(p)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()  # flush per-record so live monitoring can read partial results
            show_progress(i, task.task_id, "OK", rec)
    print(
        f"[METHOD END] name={name} tasks={len(tasks)} "
        f"elapsed={duration(time.monotonic() - started)} records={rec_path}",
        flush=True,
    )
    return rec_path
