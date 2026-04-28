#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# 2.5
# DEFAULT_RANGES = [
#     (8001, 28000, "GEMINI_API_KEY_1"),
#     (28001, 40000, "GEMINI_API_KEY_2"),
#     (40001, 60000, "GEMINI_API_KEY_3"),
# ]

# 3.1
# DEFAULT_RANGES = [
#     # (50001,60000,"GEMINI_API_KEY_4"),
#     (53928,60000,"GEMINI_API_KEY_2"),
# ]

# 2.5
# DEFAULT_RANGES = [
#     (22290, 28000, "GEMINI_API_KEY_1"),
#     (32192, 40000, "GEMINI_API_KEY_2"),
#     (44190, 50000, "GEMINI_API_KEY_3"),
#     (52336, 60000, "GEMINI_API_KEY_4"),
# ]

# 3.1
# DEFAULT_RANGES = [
#     (45677, 50000, "GEMINI_API_KEY_1"),
#     (54606, 60000, "GEMINI_API_KEY_3"),
# ]

# 2.5
# DEFAULT_RANGES = [
#     (55359, 60000, "GEMINI_API_KEY_1"),
# ]


DEFAULT_RANGES = [
    (38493, 40000, "GEMINI_API_KEY_1"),
]
def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Launch parallel Gemini eval jobs with different API keys and task_id ranges."
    )
    parser.add_argument(
        "--run-script",
        default=str(script_path.parent / "run_vlm_eval.py"),
        help="Path to run_vlm_eval.py",
    )
    parser.add_argument(
        "--model",
        # default="gemini-3.1-flash-lite-preview",
        default="gemini-2.5-flash",
        help="Gemini model name to pass to run_vlm_eval.py",
    )
    parser.add_argument(
        "--api-mode",
        choices=["sync", "batch", "auto"],
        default="batch",
        help="Execution mode to pass through",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_path.parent / "results"),
        help="Directory to store output JSONL and logs",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch child jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without launching them",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[Tuple[str, List[str], Path]]:
    run_script = Path(args.run_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, List[str], Path]] = []
    for min_task_id, max_task_id, key_env in DEFAULT_RANGES:
        safe_model = args.model.replace(".", "_").replace("-", "_")
        output_path = output_dir / f"{safe_model}_{min_task_id}_{max_task_id}.jsonl"
        log_path = output_dir / f"{safe_model}_{min_task_id}_{max_task_id}.log"
        command = [
            args.python_bin,
            str(run_script),
            "--provider",
            "gemini",
            "--gemini-model",
            args.model,
            "--gemini-api-key-env",
            key_env,
            "--api-mode",
            args.api_mode,
            "--min-task-id",
            str(min_task_id),
            "--max-task-id",
            str(max_task_id),
            "--output",
            str(output_path),
        ]
        jobs.append((key_env, command, log_path))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)

    for index, (key_env, command, log_path) in enumerate(jobs, start=1):
        printable = " ".join(shlex.quote(part) for part in command)
        print(f"[job {index}] key={key_env} log={log_path}")
        print(printable)
        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
        print(f"  started pid={process.pid}")


if __name__ == "__main__":
    main()
