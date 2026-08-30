#!/usr/bin/env python3
"""Verify configured checkpoints and diagnostic outputs after an OPD run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.config.load_opd_config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]["name"]
    checkpoint_root = Path(config["experiment"]["output_dir"]) / experiment
    diagnostic_root = Path(config["experiment"]["diagnostics_dir"]) / experiment
    save_steps = config["checkpoints"]["save_steps"]
    optimizer_steps = set(config["checkpoints"]["optimizer_save_steps"])
    errors: list[str] = []

    for step in save_steps:
        step_root = checkpoint_root / f"global_step_{step}"
        manifest_path = step_root / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing checkpoint manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_resumable = step in optimizer_steps
        if bool(manifest.get("resumable")) != expected_resumable:
            errors.append(
                f"step {step}: resumable={manifest.get('resumable')}, expected {expected_resumable}"
            )

    expected_last = max(save_steps)
    latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file() or latest_path.read_text(encoding="utf-8").strip() != str(expected_last):
        errors.append(f"latest checkpoint marker is not {expected_last}: {latest_path}")

    required_diagnostics = (
        "metrics.jsonl",
        "gpu-telemetry.csv",
        "pip-freeze.txt",
        "git-commit.txt",
        "nvidia-smi-q.txt",
    )
    for name in required_diagnostics:
        if not (diagnostic_root / name).is_file():
            errors.append(f"missing diagnostic file: {diagnostic_root / name}")

    metrics_path = diagnostic_root / "metrics.jsonl"
    if metrics_path.is_file():
        metric_steps = {
            int(json.loads(line)["step"])
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_metric_steps = sorted(set(range(1, config["training"]["steps"] + 1)) - metric_steps)
        if missing_metric_steps:
            errors.append(f"metrics JSONL is missing steps: {missing_metric_steps}")

    if config["tracking"]["dump_rollouts"]:
        expected_rows = config["training"]["train_batch_size"] * config["rollout"]["responses_per_prompt"]
        for step in config["tracking"]["rollout_data_steps"]:
            rollout_path = diagnostic_root / "rollouts" / f"{step}.jsonl"
            if not rollout_path.is_file():
                errors.append(f"missing rollout dump: {rollout_path}")
                continue
            rows = sum(1 for line in rollout_path.open(encoding="utf-8") if line.strip())
            if rows != expected_rows:
                errors.append(f"{rollout_path}: expected {expected_rows} rows, got {rows}")

    if errors:
        raise SystemExit("run output verification failed:\n- " + "\n- ".join(errors))
    print(f"verified {len(save_steps)} checkpoints and diagnostics through step {expected_last}")
    print(f"optimizer/resume state exists only at steps {sorted(optimizer_steps)}")


if __name__ == "__main__":
    main()
