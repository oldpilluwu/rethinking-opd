#!/usr/bin/env python3
"""Audit the CUDA/Python environment and optionally the experiment assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.config.load_opd_config import load_config  # noqa: E402


MODULES = (
    "torch",
    "transformers",
    "vllm",
    "ray",
    "numpy",
    "scipy",
    "cv2",
    "cupy",
    "datasets",
    "fsspec",
    "pyarrow",
    "pandas",
    "swanlab",
    "math_verify",
)


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        package_name = "opencv-python" if module_name == "cv2" else module_name.replace("_", "-")
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        report["packages"][module_name] = str(version)

    import torch
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        raise RuntimeError("transformers does not export AutoModelForVision2Seq")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA")

    gpu_rows = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    report["gpus"] = gpu_rows
    if not any("A100" in row for row in gpu_rows):
        raise RuntimeError(f"expected an A100, found: {gpu_rows}")
    if max(int(row.split(",")[1].strip()) for row in gpu_rows) < 75_000:
        raise RuntimeError(f"expected an 80 GB-class A100, found: {gpu_rows}")
    report["cuda"] = {
        "torch_cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }

    report["pip_check"] = run([sys.executable, "-m", "pip", "check"])
    report["git"] = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status": run(["git", "status", "--short"]),
    }

    if args.config:
        config = load_config(args.config)
        assets = [
            Path(config["models"]["student"]),
            Path(config["models"]["teacher"]),
            Path(config["data"]["train_file"]),
            *(Path(item) for item in config["data"]["validation_files"]),
        ]
        missing = [str(path) for path in assets if not path.exists()]
        report["config"] = str(args.config)
        asset_report: dict[str, Any] = {}
        for path in assets:
            details: dict[str, Any] = {"exists": path.exists(), "kind": "missing"}
            if path.is_file():
                details.update(kind="file", size_bytes=path.stat().st_size, sha256=sha256(path))
            elif path.is_dir():
                details["kind"] = "directory"
                for metadata_name in ("config.json", "tokenizer_config.json"):
                    metadata = path / metadata_name
                    if metadata.is_file():
                        details[f"{metadata_name}_sha256"] = sha256(metadata)
            asset_report[str(path)] = details
        revisions = Path("model/paper-model-revisions.txt")
        if revisions.is_file():
            report["model_revisions"] = revisions.read_text(encoding="utf-8").splitlines()
        report["assets"] = asset_report
        if args.require_assets and missing:
            raise FileNotFoundError(f"missing experiment assets: {missing}")

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
