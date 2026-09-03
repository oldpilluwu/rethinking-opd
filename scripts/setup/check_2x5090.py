#!/usr/bin/env python3
"""Validate a two-RTX-5090 OPD configuration and its visible CUDA runtime."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.config.load_opd_config import load_config  # noqa: E402


class HardwareError(RuntimeError):
    """Raised when the visible CUDA runtime is not the supported 2x5090 target."""


def cuda_version_tuple(value: str | None) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value or "")
    if not match:
        raise HardwareError(
            f"could not determine the PyTorch CUDA build version: {value!r}"
        )
    return int(match.group(1)), int(match.group(2))


def validate_hardware(
    gpus: list[dict[str, Any]], torch_cuda: str | None, bf16_supported: bool
) -> None:
    if len(gpus) != 2:
        raise HardwareError(
            f"expected exactly 2 CUDA-visible GPUs, found {len(gpus)}; "
            "set CUDA_VISIBLE_DEVICES to the two RTX 5090 device IDs"
        )

    for index, gpu in enumerate(gpus):
        name = str(gpu["name"])
        memory_gib = float(gpu["memory_gib"])
        capability = tuple(gpu["capability"])
        if "RTX 5090" not in name.upper():
            raise HardwareError(f"CUDA device {index} is not an RTX 5090: {name}")
        if memory_gib < 30.0:
            raise HardwareError(
                f"CUDA device {index} exposes only {memory_gib:.1f} GiB: {name}"
            )
        if capability != (12, 0):
            raise HardwareError(
                f"CUDA device {index} has compute capability {capability[0]}.{capability[1]}, expected 12.0: {name}"
            )

    if cuda_version_tuple(torch_cuda) < (12, 8):
        raise HardwareError(
            f"RTX 5090 (SM120) requires a PyTorch build with CUDA 12.8 or newer; found CUDA {torch_cuda}"
        )
    if not bf16_supported:
        raise HardwareError("the visible CUDA runtime does not report BF16 support")


def inspect_hardware() -> tuple[list[dict[str, Any]], str | None, bool]:
    import torch

    if not torch.cuda.is_available():
        raise HardwareError("PyTorch cannot see CUDA")

    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "name": properties.name,
                "memory_gib": properties.total_memory / 1024**3,
                "capability": torch.cuda.get_device_capability(index),
            }
        )
    return gpus, torch.version.cuda, torch.cuda.is_bf16_supported()


def validate_topology(config_path: Path) -> None:
    config = load_config(config_path)
    runtime = config["runtime"]
    if runtime["nodes"] != 1 or runtime["gpus_per_node"] != 2:
        raise HardwareError(
            "opd_2x5090.sh requires runtime.nodes=1 and runtime.gpus_per_node=2; "
            f"found nodes={runtime['nodes']}, gpus_per_node={runtime['gpus_per_node']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="validate only the experiment topology (used by CONFIG_ONLY=1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_topology(args.config)
    if args.skip_hardware:
        print(f"2x5090 topology valid: {args.config}")
        return

    gpus, torch_cuda, bf16_supported = inspect_hardware()
    validate_hardware(gpus, torch_cuda, bf16_supported)
    rendered = ", ".join(f"{gpu['name']} ({gpu['memory_gib']:.1f} GiB)" for gpu in gpus)
    print(
        f"2x5090 preflight passed: {rendered}; PyTorch CUDA {torch_cuda}; BF16 supported"
    )


if __name__ == "__main__":
    main()
