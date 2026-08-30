#!/usr/bin/env python3
"""Compare a resolved Hydra job with its source OPD experiment config."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.config.load_opd_config import ConfigError, expected_hydra_values, load_config  # noqa: E402


def select(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"resolved config is missing {dotted_key}")
        value = value[key]
    return value


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate(resolved: dict[str, Any], experiment: dict[str, Any]) -> None:
    errors: list[str] = []
    for key, expected in expected_hydra_values(experiment).items():
        try:
            actual = select(resolved, key)
        except KeyError as error:
            errors.append(str(error))
            continue
        if not equal(actual, expected):
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    if errors:
        raise SystemExit("resolved Hydra config differs from the experiment TOML:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resolved_config", type=Path)
    parser.add_argument("experiment_config", type=Path)
    args = parser.parse_args()

    with args.resolved_config.open("r", encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream)
    if not isinstance(resolved, dict):
        raise SystemExit("resolved Hydra config is not a mapping")

    try:
        experiment = load_config(args.experiment_config)
    except ConfigError as error:
        raise SystemExit(f"invalid OPD experiment config: {error}") from error
    validate(resolved, experiment)
    print(f"validated resolved Hydra config: {args.resolved_config}")


if __name__ == "__main__":
    main()
