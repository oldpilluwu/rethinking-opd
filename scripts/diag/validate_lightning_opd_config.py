#!/usr/bin/env python3
"""Reject a resolved Hydra config that is not the scaled Lightning-OPD recipe."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml


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


def validate(config: dict[str, Any], smoke: bool) -> None:
    expected = {
        "algorithm.adv_estimator": "token_reward_direct",
        "algorithm.adv_clip_range": 10.0,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": "datasets/dapo-math-17k.parquet",
        "data.shuffle": True,
        "data.max_prompt_length": 1024,
        "data.return_raw_chat": True,
        "actor_rollout_ref.actor.optim.lr": 2e-6,
        "actor_rollout_ref.actor.optim.weight_decay": 0.1,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.98],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.rollout.temperature": 0.8,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_top_k": 0,
        "actor_rollout_ref.rollout.diagnostic_top_k": 16,
        "actor_rollout_ref.rollout.teacher_temperature": 1.0,
        "reward_model.enable": True,
        "reward_model.model.fsdp_config.param_offload": False,
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
        "trainer.test_freq": -1,
    }
    if not smoke:
        expected.update(
            {
                "data.train_batch_size": 64,
                "data.max_response_length": 4096,
                "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
                "trainer.total_training_steps": 150,
            }
        )

    errors: list[str] = []
    for key, wanted in expected.items():
        try:
            actual = select(config, key)
        except KeyError as error:
            errors.append(str(error))
            continue
        if not equal(actual, wanted):
            errors.append(f"{key}: expected {wanted!r}, got {actual!r}")

    template_kwargs = config.get("data", {}).get("apply_chat_template_kwargs", {}) or {}
    if template_kwargs.get("enable_thinking") is not False:
        errors.append("data.apply_chat_template_kwargs.enable_thinking must be false")

    if smoke:
        smoke_bounds = {
            "data.train_batch_size": 64,
            "data.max_response_length": 4096,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
            "trainer.total_training_steps": 3,
        }
        for key, upper_bound in smoke_bounds.items():
            actual = select(config, key)
            if not isinstance(actual, int) or actual <= 0 or actual > upper_bound:
                errors.append(f"{key}: invalid smoke value {actual!r}; expected 1..{upper_bound}")

    if errors:
        raise SystemExit("resolved Hydra config is not paper-faithful:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise SystemExit("resolved Hydra config is not a mapping")
    validate(config, args.smoke)
    print(f"validated resolved Hydra config: {args.config}")


if __name__ == "__main__":
    main()
