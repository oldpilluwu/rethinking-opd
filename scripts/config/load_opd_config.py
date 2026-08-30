#!/usr/bin/env python3
"""Load, validate, and translate a reproducible OPD TOML experiment config."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
METHODS = {"sampled_token", "top_k"}
TOP_K_STRATEGIES = {"only_stu", "only_tch", "intersection", "union", "union-intersection"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}
LOSS_AGGREGATIONS = {
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
}
PREFLIGHT_MODES = {"lightning_dapo", "paper_dapo", "exists", "none"}
SECTION_KEYS = {
    "experiment": {"name", "project_name", "output_dir", "log_dir", "validation_dir"},
    "models": {
        "student",
        "teacher",
        "enable_thinking",
        "model_dtype",
        "teacher_dtype",
        "gradient_checkpointing",
        "activation_offload",
    },
    "data": {
        "name",
        "train_file",
        "validation_files",
        "shuffle",
        "seed",
        "max_prompt_length",
        "max_response_length",
        "filter_overlong_prompts",
        "truncation",
        "preflight",
    },
    "rollout": {
        "engine",
        "responses_per_prompt",
        "do_sample",
        "temperature",
        "teacher_temperature",
        "sampling_top_k",
        "top_p",
        "repetition_penalty",
        "ignore_eos",
    },
    "objective": {
        "adv_estimator",
        "method",
        "top_k",
        "diagnostic_top_k",
        "top_k_strategy",
        "reward_weight_mode",
        "advantage_clip",
        "loss_aggregation",
        "entropy_coefficient",
        "use_kl_loss",
        "kl_loss_coefficient",
        "kl_loss_type",
        "use_kl_in_reward",
        "kl_reward_coefficient",
        "grpo_outcome_weight",
    },
    "optimizer": {
        "learning_rate",
        "scheduler",
        "warmup_ratio",
        "weight_decay",
        "betas",
        "epsilon",
        "gradient_clip",
    },
    "training": {
        "steps",
        "epochs",
        "train_batch_size",
        "mini_batch_size",
        "ppo_epochs",
        "shuffle_minibatches",
        "ppo_micro_batch_size_per_gpu",
        "reward_micro_batch_size_per_gpu",
    },
    "runtime": {
        "gpus_per_node",
        "nodes",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "actor_max_tokens_per_gpu",
        "teacher_max_tokens_per_gpu",
        "gpu_memory_utilization",
        "actor_param_offload",
        "optimizer_offload",
        "teacher_param_offload",
        "forward_prefetch",
    },
    "checkpoints": {"save_steps", "optimizer_save_steps"},
    "validation": {
        "before_training",
        "frequency",
        "max_response_length",
        "responses_per_prompt",
        "temperature",
        "top_k",
        "top_p",
        "do_sample",
        "prompt_suffix",
        "grader",
        "seed",
        "gpu_memory_utilization",
        "model_dtype",
    },
    "reward": {"enable_format_reward", "custom_function_path", "custom_function_name"},
    "tracking": {"loggers", "swanlab_mode", "is_plot", "opd_text_diagnostics"},
    "hydra": {"extra_overrides"},
}


class ConfigError(ValueError):
    """Raised when an experiment config is incomplete or internally inconsistent."""


def _get(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing required setting: {dotted_key}")
        value = value[key]
    return value


def _require_type(config: dict[str, Any], dotted_key: str, expected: type | tuple[type, ...]) -> Any:
    value = _get(config, dotted_key)
    if isinstance(value, bool) and expected in (int, float, (int, float)):
        raise ConfigError(f"{dotted_key} must be numeric, not boolean")
    if not isinstance(value, expected):
        expected_name = (
            "/".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ConfigError(f"{dotted_key} must be {expected_name}, got {type(value).__name__}")
    return value


def _require_positive_int(config: dict[str, Any], dotted_key: str) -> int:
    value = _require_type(config, dotted_key, int)
    if value <= 0:
        raise ConfigError(f"{dotted_key} must be greater than zero")
    return value


def _require_nonnegative_int(config: dict[str, Any], dotted_key: str) -> int:
    value = _require_type(config, dotted_key, int)
    if value < 0:
        raise ConfigError(f"{dotted_key} must be non-negative")
    return value


def _require_number(config: dict[str, Any], dotted_key: str, *, minimum: float | None = None) -> float:
    value = _require_type(config, dotted_key, (int, float))
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{dotted_key} must be finite")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{dotted_key} must be at least {minimum}")
    return value


def _require_choice(config: dict[str, Any], dotted_key: str, choices: set[str]) -> str:
    value = _require_type(config, dotted_key, str)
    if value not in choices:
        raise ConfigError(f"{dotted_key} must be one of {sorted(choices)}, got {value!r}")
    return value


def _require_list(config: dict[str, Any], dotted_key: str, item_type: type) -> list[Any]:
    value = _require_type(config, dotted_key, list)
    if not all(isinstance(item, item_type) and not isinstance(item, bool) for item in value):
        expected_name = (
            "/".join(item.__name__ for item in item_type)
            if isinstance(item_type, tuple)
            else item_type.__name__
        )
        raise ConfigError(f"every item in {dotted_key} must be {expected_name}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"config file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error

    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    allowed_top_level = {"schema_version", *SECTION_KEYS}
    unknown_top_level = sorted(set(config) - allowed_top_level)
    if unknown_top_level:
        raise ConfigError(f"unknown top-level settings: {unknown_top_level}")
    for section, allowed_keys in SECTION_KEYS.items():
        section_value = _require_type(config, section, dict)
        unknown_keys = sorted(set(section_value) - allowed_keys)
        if unknown_keys:
            raise ConfigError(f"unknown settings in [{section}]: {unknown_keys}")

    if _require_type(config, "schema_version", int) != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")

    string_fields = (
        "experiment.name",
        "experiment.project_name",
        "experiment.output_dir",
        "experiment.log_dir",
        "experiment.validation_dir",
        "models.student",
        "models.teacher",
        "models.model_dtype",
        "models.teacher_dtype",
        "data.name",
        "data.train_file",
        "data.truncation",
        "rollout.engine",
        "objective.adv_estimator",
        "optimizer.scheduler",
        "objective.kl_loss_type",
        "tracking.swanlab_mode",
        "reward.custom_function_path",
        "reward.custom_function_name",
        "validation.prompt_suffix",
        "validation.grader",
        "validation.model_dtype",
    )
    for field in string_fields:
        if not _require_type(config, field, str):
            raise ConfigError(f"{field} must not be empty")

    boolean_fields = (
        "models.enable_thinking",
        "models.gradient_checkpointing",
        "models.activation_offload",
        "data.shuffle",
        "data.filter_overlong_prompts",
        "rollout.do_sample",
        "rollout.ignore_eos",
        "objective.use_kl_loss",
        "objective.use_kl_in_reward",
        "reward.enable_format_reward",
        "runtime.actor_param_offload",
        "runtime.optimizer_offload",
        "runtime.teacher_param_offload",
        "runtime.forward_prefetch",
        "training.shuffle_minibatches",
        "validation.before_training",
        "validation.do_sample",
        "tracking.is_plot",
        "tracking.opd_text_diagnostics",
    )
    for field in boolean_fields:
        _require_type(config, field, bool)

    positive_int_fields = (
        "data.max_prompt_length",
        "data.max_response_length",
        "rollout.responses_per_prompt",
        "training.steps",
        "training.epochs",
        "training.train_batch_size",
        "training.mini_batch_size",
        "training.ppo_epochs",
        "training.ppo_micro_batch_size_per_gpu",
        "training.reward_micro_batch_size_per_gpu",
        "runtime.gpus_per_node",
        "runtime.nodes",
        "runtime.tensor_parallel_size",
        "runtime.sequence_parallel_size",
        "runtime.actor_max_tokens_per_gpu",
        "runtime.teacher_max_tokens_per_gpu",
        "validation.max_response_length",
        "validation.responses_per_prompt",
    )
    for field in positive_int_fields:
        _require_positive_int(config, field)

    _require_nonnegative_int(config, "data.seed")
    _require_nonnegative_int(config, "validation.seed")
    _require_type(config, "validation.frequency", int)
    sampling_top_k = _require_type(config, "rollout.sampling_top_k", int)
    validation_top_k = _require_type(config, "validation.top_k", int)
    if sampling_top_k < -1 or validation_top_k < -1:
        raise ConfigError("rollout.sampling_top_k and validation.top_k must be at least -1")
    _require_nonnegative_int(config, "objective.top_k")
    _require_nonnegative_int(config, "objective.diagnostic_top_k")

    nonnegative_number_fields = (
        "objective.advantage_clip",
        "objective.entropy_coefficient",
        "objective.kl_loss_coefficient",
        "objective.kl_reward_coefficient",
        "objective.grpo_outcome_weight",
        "optimizer.weight_decay",
        "optimizer.warmup_ratio",
        "optimizer.epsilon",
        "optimizer.gradient_clip",
    )
    for field in nonnegative_number_fields:
        _require_number(config, field, minimum=0.0)

    positive_number_fields = (
        "rollout.temperature",
        "rollout.teacher_temperature",
        "rollout.repetition_penalty",
        "optimizer.learning_rate",
        "runtime.gpu_memory_utilization",
        "validation.gpu_memory_utilization",
    )
    for field in positive_number_fields:
        value = _require_number(config, field, minimum=0.0)
        if value == 0.0:
            raise ConfigError(f"{field} must be greater than zero")

    for field in ("rollout.top_p", "validation.top_p"):
        value = _require_number(config, field, minimum=0.0)
        if not 0.0 < value <= 1.0:
            raise ConfigError(f"{field} must be in (0, 1]")
    _require_number(config, "validation.temperature", minimum=0.0)

    betas = _require_list(config, "optimizer.betas", (int, float))
    if len(betas) != 2 or not all(0.0 <= float(beta) < 1.0 for beta in betas):
        raise ConfigError("optimizer.betas must contain exactly two values in [0, 1)")

    save_steps = _require_list(config, "checkpoints.save_steps", int)
    optimizer_steps = _require_list(config, "checkpoints.optimizer_save_steps", int)
    if save_steps != sorted(set(save_steps)) or any(step <= 0 for step in save_steps):
        raise ConfigError("checkpoints.save_steps must be sorted, unique, and positive")
    if optimizer_steps != sorted(set(optimizer_steps)) or any(step <= 0 for step in optimizer_steps):
        raise ConfigError("checkpoints.optimizer_save_steps must be sorted, unique, and positive")
    missing_optimizer_steps = sorted(set(optimizer_steps) - set(save_steps))
    if missing_optimizer_steps:
        raise ConfigError(
            "optimizer checkpoint steps must also appear in checkpoints.save_steps: "
            f"{missing_optimizer_steps}"
        )
    total_steps = _get(config, "training.steps")
    out_of_range = [step for step in save_steps if step > total_steps]
    if out_of_range:
        raise ConfigError(f"checkpoint steps exceed training.steps={total_steps}: {out_of_range}")

    validation_files = _require_list(config, "data.validation_files", str)
    if not validation_files:
        raise ConfigError("data.validation_files must contain at least one path")
    loggers = _require_list(config, "tracking.loggers", str)
    if not loggers:
        raise ConfigError("tracking.loggers must contain at least one logger")
    _require_list(config, "hydra.extra_overrides", str)

    method = _require_choice(config, "objective.method", METHODS)
    top_k = _get(config, "objective.top_k")
    if method == "sampled_token" and top_k != 0:
        raise ConfigError("objective.method='sampled_token' requires objective.top_k=0")
    if method == "top_k" and top_k <= 0:
        raise ConfigError("objective.method='top_k' requires objective.top_k>0")
    _require_choice(config, "objective.top_k_strategy", TOP_K_STRATEGIES)
    _require_choice(config, "objective.reward_weight_mode", REWARD_WEIGHT_MODES)
    _require_choice(config, "objective.loss_aggregation", LOSS_AGGREGATIONS)
    _require_choice(config, "data.preflight", PREFLIGHT_MODES)
    _require_choice(config, "validation.grader", {"verl_rule"})
    _require_choice(config, "validation.model_dtype", {"auto", "bfloat16", "float16"})

    if _get(config, "objective.adv_estimator") != "token_reward_direct":
        raise ConfigError("OPD configs must use objective.adv_estimator='token_reward_direct'")
    if _get(config, "training.mini_batch_size") > _get(config, "training.train_batch_size"):
        raise ConfigError("training.mini_batch_size cannot exceed training.train_batch_size")
    if _get(config, "runtime.gpus_per_node") % _get(config, "runtime.tensor_parallel_size") != 0:
        raise ConfigError("runtime.gpus_per_node must be divisible by runtime.tensor_parallel_size")
    if not 0.0 < float(_get(config, "runtime.gpu_memory_utilization")) <= 1.0:
        raise ConfigError("runtime.gpu_memory_utilization must be in (0, 1]")
    if not 0.0 < float(_get(config, "validation.gpu_memory_utilization")) <= 1.0:
        raise ConfigError("validation.gpu_memory_utilization must be in (0, 1]")
    if _get(config, "optimizer.scheduler") not in {"constant", "cosine"}:
        raise ConfigError("optimizer.scheduler must be 'constant' or 'cosine'")
    experiment_name = _get(config, "experiment.name")
    if experiment_name in {".", ".."} or "/" in experiment_name or "\\" in experiment_name:
        raise ConfigError("experiment.name must be one path component without '/' or '\\'")


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def derived(config: dict[str, Any]) -> dict[str, Any]:
    prompt_length = _get(config, "data.max_prompt_length")
    response_length = _get(config, "data.max_response_length")
    validation_length = _get(config, "validation.max_response_length")
    validation_enabled = _get(config, "validation.before_training") or _get(config, "validation.frequency") > 0
    active_response_length = max(response_length, validation_length) if validation_enabled else response_length
    max_model_len = prompt_length + active_response_length + 1
    actor_token_budget = max(_get(config, "runtime.actor_max_tokens_per_gpu"), max_model_len)
    teacher_token_budget = max(_get(config, "runtime.teacher_max_tokens_per_gpu"), max_model_len)
    output_dir = Path(_get(config, "experiment.output_dir"))
    experiment_name = _get(config, "experiment.name")
    return {
        "max_model_len": max_model_len,
        "actor_token_budget": actor_token_budget,
        "teacher_token_budget": teacher_token_budget,
        "checkpoint_dir": str(output_dir / experiment_name),
        "experiment_name": experiment_name,
        "student_model": _get(config, "models.student"),
        "teacher_model": _get(config, "models.teacher"),
        "train_file": _get(config, "data.train_file"),
        "preflight": _get(config, "data.preflight"),
        "prompt_length": prompt_length,
        "log_dir": _get(config, "experiment.log_dir"),
        "output_dir": str(output_dir),
        "swanlab_mode": _get(config, "tracking.swanlab_mode"),
        "seed": _get(config, "data.seed"),
    }


def managed_hydra_values(config: dict[str, Any]) -> list[tuple[str, Any]]:
    d = derived(config)
    values: list[tuple[str, Any]] = [
        ("algorithm.adv_estimator", _get(config, "objective.adv_estimator")),
        ("algorithm.adv_clip_range", _get(config, "objective.advantage_clip")),
        ("algorithm.grpo_outcome_weight", _get(config, "objective.grpo_outcome_weight")),
        ("algorithm.use_kl_in_reward", _get(config, "objective.use_kl_in_reward")),
        ("algorithm.kl_ctrl.kl_coef", _get(config, "objective.kl_reward_coefficient")),
        ("data.train_files", _get(config, "data.train_file")),
        ("data.val_files", _get(config, "data.validation_files")),
        ("data.shuffle", _get(config, "data.shuffle")),
        ("data.seed", _get(config, "data.seed")),
        ("data.train_batch_size", _get(config, "training.train_batch_size")),
        ("data.max_prompt_length", _get(config, "data.max_prompt_length")),
        ("data.max_response_length", _get(config, "data.max_response_length")),
        ("data.filter_overlong_prompts", _get(config, "data.filter_overlong_prompts")),
        ("data.truncation", _get(config, "data.truncation")),
        ("data.return_raw_chat", True),
        ("+data.apply_chat_template_kwargs.enable_thinking", _get(config, "models.enable_thinking")),
        ("actor_rollout_ref.model.path", _get(config, "models.student")),
        ("actor_rollout_ref.model.use_remove_padding", True),
        ("actor_rollout_ref.model.enable_gradient_checkpointing", _get(config, "models.gradient_checkpointing")),
        ("actor_rollout_ref.model.enable_activation_offload", _get(config, "models.activation_offload")),
        ("actor_rollout_ref.actor.optim.lr", _get(config, "optimizer.learning_rate")),
        ("actor_rollout_ref.actor.optim.weight_decay", _get(config, "optimizer.weight_decay")),
        ("actor_rollout_ref.actor.optim.betas", _get(config, "optimizer.betas")),
        ("actor_rollout_ref.actor.optim.lr_scheduler_type", _get(config, "optimizer.scheduler")),
        ("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", _get(config, "optimizer.warmup_ratio")),
        ("actor_rollout_ref.actor.optim.optimizer", "AdamW"),
        ("actor_rollout_ref.actor.optim.optimizer_impl", "torch.optim"),
        ("actor_rollout_ref.actor.optim.eps", _get(config, "optimizer.epsilon")),
        ("actor_rollout_ref.actor.optim.clip_grad", _get(config, "optimizer.gradient_clip")),
        ("actor_rollout_ref.actor.grad_clip", _get(config, "optimizer.gradient_clip")),
        ("actor_rollout_ref.actor.ppo_epochs", _get(config, "training.ppo_epochs")),
        ("actor_rollout_ref.actor.shuffle", _get(config, "training.shuffle_minibatches")),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", _get(config, "training.mini_batch_size")),
        ("actor_rollout_ref.actor.use_dynamic_bsz", True),
        (
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            _get(config, "training.ppo_micro_batch_size_per_gpu"),
        ),
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", d["actor_token_budget"]),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", _get(config, "runtime.sequence_parallel_size")),
        ("actor_rollout_ref.actor.entropy_coeff", _get(config, "objective.entropy_coefficient")),
        ("actor_rollout_ref.actor.use_kl_loss", _get(config, "objective.use_kl_loss")),
        ("actor_rollout_ref.actor.kl_loss_coef", _get(config, "objective.kl_loss_coefficient")),
        ("actor_rollout_ref.actor.kl_loss_type", _get(config, "objective.kl_loss_type")),
        ("actor_rollout_ref.actor.loss_agg_mode", _get(config, "objective.loss_aggregation")),
        ("actor_rollout_ref.actor.fsdp_config.param_offload", _get(config, "runtime.actor_param_offload")),
        ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", _get(config, "runtime.optimizer_offload")),
        ("actor_rollout_ref.actor.fsdp_config.forward_prefetch", _get(config, "runtime.forward_prefetch")),
        ("actor_rollout_ref.actor.fsdp_config.model_dtype", _get(config, "models.model_dtype")),
        ("actor_rollout_ref.actor.checkpoint.save_contents", ["model", "optimizer", "extra"]),
        ("actor_rollout_ref.actor.checkpoint.load_contents", ["model", "optimizer", "extra"]),
        ("actor_rollout_ref.rollout.name", _get(config, "rollout.engine")),
        ("actor_rollout_ref.rollout.seed", _get(config, "data.seed")),
        ("actor_rollout_ref.rollout.temperature", _get(config, "rollout.temperature")),
        ("actor_rollout_ref.rollout.top_k", _get(config, "rollout.sampling_top_k")),
        ("actor_rollout_ref.rollout.top_p", _get(config, "rollout.top_p")),
        ("actor_rollout_ref.rollout.repetition_penalty", _get(config, "rollout.repetition_penalty")),
        ("actor_rollout_ref.rollout.do_sample", _get(config, "rollout.do_sample")),
        ("actor_rollout_ref.rollout.ignore_eos", _get(config, "rollout.ignore_eos")),
        ("actor_rollout_ref.rollout.n", _get(config, "rollout.responses_per_prompt")),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", _get(config, "runtime.tensor_parallel_size")),
        ("actor_rollout_ref.rollout.gpu_memory_utilization", _get(config, "runtime.gpu_memory_utilization")),
        ("actor_rollout_ref.rollout.max_model_len", d["max_model_len"]),
        ("actor_rollout_ref.rollout.max_num_batched_tokens", d["actor_token_budget"]),
        ("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True),
        ("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", d["actor_token_budget"]),
        (
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
            _get(config, "training.ppo_micro_batch_size_per_gpu"),
        ),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("+actor_rollout_ref.rollout.log_prob_top_k", _get(config, "objective.top_k")),
        ("+actor_rollout_ref.rollout.diagnostic_top_k", _get(config, "objective.diagnostic_top_k")),
        ("+actor_rollout_ref.rollout.top_k_strategy", _get(config, "objective.top_k_strategy")),
        ("+actor_rollout_ref.rollout.reward_weight_mode", _get(config, "objective.reward_weight_mode")),
        ("+actor_rollout_ref.rollout.teacher_temperature", _get(config, "rollout.teacher_temperature")),
        ("actor_rollout_ref.rollout.val_kwargs.do_sample", _get(config, "validation.do_sample")),
        ("+actor_rollout_ref.rollout.val_kwargs.max_tokens", _get(config, "validation.max_response_length")),
        ("actor_rollout_ref.rollout.val_kwargs.n", _get(config, "validation.responses_per_prompt")),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", _get(config, "validation.temperature")),
        ("actor_rollout_ref.rollout.val_kwargs.top_k", _get(config, "validation.top_k")),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", _get(config, "validation.top_p")),
        ("critic.ppo_max_token_len_per_gpu", d["teacher_token_budget"]),
        ("reward_model.enable", True),
        ("+reward_model.reward_kwargs.enable_format_reward", _get(config, "reward.enable_format_reward")),
        ("reward_model.model.path", _get(config, "models.teacher")),
        ("reward_model.model.input_tokenizer", None),
        ("reward_model.model.use_remove_padding", True),
        ("+reward_model.model.dtype", _get(config, "models.teacher_dtype")),
        ("reward_model.model.fsdp_config.param_offload", _get(config, "runtime.teacher_param_offload")),
        ("reward_model.micro_batch_size_per_gpu", _get(config, "training.reward_micro_batch_size_per_gpu")),
        ("custom_reward_function.path", _get(config, "reward.custom_function_path")),
        ("custom_reward_function.name", _get(config, "reward.custom_function_name")),
        ("trainer.logger", _get(config, "tracking.loggers")),
        ("trainer.project_name", _get(config, "experiment.project_name")),
        ("trainer.experiment_name", _get(config, "experiment.name")),
        ("trainer.validation_data_dir", str(Path(_get(config, "experiment.validation_dir")) / d["experiment_name"])),
        ("trainer.n_gpus_per_node", _get(config, "runtime.gpus_per_node")),
        ("trainer.nnodes", _get(config, "runtime.nodes")),
        ("trainer.total_epochs", _get(config, "training.epochs")),
        ("trainer.total_training_steps", _get(config, "training.steps")),
        ("trainer.val_before_train", _get(config, "validation.before_training")),
        ("trainer.test_freq", _get(config, "validation.frequency")),
        ("trainer.save_freq", -1),
        ("trainer.save_steps", _get(config, "checkpoints.save_steps")),
        ("trainer.optimizer_save_steps", _get(config, "checkpoints.optimizer_save_steps")),
        ("trainer.max_actor_ckpt_to_keep", None),
        ("trainer.del_local_ckpt_after_load", False),
        ("trainer.default_local_dir", d["checkpoint_dir"]),
        ("trainer.is_plot", _get(config, "tracking.is_plot")),
        ("trainer.opd_text_diagnostics", _get(config, "tracking.opd_text_diagnostics")),
    ]
    return values


def hydra_overrides(config: dict[str, Any]) -> list[str]:
    managed = [f"{key}={_hydra_value(value)}" for key, value in managed_hydra_values(config)]
    return managed + list(_get(config, "hydra.extra_overrides"))


def expected_hydra_values(config: dict[str, Any]) -> dict[str, Any]:
    return {key.lstrip("+"): value for key, value in managed_hydra_values(config)}


def shell_environment(config: dict[str, Any]) -> dict[str, str]:
    d = derived(config)
    return {
        "OPD_CONFIG_PATH": config["_config_path"],
        "EXPERIMENT_NAME": d["experiment_name"],
        "ACTOR_MODEL_PATH": d["student_model"],
        "REWARD_MODEL_PATH": d["teacher_model"],
        "TRAIN_DATASET": d["train_file"],
        "DATA_PREFLIGHT": d["preflight"],
        "MAX_PROMPT_LENGTH": str(d["prompt_length"]),
        "CKPT_PATH": d["checkpoint_dir"],
        "PROJECT_PATH": d["output_dir"],
        "LOG_DIR": d["log_dir"],
        "CONFIG_SWANLAB_MODE": d["swanlab_mode"],
        "SEED": str(d["seed"]),
    }


def _write_nul(items: list[str]) -> None:
    payload = b"".join(item.encode("utf-8") + b"\0" for item in items)
    sys.stdout.buffer.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "env0", "hydra-args0", "show"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("config", type=Path)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as error:
        raise SystemExit(f"invalid OPD experiment config: {error}") from error

    if args.command == "validate":
        print(f"validated OPD experiment config: {args.config}")
    elif args.command == "env0":
        _write_nul([f"{key}={value}" for key, value in shell_environment(config).items()])
    elif args.command == "hydra-args0":
        _write_nul(hydra_overrides(config))
    elif args.command == "show":
        visible = {key: value for key, value in config.items() if not key.startswith("_")}
        print(json.dumps(visible, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
