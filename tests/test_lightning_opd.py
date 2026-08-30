from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))
sys.path.insert(0, str(ROOT))

from scripts.config.load_opd_config import (  # noqa: E402
    ConfigError,
    expected_hydra_values,
    hydra_overrides,
    load_config,
    validate_config,
)
from scripts.data.prepare_lightning_dapo import BOXED_INSTRUCTION, normalize_prompt  # noqa: E402
from scripts.diag.validate_opd_config import validate as validate_resolved_config  # noqa: E402
from scripts.eval.run_opd_eval import (  # noqa: E402
    ensure_prompt_suffix,
    grade_records,
    load_samples,
    output_filename,
    settings_from_config,
    validate_generation_count,
)

OPD_UTILS_PATH = ROOT / "verl" / "verl" / "trainer" / "ppo" / "opd_utils.py"
OPD_UTILS_SPEC = importlib.util.spec_from_file_location("lightning_opd_utils", OPD_UTILS_PATH)
assert OPD_UTILS_SPEC and OPD_UTILS_SPEC.loader
OPD_UTILS = importlib.util.module_from_spec(OPD_UTILS_SPEC)
OPD_UTILS_SPEC.loader.exec_module(OPD_UTILS)

clip_and_mask_token_rewards = OPD_UTILS.clip_and_mask_token_rewards
compute_sampled_token_opd_reward = OPD_UTILS.compute_sampled_token_opd_reward
resolve_opd_top_k = OPD_UTILS.resolve_opd_top_k
sequence_mean_token_mean = OPD_UTILS.sequence_mean_token_mean

CONFIG_DIR = ROOT / "configs" / "opd"
LIGHTNING_CONFIG = CONFIG_DIR / "lightning_standard_a100.toml"
SMOKE_CONFIG = CONFIG_DIR / "lightning_standard_a100_smoke.toml"
PAPER_CONFIG = CONFIG_DIR / "paper_qwen3_1p7b_rl_math_teacher_a100.toml"


def test_sampled_token_reward_uses_student_t08_and_teacher_t1() -> None:
    student_logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 3.0]])
    teacher_logits = torch.tensor([[1.0, 2.0, -0.5], [2.0, 0.0, 1.0]])
    sampled_ids = torch.tensor([[0], [2]])

    student_logp_t08 = torch.log_softmax(student_logits / 0.8, dim=-1).gather(-1, sampled_ids).squeeze(-1)
    teacher_logp_t1 = torch.log_softmax(teacher_logits / 1.0, dim=-1).gather(-1, sampled_ids).squeeze(-1)

    reward = compute_sampled_token_opd_reward(student_logp_t08, teacher_logp_t1)
    torch.testing.assert_close(reward, teacher_logp_t1 - student_logp_t08)


def test_direct_advantage_masks_and_clips_to_ten() -> None:
    rewards = torch.tensor([[-20.0, -3.0, 15.0], [8.0, 12.0, -11.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    advantages = clip_and_mask_token_rewards(rewards, mask, clip_range=10.0)

    expected = torch.tensor([[-10.0, -3.0, 0.0], [8.0, 10.0, -10.0]])
    torch.testing.assert_close(advantages, expected)


def test_zero_advantage_clip_preserves_author_reward_values() -> None:
    rewards = torch.tensor([[-20.0, -3.0, 15.0], [8.0, 12.0, -11.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    advantages = clip_and_mask_token_rewards(rewards, mask, clip_range=0.0)

    expected = torch.tensor([[-20.0, -3.0, 0.0], [8.0, 12.0, -11.0]])
    torch.testing.assert_close(advantages, expected)


def test_seq_mean_token_mean_weights_responses_equally() -> None:
    losses = torch.tensor([[1.0, 3.0, 0.0], [9.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    sequence_balanced = sequence_mean_token_mean(losses, mask)
    token_balanced = torch.sum(losses * mask) / torch.sum(mask)

    torch.testing.assert_close(sequence_balanced, torch.tensor(5.5))
    torch.testing.assert_close(token_balanced, torch.tensor(13.0 / 3.0))


def test_diagnostic_top_k_does_not_select_top_k_reward() -> None:
    reward_top_k, forward_top_k = resolve_opd_top_k(log_prob_top_k=0, diagnostic_top_k=16)
    assert reward_top_k == 0
    assert forward_top_k == 16


@pytest.mark.parametrize("path", [LIGHTNING_CONFIG, SMOKE_CONFIG, PAPER_CONFIG])
def test_checked_in_experiment_configs_are_valid(path: Path) -> None:
    load_config(path)


def test_lightning_config_reproduces_standard_recipe() -> None:
    config = load_config(LIGHTNING_CONFIG)
    expected = expected_hydra_values(config)

    assert expected["actor_rollout_ref.model.path"] == "model/Qwen3-1.7B-SFT"
    assert expected["reward_model.model.path"] == "model/Qwen3-4B"
    assert expected["data.max_prompt_length"] == 1024
    assert expected["data.max_response_length"] == 4096
    assert expected["actor_rollout_ref.rollout.temperature"] == 0.8
    assert expected["actor_rollout_ref.rollout.teacher_temperature"] == 1.0
    assert expected["actor_rollout_ref.rollout.log_prob_top_k"] == 0
    assert expected["actor_rollout_ref.rollout.diagnostic_top_k"] == 16
    assert expected["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert expected["trainer.total_training_steps"] == 150
    assert expected["trainer.optimizer_save_steps"] == [50]
    assert expected["data.apply_chat_template_kwargs.enable_thinking"] is False
    assert expected["algorithm.adv_clip_range"] == 10.0


def test_lightning_config_has_one_256_sequence_optimizer_minibatch() -> None:
    config = load_config(LIGHTNING_CONFIG)
    prompts = config["training"]["train_batch_size"]
    mini_batch_prompts = config["training"]["mini_batch_size"]
    responses = config["rollout"]["responses_per_prompt"]
    world_size = config["runtime"]["gpus_per_node"] * config["runtime"]["nodes"]
    sequence_parallel_size = config["runtime"]["sequence_parallel_size"]

    rollout_sequences = prompts * responses
    normalized_mini_batch = mini_batch_prompts * responses // (world_size // sequence_parallel_size)
    assert rollout_sequences == 256
    assert normalized_mini_batch == rollout_sequences


def test_paper_config_reproduces_qwen_rl_teacher_experiment() -> None:
    config = load_config(PAPER_CONFIG)
    expected = expected_hydra_values(config)

    assert expected["actor_rollout_ref.model.path"] == "model/Qwen3-1.7B"
    assert expected["reward_model.model.path"] == "model/Qwen3-4B-Non-Thinking-RL-Math"
    assert expected["data.train_files"] == "datasets/dapo-math-17k-processed.parquet"
    assert expected["data.apply_chat_template_kwargs.enable_thinking"] is False
    assert expected["data.max_prompt_length"] == 1024
    assert expected["data.max_response_length"] == 7168
    assert expected["data.train_batch_size"] == 64
    assert expected["actor_rollout_ref.actor.ppo_mini_batch_size"] == 64
    assert expected["actor_rollout_ref.rollout.n"] == 4
    assert expected["actor_rollout_ref.rollout.temperature"] == 1.0
    assert expected["actor_rollout_ref.rollout.top_p"] == 1.0
    assert expected["actor_rollout_ref.rollout.log_prob_top_k"] == 16
    assert expected["actor_rollout_ref.rollout.top_k_strategy"] == "only_stu"
    assert expected["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert expected["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    assert expected["actor_rollout_ref.actor.use_kl_loss"] is False
    assert expected["algorithm.kl_ctrl.kl_coef"] == 0.0
    assert expected["algorithm.adv_clip_range"] == 0.0
    assert expected["trainer.total_training_steps"] == 279
    assert expected["trainer.optimizer_save_steps"] == [50]


def test_paper_evaluation_config_matches_reported_avg16_protocol() -> None:
    config = load_config(PAPER_CONFIG)
    settings = settings_from_config(config)

    assert config["data"]["validation_files"] == [
        "datasets/test_data/AIME24/test.parquet",
        "datasets/test_data/AIME25/test.parquet",
        "datasets/test_data/AMC23/test.parquet",
    ]
    assert settings.responses_per_prompt == 16
    assert settings.temperature == 0.7
    assert settings.top_p == 0.95
    assert settings.top_k == -1
    assert settings.max_tokens == 31_744
    assert settings.enable_thinking is False
    assert config["validation"]["grader"] == "verl_rule"


def test_evaluator_does_not_duplicate_existing_paper_prompt_suffix() -> None:
    config = load_config(PAPER_CONFIG)
    suffix = config["validation"]["prompt_suffix"]
    samples = load_samples(ROOT / config["data"]["validation_files"][0], suffix)

    assert samples
    assert samples[0].prompt.endswith(suffix.strip())
    assert samples[0].prompt.count(suffix.strip()) == 1
    assert ensure_prompt_suffix("Solve this problem.", suffix).endswith(suffix.strip())


def test_evaluation_filename_and_avg_at_n_are_reproducible() -> None:
    settings = settings_from_config(load_config(PAPER_CONFIG))
    assert output_filename("AIME24", settings) == "aime24_t0.7_p0.95_n16-MNT31744.jsonl"

    records = [
        {"example_id": 0, "rollout_id": 0, "response": r"\boxed{1}", "answer": "1"},
        {"example_id": 0, "rollout_id": 1, "response": r"\boxed{2}", "answer": "1"},
        {"example_id": 1, "rollout_id": 0, "response": r"\boxed{3}", "answer": "3"},
        {"example_id": 1, "rollout_id": 1, "response": "no box", "answer": "3"},
    ]
    validate_generation_count(records, sample_count=2, responses_per_prompt=2)
    _, summary = grade_records(
        records,
        responses_per_prompt=2,
        grader=lambda response, answer: response == rf"\boxed{{{answer}}}",
    )
    assert summary["avg_at_n"] == 0.5
    assert summary["pass_at_n"] == 1.0
    assert summary["format_errors"] == 1


def test_config_seed_controls_dataset_and_rollout() -> None:
    config = load_config(LIGHTNING_CONFIG)
    expected = expected_hydra_values(config)
    assert expected["data.seed"] == 42
    assert expected["actor_rollout_ref.rollout.seed"] == 42


def test_seed_is_part_of_rollout_config_not_validation_sampling_config() -> None:
    source = (ROOT / "verl" / "verl" / "workers" / "config" / "rollout.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    annotations = {
        node.name: {
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "seed" in annotations["RolloutConfig"]
    assert "seed" not in annotations["SamplingConfig"]


def test_method_and_top_k_must_agree() -> None:
    config = load_config(LIGHTNING_CONFIG)
    config.pop("_config_path")
    invalid = copy.deepcopy(config)
    invalid["objective"]["method"] = "top_k"
    with pytest.raises(ConfigError, match="top_k>0"):
        validate_config(invalid)


def test_unknown_config_key_is_rejected_instead_of_ignored() -> None:
    config = load_config(LIGHTNING_CONFIG)
    config.pop("_config_path")
    config["rollout"]["temprature"] = 0.8
    with pytest.raises(ConfigError, match="temprature"):
        validate_config(config)


def test_optimizer_checkpoint_must_also_be_a_save_step() -> None:
    config = load_config(LIGHTNING_CONFIG)
    config.pop("_config_path")
    invalid = copy.deepcopy(config)
    invalid["checkpoints"]["optimizer_save_steps"] = [49]
    with pytest.raises(ConfigError, match="also appear"):
        validate_config(invalid)


def test_hydra_arguments_are_generated_from_config() -> None:
    config = load_config(PAPER_CONFIG)
    args = hydra_overrides(config)
    assert "+actor_rollout_ref.rollout.log_prob_top_k=16" in args
    assert "actor_rollout_ref.rollout.seed=42" in args
    assert "data.train_files=datasets/dapo-math-17k-processed.parquet" in args
    assert "trainer.total_training_steps=279" in args
    assert not any("Qwen3-4B-SFT" in arg for arg in args)


def test_resolved_config_validator_rejects_drift() -> None:
    experiment = load_config(LIGHTNING_CONFIG)
    resolved: dict = {}

    def assign(dotted_key: str, value: object) -> None:
        cursor = resolved
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    for key, value in expected_hydra_values(experiment).items():
        assign(key, value)

    validate_resolved_config(resolved, experiment)
    resolved["actor_rollout_ref"]["rollout"]["temperature"] = 1.0
    with pytest.raises(SystemExit, match="temperature"):
        validate_resolved_config(resolved, experiment)


def test_official_prompt_normalization_is_lossless() -> None:
    content = f"Solve this exactly. Final line: {BOXED_INSTRUCTION}"
    source = [{"content": content, "role": "user"}]
    assert normalize_prompt(source) == [{"role": "user", "content": content}]
