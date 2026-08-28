from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))
sys.path.insert(0, str(ROOT))

from scripts.data.prepare_lightning_dapo import BOXED_INSTRUCTION, normalize_prompt  # noqa: E402
from scripts.diag.validate_lightning_opd_config import validate as validate_resolved_config  # noqa: E402
OPD_UTILS_PATH = ROOT / "verl" / "verl" / "trainer" / "ppo" / "opd_utils.py"
OPD_UTILS_SPEC = importlib.util.spec_from_file_location("lightning_opd_utils", OPD_UTILS_PATH)
assert OPD_UTILS_SPEC and OPD_UTILS_SPEC.loader
OPD_UTILS = importlib.util.module_from_spec(OPD_UTILS_SPEC)
OPD_UTILS_SPEC.loader.exec_module(OPD_UTILS)

clip_and_mask_token_rewards = OPD_UTILS.clip_and_mask_token_rewards
compute_sampled_token_opd_reward = OPD_UTILS.compute_sampled_token_opd_reward
resolve_opd_top_k = OPD_UTILS.resolve_opd_top_k
sequence_mean_token_mean = OPD_UTILS.sequence_mean_token_mean


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


def test_recipe_has_one_256_sequence_optimizer_minibatch() -> None:
    script = (ROOT / "opd_lightning_a100.sh").read_text(encoding="utf-8")

    def literal_export(name: str) -> int:
        match = re.search(rf"^export {name}=(\d+)", script, flags=re.MULTILINE)
        assert match, f"missing literal export for {name}"
        return int(match.group(1))

    prompts = literal_export("TRAIN_BATCH_SIZE")
    mini_batch_prompts = literal_export("MINI_BATCH_SIZE")
    responses = literal_export("N_RESPONSES")
    world_size = literal_export("N_GPUS")
    sequence_parallel_size = literal_export("PARALLEL_SIZE")

    rollout_sequences = prompts * responses
    normalized_mini_batch = mini_batch_prompts * responses // (world_size // sequence_parallel_size)
    assert rollout_sequences == 256
    assert normalized_mini_batch == rollout_sequences


def test_recipe_pins_paper_faithful_settings_and_thinking_default() -> None:
    script = (ROOT / "opd_lightning_a100.sh").read_text(encoding="utf-8")
    required = (
        "export TEMPERATURE=0.8",
        "export TEACHER_TEMPERATURE=1.0",
        "export MAX_PROMPT_LENGTH=1024",
        "export LOSS_AGG_MODE=seq-mean-token-mean",
        "export LOG_PROB_TOP_K=0",
        "export ADV_CLIP_RANGE=10.0",
        "export TOTAL_STEPS=${TOTAL_STEPS:-150}",
        "reward_model.model.fsdp_config.param_offload=$TEACHER_PARAM_OFFLOAD",
    )
    for setting in required:
        assert setting in script
    assert "+data.apply_chat_template_kwargs.enable_thinking=False" in script


def test_official_prompt_normalization_is_lossless() -> None:
    content = f"Solve this exactly. Final line: {BOXED_INSTRUCTION}"
    source = [{"content": content, "role": "user"}]
    assert normalize_prompt(source) == [{"role": "user", "content": content}]


def test_resolved_config_validator_rejects_wrong_temperature() -> None:
    config: dict = {}

    def assign(dotted_key: str, value: object) -> None:
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    values = {
        "algorithm.adv_estimator": "token_reward_direct",
        "algorithm.adv_clip_range": 10.0,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": "datasets/dapo-math-17k.parquet",
        "data.shuffle": True,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 4096,
        "data.train_batch_size": 64,
        "data.return_raw_chat": True,
        "actor_rollout_ref.actor.optim.lr": 2e-6,
        "actor_rollout_ref.actor.optim.weight_decay": 0.1,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.98],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
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
        "trainer.total_training_steps": 150,
    }
    for key, value in values.items():
        assign(key, value)
    assign("data.apply_chat_template_kwargs.enable_thinking", False)

    validate_resolved_config(config, smoke=False)
    config["actor_rollout_ref"]["rollout"]["temperature"] = 1.0
    with pytest.raises(SystemExit, match="temperature"):
        validate_resolved_config(config, smoke=False)
