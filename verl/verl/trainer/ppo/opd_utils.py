"""Small, dependency-light invariants shared by standard OPD training paths."""

from __future__ import annotations

import torch


def compute_sampled_token_opd_reward(
    student_log_prob: torch.Tensor, teacher_log_prob: torch.Tensor
) -> torch.Tensor:
    """Return the standard OPD reward on tokens sampled by the student.

    Temperature handling belongs to the respective model forward passes. For
    the Lightning recipe this receives student log-probabilities at T=0.8 and
    teacher log-probabilities at T=1.0.
    """
    if student_log_prob.shape != teacher_log_prob.shape:
        raise ValueError(
            "student and teacher sampled-token log-probabilities must have the same shape: "
            f"{student_log_prob.shape} != {teacher_log_prob.shape}"
        )
    return teacher_log_prob - student_log_prob


def clip_and_mask_token_rewards(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, clip_range: float
) -> torch.Tensor:
    if clip_range > 0:
        token_level_rewards = torch.clamp(token_level_rewards, min=-clip_range, max=clip_range)
    return token_level_rewards * response_mask


def sequence_mean_token_mean(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    token_counts = torch.sum(loss_mask, dim=-1)
    sequence_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (token_counts + 1e-8)
    sequence_mask = (token_counts > 0).float()
    return torch.sum(sequence_losses * sequence_mask) / sequence_mask.sum().clamp_min(1.0)


def resolve_opd_top_k(log_prob_top_k: int, diagnostic_top_k: int) -> tuple[int, int]:
    """Return ``(reward_top_k, forward_top_k)`` for an OPD forward pass.

    Diagnostics may request top-k tensors during standard sampled-token OPD,
    but they must never switch the reward from sampled-token to top-k OPD.
    """
    if log_prob_top_k < 0 or diagnostic_top_k < 0:
        raise ValueError("OPD top-k values must be non-negative")
    forward_top_k = log_prob_top_k if log_prob_top_k > 0 else diagnostic_top_k
    return log_prob_top_k, forward_top_k
