# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-training diagnostics for on-policy distillation.

Everything here is computed from tensors the OPD step has already produced, at the
student-visited states of that step. None of it survives into the checkpoint, which is
why it has to be logged live rather than reconstructed afterwards.

Works in the sampled-token ("standard") OPD configuration, i.e.
``rollout.log_prob_top_k=0`` with ``rollout.diagnostic_top_k=K`` supplying the top-k
tensors for logging only. The reward path is untouched by any of this.

Metric groups
-------------
``opd/reverse_kl_*``   the training signal itself (reward = -reverse KL on the sampled token)
``opd/adv_*``          advantage magnitude and clip saturation
``opd/overlap_*``      student/teacher top-k set agreement
``opd/*_mass*``        where the probability mass sits (the paper's 97-99% claim)
``opd/coverage_*``     whether the student still covers the teacher's modes
"""

from typing import Optional

import torch

# Log-probs below this are treated as structural zeros when exponentiated. Guards
# against -inf entering a sum (the "intersection" strategy can emit -inf padding).
_LOGP_FLOOR = -30.0

# (quantile, label) -- labels are explicit so 0.5 does not render as "p5".
_QUANTILES = ((0.5, "p50"), (0.9, "p90"), (0.99, "p99"), (0.999, "p999"))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of ``x`` over positions where ``mask`` is truthy. Shapes must broadcast."""
    m = mask.to(torch.float32)
    x = x.to(torch.float32)
    denom = m.sum().clamp_min(1.0)
    return ((x * m).sum() / denom).item()


def _valid_values(x: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Flatten ``x`` down to just the positions inside the response."""
    return x.to(torch.float32)[response_mask.bool()].reshape(-1)


def _quantile_metrics(values: torch.Tensor, prefix: str) -> dict:
    """Quantiles + max of a 1-D tensor, emitted as ``{prefix}_p50`` etc."""
    out = {}
    if values.numel() == 0:
        return out

    # torch.quantile refuses inputs beyond ~16M elements; subsample deterministically
    # enough for a monitoring metric.
    xs = values
    if xs.numel() > 8_000_000:
        idx = torch.randperm(xs.numel(), device=xs.device)[:8_000_000]
        xs = xs[idx]

    qs = torch.tensor([q for q, _ in _QUANTILES], device=xs.device, dtype=xs.dtype)
    vals = torch.quantile(xs, qs).tolist()
    for (_, label), v in zip(_QUANTILES, vals):
        out[f"{prefix}_{label}"] = v
    out[f"{prefix}_max"] = xs.max().item()
    out[f"{prefix}_mean"] = xs.mean().item()
    return out


def _probs_from_logp(logp: torch.Tensor) -> torch.Tensor:
    return logp.to(torch.float32).clamp_min(_LOGP_FLOOR).exp()


# --------------------------------------------------------------------- repetition
# Single shared implementation. The preflight and offline benchmark import this rather
# than carrying their own copies, so the three call sites cannot drift apart.

REPETITION_TAIL_CHARS = 1024
REPETITION_MIN_CHARS = 256
REPETITION_NGRAM = 4
REPETITION_CHAR_NGRAM = 16
REPETITION_MIN_WORDS = 20
# Calibrated on synthetic prose / enumerations / loops: worst benign case (numbered
# step lists) scores ~0.48, mildest genuine loop ~0.88. 0.65 sits in that gap.
REPETITION_THRESHOLD = 0.65


def _distinct_ratio(units, n):
    if len(units) < n + 4:
        return None
    grams = [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def repetition_score(text: str, tail_chars: int = REPETITION_TAIL_CHARS) -> Optional[float]:
    """Degeneracy score in [0, 1) for the tail of a response. Higher = more repetitive.

    ``1 - distinct_4gram_ratio`` over the last ``tail_chars`` characters. Continuous, so
    it can be tracked as a trend rather than only thresholded, and it catches loops of
    any period.

    Deliberately NOT a compression ratio. zlib was tried first and cannot separate
    degeneration from benign structure: on synthetic cases a numbered step list scored
    0.886 against 0.888 for a genuine long-period loop -- a 0.002 margin. Distinct-n-gram
    gives 0.48 vs 0.88 on the same cases, because enumerations vary at the token that
    carries the index while loops repeat verbatim.

    Falls back to character n-grams when the tail has too few whitespace-separated
    tokens, which is what catches whitespace-free degeneration such as "aaaa...".

    Returns None when the text is too short to score, so callers can exclude those
    rather than silently counting them as clean.
    """
    if not text or len(text) < REPETITION_MIN_CHARS:
        return None
    tail = text[-tail_chars:]
    words = tail.split()
    if len(words) >= REPETITION_MIN_WORDS:
        score = _distinct_ratio(words, REPETITION_NGRAM)
    else:
        score = _distinct_ratio(tail, REPETITION_CHAR_NGRAM)
    return None if score is None else max(0.0, score)


def is_repetitive(text: str, threshold: float = REPETITION_THRESHOLD) -> bool:
    s = repetition_score(text)
    return s is not None and s >= threshold


def compute_opd_diagnostics(batch, adv_clip_range: float = 0.0) -> dict:
    """Return a flat ``{metric_name: float}`` dict for one OPD step.

    Args:
        batch: the ``DataProto`` after ``compute_advantage``.
        adv_clip_range: the symmetric clip that was applied, used to report saturation.
                        Pass 0 if clipping is disabled.

    Every metric is optional: whatever tensors are absent are simply skipped, so this
    degrades cleanly if ``diagnostic_top_k`` is off.
    """
    tb = batch.batch
    metrics: dict = {}

    if "response_mask" not in tb.keys():
        return metrics
    response_mask = tb["response_mask"]
    valid = response_mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return metrics
    metrics["opd/valid_tokens"] = float(n_valid)

    # ------------------------------------------------- reverse KL / raw advantage
    # rm_scores is the per-token reward BEFORE the advantage clip, and in sampled-token
    # OPD it is exactly -(student_logp - teacher_logp). Since advantage == rm_scores,
    # rm_scores is also the only place the *pre-clip* advantage magnitude can be read:
    # batch["advantages"] has already been clamped by compute_advantage, so its tails
    # saturate at adv_clip_range by construction and say nothing about the raw signal.
    if "rm_scores" in tb.keys() and tb["rm_scores"] is not None:
        rm_scores = tb["rm_scores"]
        if rm_scores.dim() == 3:
            rm_scores = rm_scores.mean(dim=-1)
        if rm_scores.dim() == 2:
            rev_kl = _valid_values(-rm_scores, response_mask)
            metrics.update(_quantile_metrics(rev_kl, "opd/reverse_kl_sampled"))
            # Negative reverse KL means the teacher likes the sampled token *more*
            # than the student does; the fraction of such tokens is the share of
            # positive learning signal in the step.
            metrics["opd/reverse_kl_negative_frac"] = (rev_kl < 0).to(torch.float32).mean().item()

            # Pre-clip advantage magnitude. This is the series to watch at steps 1-5:
            # if the upper tail runs well past adv_clip_range, the clip is load-bearing
            # and the unclipped run would have been dominated by a handful of tokens.
            adv_raw = _valid_values(rm_scores.abs(), response_mask)
            metrics.update(_quantile_metrics(adv_raw, "opd/adv_preclip_abs"))
            if adv_clip_range and adv_clip_range > 0:
                metrics["opd/adv_clip_saturation"] = (
                    (adv_raw >= adv_clip_range).to(torch.float32).mean().item()
                )

    # Post-clip mean only, as a sanity check that the clip actually applied. The
    # post-clip quantiles are deliberately not reported: they are pinned to the clip.
    if "advantages" in tb.keys():
        adv = tb["advantages"]
        if adv.dim() == 3:
            adv = adv.mean(dim=-1)
        metrics["opd/adv_postclip_abs_mean"] = _masked_mean(adv.abs(), response_mask)

    # ------------------------------------------------- top-k alignment (optional)
    has_student_topk = "student_top_k_ids" in tb.keys() and "student_top_k_log_probs" in tb.keys()
    has_teacher_topk = "teacher_top_k_ids" in tb.keys() and "teacher_top_k_log_probs" in tb.keys()

    if has_student_topk:
        s_logp = tb["student_top_k_log_probs"]  # (B, T, K)
        s_p = _probs_from_logp(s_logp)
        mask3 = response_mask.unsqueeze(-1)

        # The paper's concentration claim: how much of the student's distribution lives
        # in its own top-K. Expect 0.97-0.99 at K=16 for a converged non-thinking model.
        metrics["opd/student_topk_mass"] = _masked_mean(s_p.sum(dim=-1), response_mask)

        if "overlap_mask" in tb.keys():
            overlap = tb["overlap_mask"].to(torch.float32)  # (B, T, K), student-token-in-teacher-topk
            # Fraction of the student's top-K tokens the teacher also ranks top-K.
            metrics["opd/overlap_ratio"] = _masked_mean(overlap.mean(dim=-1), response_mask)
            # Same thing weighted by probability: mass the student puts on shared tokens.
            metrics["opd/student_mass_on_shared"] = _masked_mean((s_p * overlap).sum(dim=-1), response_mask)

        if "teacher_on_student_log_probs" in tb.keys():
            t_on_s = tb["teacher_on_student_log_probs"]  # (B, T, K)
            t_on_s_p = _probs_from_logp(t_on_s)
            # Teacher mass sitting on the student's top-K support.
            metrics["opd/teacher_mass_on_student_topk"] = _masked_mean(t_on_s_p.sum(dim=-1), response_mask)

            # Partial sum of the reverse KL over the student's own top-K:
            #   sum_{j in topK} p_S(j) [log p_S(j) - log p_T(j)]
            # NOT a divergence. It is truncated (the tail beyond K is dropped) and
            # unnormalised (the top-K probabilities do not sum to 1), so it can be
            # negative even though the true reverse KL cannot. Read it as a trend line
            # for how the student's high-probability support is aligning, not as a KL
            # value. It covers the whole top-K support rather than the single sampled
            # token, and costs nothing since both tensors are already on hand.
            s_logp_c = s_logp.to(torch.float32).clamp_min(_LOGP_FLOOR)
            t_logp_c = t_on_s.to(torch.float32).clamp_min(_LOGP_FLOOR)
            kl_topk = (s_p * (s_logp_c - t_logp_c)).sum(dim=-1)
            metrics["opd/topk_kl_contribution"] = _masked_mean(kl_topk, response_mask)

        if has_teacher_topk:
            # Top-1 agreement: cheapest possible alignment proxy, and the one that
            # correlates most directly with greedy-decode behaviour changing.
            s_top1 = tb["student_top_k_ids"][..., 0]
            t_top1 = tb["teacher_top_k_ids"][..., 0]
            metrics["opd/top1_agreement"] = _masked_mean((s_top1 == t_top1).to(torch.float32), response_mask)

    # NOTE: the student/teacher entropy gap is computed in ray_trainer's entropy block,
    # not here -- "entropys" is popped from the batch before this function runs.

    if has_teacher_topk:
        t_logp = tb["teacher_top_k_log_probs"]
        t_p = _probs_from_logp(t_logp)
        metrics["opd/teacher_topk_mass"] = _masked_mean(t_p.sum(dim=-1), response_mask)

        if "teacher_in_student_mask" in tb.keys():
            t_in_s = tb["teacher_in_student_mask"].to(torch.float32)
            # Mode coverage: the share of the teacher's top-K mass that the student's
            # top-K still contains. A support-overlap statistic, NOT a substitute for
            # forward KL: it detects the student dropping tokens the teacher ranks
            # highly, but is blind to probability mismatch on tokens both models cover.
            # Full-vocab forward KL is not obtainable here -- it would need the student
            # and teacher distributions co-resident, and they live in separate workers.
            metrics["opd/coverage_teacher_mass_in_student_topk"] = _masked_mean(
                (t_p * t_in_s).sum(dim=-1), response_mask
            )
            metrics["opd/coverage_teacher_topk_in_student"] = _masked_mean(t_in_s.mean(dim=-1), response_mask)

    return metrics


def compute_text_diagnostics(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    think_token: str = "<think>",
    sample_limit: int = 64,
) -> dict:
    """Cheap failure-mode canaries decoded from the rollout text.

    Kept separate from :func:`compute_opd_diagnostics` because it needs the tokenizer and
    only looks at a subsample -- decoding every response every step is not worth it.
    """
    metrics: dict = {}
    bsz = responses.size(0)
    if bsz == 0:
        return metrics

    n = min(sample_limit, bsz)
    lengths = response_mask.sum(dim=-1)

    think_hits = 0
    boxed_hits = 0
    rep_scores = []
    scored = 0

    for i in range(n):
        ln = int(lengths[i].item())
        if ln <= 0:
            continue
        text = tokenizer.decode(responses[i, :ln], skip_special_tokens=False)
        if think_token in text:
            think_hits += 1
        if "\\boxed{" in text:
            boxed_hits += 1
        s = repetition_score(text)
        if s is not None:
            rep_scores.append(s)
            scored += 1

    metrics["opd/think_token_rate"] = think_hits / n
    metrics["opd/boxed_answer_rate"] = boxed_hits / n
    if rep_scores:
        # Continuous mean plus the thresholded rate: the mean moves before any response
        # crosses the threshold, which is the earlier warning of the two.
        metrics["opd/repetition_score_mean"] = sum(rep_scores) / len(rep_scores)
        metrics["opd/repetition_score_max"] = max(rep_scores)
        metrics["opd/repetition_rate"] = sum(s >= REPETITION_THRESHOLD for s in rep_scores) / len(rep_scores)
        metrics["opd/repetition_scored_frac"] = scored / n
    return metrics
