#!/usr/bin/env python3
"""Tier-0 go/no-go check, to be run BEFORE starting OPD.

The Rethinking-OPD paper's second success condition is that the teacher must offer
capability the student does not already have. Same-family teachers frequently fail
this -- the paper shows 1.5B and 7B same-family teachers can be distributionally
indistinguishable from the student's point of view. When that happens OPD has
nothing to transfer and the run comes out flat.

This script measures the gap directly: pass@1 and pass@k for teacher and student on
the same training prompts, plus the format/length statistics that reveal whether the
two are even speaking the same dialect (condition (i), thinking-pattern
compatibility). Roughly 30 minutes on one GPU. It is much cheaper than discovering
the same thing from a flat 6-hour training curve.

Usage
-----
    python scripts/diag/pretrain_gap_check.py \
        --student model/Qwen3-1.7B-SFT \
        --teacher model/Qwen3-4B \
        --parquet datasets/dapo-math-17k.parquet \
        --n-prompts 500 --k 8

Reading the result
------------------
  teacher pass@1 - student pass@1   headroom OPD can plausibly capture
  teacher pass@k - student pass@k   capability the teacher genuinely has and the
                                    student does not. If this is near zero, stop:
                                    the teacher has nothing new to teach at these
                                    prompts and OPD will mostly re-weight what the
                                    student already does.
"""

import argparse
import gc
import json
import os
import random
import sys

import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl"))

# FlashInfer JIT-compiles its sampling kernels and needs nvcc (the CUDA toolkit, not
# just the driver). Fall back to vLLM's native top-k/top-p sampler when nvcc is absent.
# Set VLLM_USE_FLASHINFER_SAMPLER=1 explicitly to override.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

THINK_OPEN = "<think>"


def load_prompts(parquet_path, n_prompts, seed):
    table = pq.read_table(parquet_path).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(table)
    rows = table[:n_prompts]
    return [
        {
            "messages": r["prompt"],
            "ground_truth": str(r["reward_model"]["ground_truth"]),
            "data_source": r["data_source"],
        }
        for r in rows
    ]


def is_repetitive(text):
    # Shared implementation, so this and the offline benchmark cannot drift apart.
    from verl.trainer.ppo.opd_diagnostics import is_repetitive as _is_rep

    return _is_rep(text)


def grade(solution_str, ground_truth):
    from verl.utils.reward_score.ttrl_math import reward_func

    try:
        res = reward_func("math_dapo", solution_str, ground_truth)
        if isinstance(res, dict):
            res = res.get("score", 0.0)
        return float(res) > 0.5
    except Exception:
        return False


def run_model(model_path, prompts, k, max_tokens, temperature, top_p, gpu_mem_util, enable_thinking):
    """Generate k samples per prompt with vLLM, then grade. Frees the engine on exit."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    rendered = []
    for p in prompts:
        try:
            text = tokenizer.apply_chat_template(
                p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            # Template does not accept enable_thinking. Worth knowing about -- it means
            # the training flag +data.apply_chat_template_kwargs.enable_thinking=False
            # is being silently ignored too.
            print(f"  WARNING: {model_path} chat template does not accept enable_thinking")
            text = tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        rendered.append(text)

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_tokens + 1024,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sampling = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = llm.generate(rendered, sampling)

    per_prompt = []
    lengths, think_hits, boxed_hits = [], 0, 0
    trunc_hits, repeat_hits = 0, 0
    total_samples = 0
    for prompt, out in zip(prompts, outputs):
        correct = []
        for cand in out.outputs:
            text = cand.text
            correct.append(grade(text, prompt["ground_truth"]))
            lengths.append(len(cand.token_ids))
            total_samples += 1
            if THINK_OPEN in text:
                think_hits += 1
            if "\\boxed{" in text or "Answer:" in text:
                boxed_hits += 1
            # Truncation and repetition are the two signatures of the length-inflation
            # collapse mode reported for OPD (StableOPD, arXiv 2604.08527). Worth a
            # baseline reading here so the end-of-run numbers have something to move
            # against.
            if getattr(cand, "finish_reason", None) == "length":
                trunc_hits += 1
            if is_repetitive(text):
                repeat_hits += 1
        per_prompt.append(correct)

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    n = len(per_prompt)
    pass1 = sum(c[0] for c in per_prompt) / n
    pass1_avg = sum(sum(c) / len(c) for c in per_prompt) / n
    passk = sum(any(c) for c in per_prompt) / n
    lengths.sort()

    def _pct(p):
        return lengths[int(p * (len(lengths) - 1))] if lengths else 0

    return {
        "pass@1_first": pass1,
        "pass@1_avg": pass1_avg,
        f"pass@{k}": passk,
        "mean_len": sum(lengths) / max(len(lengths), 1),
        "p50_len": _pct(0.50),
        "p95_len": _pct(0.95),
        "p99_len": _pct(0.99),
        "max_len": lengths[-1] if lengths else 0,
        "truncation_rate": trunc_hits / max(total_samples, 1),
        "repetition_rate": repeat_hits / max(total_samples, 1),
        "think_token_rate": think_hits / max(total_samples, 1),
        "answer_format_rate": boxed_hits / max(total_samples, 1),
        "per_prompt": per_prompt,
    }


def bootstrap_gap_ci(student_pp, teacher_pp, k, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI on the PAIRED pass@k difference (teacher - student).

    Resamples prompts, not samples: the prompt set is the unit of uncertainty, and both
    models were evaluated on the same prompts, so the difference must be resampled
    jointly to keep the pairing.
    """
    rng = random.Random(seed)
    n = len(student_pp)
    idx_all = range(n)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in idx_all]
        s = sum(any(student_pp[i]) for i in idx) / n
        t = sum(any(teacher_pp[i]) for i in idx) / n
        diffs.append(t - s)
    diffs.sort()
    lo = diffs[int((alpha / 2) * (n_boot - 1))]
    hi = diffs[int((1 - alpha / 2) * (n_boot - 1))]
    point = sum(any(t) for t in teacher_pp) / n - sum(any(s) for s in student_pp) / n
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--parquet", default="datasets/dapo-math-17k.parquet")
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0, help="match the training temperature")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--enable-thinking", action="store_true", help="off by default, matching the OPD recipe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="diag_out/tier0_gap.json")
    args = ap.parse_args()

    prompts = load_prompts(args.parquet, args.n_prompts, args.seed)
    print(f"{len(prompts)} prompts | k={args.k} | thinking={'on' if args.enable_thinking else 'off'}\n")

    print("=== student ===")
    student = run_model(
        args.student, prompts, args.k, args.max_tokens, args.temperature,
        args.top_p, args.gpu_mem_util, args.enable_thinking,
    )
    print("\n=== teacher ===")
    teacher = run_model(
        args.teacher, prompts, args.k, args.max_tokens, args.temperature,
        args.top_p, args.gpu_mem_util, args.enable_thinking,
    )

    kk = f"pass@{args.k}"
    # Prompts the teacher can solve and the student cannot -- the concrete surface
    # OPD has to work with.
    teachable = sum(
        1 for s, t in zip(student["per_prompt"], teacher["per_prompt"]) if any(t) and not any(s)
    ) / len(prompts)

    print("\n" + "=" * 62)
    print(f"{'metric':<28}{'student':>11}{'teacher':>11}{'gap':>11}")
    print("-" * 62)
    for key in [
        "pass@1_avg", kk, "mean_len", "p50_len", "p95_len", "p99_len",
        "truncation_rate", "repetition_rate", "think_token_rate", "answer_format_rate",
    ]:
        s, t = student[key], teacher[key]
        print(f"{key:<28}{s:>11.4f}{t:>11.4f}{t - s:>11.4f}")
    print("-" * 62)
    print(f"{'teachable prompt fraction':<28}{'':>11}{'':>11}{teachable:>11.4f}")
    print("=" * 62)

    point, lo, hi = bootstrap_gap_ci(
        student["per_prompt"], teacher["per_prompt"], args.k, seed=args.seed
    )
    print(f"\npass@{args.k} gap (teacher - student): {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"({len(prompts)} prompts, paired percentile bootstrap over prompts)")

    print("\nInterpretation -- evidence, not a verdict:")
    if hi < 0:
        print("  The CI lies entirely below zero: on this prompt set the student already")
        print("  matches or beats the teacher at pass@k. Condition (ii) in the paper looks")
        print("  unmet here, and a flat OPD curve would not be surprising.")
    elif lo <= 0 <= hi:
        print("  The CI straddles zero, so this sample cannot distinguish the two models'")
        print("  pass@k. That is not evidence the run will fail -- it is an absence of")
        print("  evidence either way. Raise --n-prompts for a tighter interval if you want")
        print("  to resolve it, or proceed and let the first ~30 training steps decide.")
    else:
        print("  The CI is entirely above zero: the teacher solves strictly more at pass@k,")
        print("  which is the headroom OPD has to work with.")
    print(f"  {teachable:.1%} of prompts are solved by the teacher and not the student.")
    print("\n  Note this measures capability headroom only. OPD transfers a distribution,")
    print("  not a score, so a small gap does not prove the run is pointless and a large")
    print("  one does not guarantee it succeeds. Weigh it, do not let it decide alone.")

    if abs(teacher["think_token_rate"] - student["think_token_rate"]) > 0.05:
        print("\nWARNING: think-token rates differ materially between the two models.")
        print("That is condition (i), thinking-pattern compatibility. Check that both")
        print("are genuinely running in non-thinking mode before training.")

    if max(student["truncation_rate"], teacher["truncation_rate"]) > 0.05:
        print(f"\nWARNING: truncation rate is already >5% at max_tokens={args.max_tokens}")
        print("before any training. OPD length inflation would push this higher and")
        print("truncated trajectories would start dominating the batch.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "student": {k: v for k, v in student.items() if k != "per_prompt"},
        "teacher": {k: v for k, v in teacher.items() if k != "per_prompt"},
        "teachable_fraction": teachable,
        f"pass@{args.k}_gap": {"point": point, "ci95_low": lo, "ci95_high": hi},
        "args": vars(args),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
