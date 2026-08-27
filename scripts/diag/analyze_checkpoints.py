#!/usr/bin/env python3
"""End-of-run analysis over the saved OPD checkpoints.

Everything here is a pure function of the weights, so nothing is lost by running it
after training rather than during. (The metrics that *are* lost -- teacher logits at
student-visited states -- are logged live by verl/trainer/ppo/opd_diagnostics.py.)

Subcommands
-----------
  weights      Weight drift from the SFT init. Reads the sharded .pt directly, so it
               needs no merge step and no GPU. Seconds per checkpoint.

  build-probe  Generate a frozen probe set once from the SFT checkpoint: N prompts,
               one rollout each, stored as token ids. Every checkpoint is then scored
               at IDENTICAL states, which is what makes the curves comparable. If you
               re-rolled per checkpoint, the states would move and you could no longer
               separate "aligned with the teacher" from "went somewhere else".

  probe        Score merged HF checkpoints on that probe and report alignment vs the
               teacher and drift vs the SFT init. Needs a GPU.

Typical sequence
----------------
    python scripts/diag/analyze_checkpoints.py weights \\
        --ckpt-dir checkpoint/<exp> --ref model/Qwen3-1.7B-SFT

    bash scripts/diag/convert_checkpoints.sh checkpoint/<exp> hf_ckpts

    python scripts/diag/analyze_checkpoints.py build-probe \\
        --sft model/Qwen3-1.7B-SFT --parquet datasets/dapo-math-17k.parquet

    python scripts/diag/analyze_checkpoints.py probe \\
        --hf-dir hf_ckpts --sft model/Qwen3-1.7B-SFT --teacher model/Qwen3-4B
"""

import argparse
import glob
import json
import os
import random
import re

import torch

import sys as _sys_boot
_sys_boot.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gpu_isolate import run_isolated

# FlashInfer JIT-compiles its sampling kernels and needs nvcc (the CUDA toolkit, not
# just the driver). Fall back to vLLM's native top-k/top-p sampler when nvcc is absent.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

TOP_K = 16
LOGP_FLOOR = -30.0


# --------------------------------------------------------------------------- utils
def step_of(path):
    m = re.search(r"global_step_(\d+)", path)
    return int(m.group(1)) if m else -1


def list_steps(root, pattern="global_step_*"):
    dirs = glob.glob(os.path.join(root, pattern))
    return sorted([d for d in dirs if step_of(d) >= 0], key=step_of)


def _materialize(v):
    """Sharded checkpoints hold DTensor/ShardedTensor. Reduce to a plain CPU tensor."""
    if hasattr(v, "full_tensor"):
        try:
            return v.full_tensor().cpu()
        except Exception:
            pass
    if hasattr(v, "to_local"):
        try:
            return v.to_local().cpu()
        except Exception:
            pass
    if hasattr(v, "local_shards"):
        shards = v.local_shards()
        if len(shards) == 1:
            return shards[0].tensor.cpu()
    if torch.is_tensor(v):
        return v.cpu()
    return None


_PREFIXES = ("_fsdp_wrapped_module.", "module.", "_checkpoint_wrapped_module.")


def _clean_key(k):
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if p in k:
                k = k.replace(p, "")
                changed = True
    return k


def load_shard_state_dict(step_dir):
    hits = glob.glob(os.path.join(step_dir, "actor", "model_world_size_*_rank_0.pt"))
    if not hits:
        return None
    sd = torch.load(hits[0], map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        t = _materialize(v)
        if t is not None and t.is_floating_point():
            out[_clean_key(k)] = t
    return out


def load_reference_state_dict(ref_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=torch.float32, trust_remote_code=True)
    sd = {_clean_key(k): v.detach().cpu() for k, v in model.state_dict().items() if v.is_floating_point()}
    del model
    return sd


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        return f"layer_{int(m.group(1)):02d}"
    if "embed" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "other"


# ------------------------------------------------------------------------ weights
def cmd_weights(args):
    ref = load_reference_state_dict(args.ref)
    ref_norm = torch.sqrt(sum((v.double() ** 2).sum() for v in ref.values())).item()
    print(f"reference: {args.ref}  ({len(ref)} float tensors, ||theta_0|| = {ref_norm:.4f})\n")

    rows = []
    prev_sd, prev_step = None, None
    for step_dir in list_steps(args.ckpt_dir):
        step = step_of(step_dir)
        sd = load_shard_state_dict(step_dir)
        if sd is None:
            print(f"step {step:>4}: no model shard, skipped")
            continue

        shared = [k for k in sd if k in ref]
        if not shared:
            print(f"step {step:>4}: no overlapping keys with the reference, skipped")
            continue

        sq_total, dot, cur_sq = 0.0, 0.0, 0.0
        per_layer = {}
        for k in shared:
            a = sd[k].double().flatten()
            b = ref[k].double().flatten()
            if a.numel() != b.numel():
                continue
            d = a - b
            s = (d**2).sum().item()
            sq_total += s
            dot += (a * b).sum().item()
            cur_sq += (a**2).sum().item()
            lay = layer_of(k)
            per_layer[lay] = per_layer.get(lay, 0.0) + s

        drift = sq_total**0.5
        cos = dot / max((cur_sq**0.5) * ref_norm, 1e-12)
        row = {
            "step": step,
            "drift_l2": drift,
            "drift_relative": drift / max(ref_norm, 1e-12),
            "cosine_to_init": cos,
            "per_layer_l2": {k: v**0.5 for k, v in sorted(per_layer.items())},
        }
        # True distance between consecutive saved checkpoints, ||theta_t - theta_{t-1}||.
        # NOT the same as the change in distance-from-init: the two only coincide when
        # successive updates are radially aligned, which they are not.
        if prev_sd is not None:
            sq_adj = 0.0
            for k in shared:
                if k not in prev_sd:
                    continue
                a = sd[k].double().flatten()
                b = prev_sd[k].double().flatten()
                if a.numel() != b.numel():
                    continue
                sq_adj += ((a - b) ** 2).sum().item()
            gap = max(step - prev_step, 1)
            row["adjacent_from_step"] = prev_step
            row["net_displacement"] = sq_adj**0.5
            # NET displacement per elapsed step, not the average size of an individual
            # update: intervening updates can partly cancel, so this is a lower bound on
            # how far the optimizer actually travelled. Only the consecutive pairs in the
            # 1-5 cluster (gap == 1) measure a single update directly.
            row["net_displacement_per_step"] = (sq_adj**0.5) / gap
            row["adjacent_gap"] = gap

        rows.append(row)
        print(
            f"step {step:>4}: ||dtheta|| = {drift:>10.5f}   "
            f"relative = {row['drift_relative']:.3e}   cos = {cos:.8f}"
            + (f"   net_disp = {row['net_displacement']:.6f}" if "net_displacement" in row else "")
        )

        if args.adjacent:
            del prev_sd
            prev_sd, prev_step = sd, step
        else:
            del sd

    if any("net_displacement" in r for r in rows):
        print("\nnet displacement between consecutive saved checkpoints:")
        print(f"  {'from':>6}{'to':>6}{'gap':>5}{'net disp':>14}{'per step':>14}")
        for r in rows:
            if "net_displacement" in r:
                print(
                    f"  {r['adjacent_from_step']:>6}{r['step']:>6}{r['adjacent_gap']:>5}"
                    f"{r['net_displacement']:>14.6f}{r['net_displacement_per_step']:>14.6f}"
                )
        print("  NB: net displacement, not distance travelled -- updates within a gap can")
        print("  cancel. Only gap==1 rows (the 1-5 cluster) measure a single update.")
    elif len(rows) >= 2:
        print("\n(adjacent distances disabled; --adjacent holds two checkpoints in RAM)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")


# -------------------------------------------------------------------- build-probe
def cmd_build_probe(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = pq.read_table(args.parquet).to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_prompts]

    tok = AutoTokenizer.from_pretrained(args.sft, trust_remote_code=True)
    rendered = []
    for r in rows:
        try:
            rendered.append(
                tok.apply_chat_template(
                    r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
        except TypeError:
            rendered.append(tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True))

    llm = LLM(
        model=args.sft,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens + 1024,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    outs = llm.generate(
        rendered, SamplingParams(n=1, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens)
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w") as f:
        for prompt_text, out in zip(rendered, outs):
            cand = out.outputs[0]
            if len(cand.token_ids) < 8:
                continue
            f.write(
                json.dumps(
                    {
                        "prompt_ids": list(out.prompt_token_ids),
                        "response_ids": list(cand.token_ids),
                    }
                )
                + "\n"
            )
            kept += 1
    print(f"wrote {kept} frozen probe sequences to {args.out}")


# -------------------------------------------------------------------------- probe
def _score_model(model_path, probe, device, max_len, batch_size):
    """Return per-token logprob of the probe tokens plus top-K ids/logprobs."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    tok_logp, topk_ids, topk_logp = [], [], []
    with torch.no_grad():
        for i in range(0, len(probe), batch_size):
            for item in probe[i : i + batch_size]:
                ids = (item["prompt_ids"] + item["response_ids"])[:max_len]
                n_resp = min(len(item["response_ids"]), max_len - len(item["prompt_ids"]))
                if n_resp <= 0:
                    continue
                x = torch.tensor([ids], device=device)
                logits = model(x).logits[0].float()
                # logits[t] predicts token t+1; response occupies the final n_resp slots
                sl = logits[-n_resp - 1 : -1]
                lsm = torch.log_softmax(sl, dim=-1)
                targets = x[0, -n_resp:]
                tok_logp.append(lsm.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu())
                v, idx = lsm.topk(TOP_K, dim=-1)
                topk_ids.append(idx.cpu())
                topk_logp.append(v.cpu())

    del model
    torch.cuda.empty_cache()
    return (
        torch.cat(tok_logp),
        torch.cat(topk_ids),
        torch.cat(topk_logp),
    )


def _pair_metrics(a, b, prefix):
    """a, b = (tok_logp, topk_ids, topk_logp), both scored on the SAME frozen probe.

    NOTE ON ``logratio_on_probe_states``: this is mean[log p_a(x) - log p_b(x)] over
    tokens that were sampled ONCE from the SFT init. It is a fixed-data log-likelihood
    ratio, NOT a reverse KL -- a reverse KL would require the expectation to be taken
    under the current checkpoint's own policy, which would mean re-rolling out per
    checkpoint and losing the fixed states that make these curves comparable at all.
    It is a legitimate comparison across checkpoints; it is not the training objective,
    and there is no reason for it to be monotone. The on-policy version of this
    quantity is logged live during training as ``opd/reverse_kl_sampled``.
    """
    a_lp, a_ids, a_tk = a
    b_lp, b_ids, b_tk = b
    n = min(a_lp.numel(), b_lp.numel())
    a_lp, b_lp = a_lp[:n], b_lp[:n]
    a_ids, b_ids = a_ids[:n], b_ids[:n]
    a_tk, b_tk = a_tk[:n], b_tk[:n]

    out = {f"{prefix}/logratio_on_probe_states": (a_lp - b_lp).mean().item()}
    out[f"{prefix}/top1_agreement"] = (a_ids[:, 0] == b_ids[:, 0]).float().mean().item()

    overlap = (a_ids.unsqueeze(-1) == b_ids.unsqueeze(-2)).any(-1).float()  # (N, K)
    out[f"{prefix}/overlap_ratio"] = overlap.mean().item()

    a_p = a_tk.clamp_min(LOGP_FLOOR).exp()
    out[f"{prefix}/mass_on_shared"] = (a_p * overlap).sum(-1).mean().item()
    out[f"{prefix}/topk_mass"] = a_p.sum(-1).mean().item()
    return out


def cmd_probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = [json.loads(l) for l in open(args.probe)]
    if args.limit:
        probe = probe[: args.limit]
    print(f"probe: {len(probe)} sequences\n")

    print("scoring SFT init ...")
    sft = _score_model(args.sft, probe, device, args.max_len, args.batch_size)
    print("scoring teacher ...")
    teacher = _score_model(args.teacher, probe, device, args.max_len, args.batch_size)

    rows = []
    for d in list_steps(args.hf_dir):
        step = step_of(d)
        print(f"scoring step {step} ...")
        try:
            cur = _score_model(d, probe, device, args.max_len, args.batch_size)
        except Exception as e:
            print(f"  skipped: {type(e).__name__}: {e}")
            continue
        row = {"step": step}
        row.update(_pair_metrics(cur, teacher, "vs_teacher"))
        row.update(_pair_metrics(cur, sft, "vs_sft"))
        rows.append(row)

    rows.sort(key=lambda r: r["step"])
    cols = [
        "vs_teacher/logratio_on_probe_states",
        "vs_teacher/top1_agreement",
        "vs_teacher/overlap_ratio",
        "vs_teacher/mass_on_shared",
        "vs_sft/logratio_on_probe_states",
        "vs_sft/top1_agreement",
    ]
    print("\n" + "step".rjust(6) + "".join(c.split("/")[-1][:13].rjust(15) for c in cols))
    print("-" * (6 + 15 * len(cols)))
    for r in rows:
        print(f"{r['step']:>6}" + "".join(f"{r.get(c, float('nan')):>15.5f}" for c in cols))

    print("\nWhat to look for:")
    print("  vs_teacher/overlap_ratio       should rise -- the paper's alignment claim, and")
    print("                                 the most trustworthy series here")
    print("  vs_teacher/top1_agreement      should rise alongside it")
    print("  vs_teacher/mass_on_shared      should approach 0.97-0.99")
    print("  vs_*/logratio_on_probe_states  fixed-data log-likelihood ratios on states frozen")
    print("                                 from the SFT init. NOT a KL and NOT the training")
    print("                                 objective -- no reason to be monotone. Use them")
    print("                                 for direction and relative magnitude only.")
    print("  vs_sft/*                       how far the policy moved. Large movement from the")
    print("                                 SFT init with a flat vs_teacher/overlap_ratio is")
    print("                                 drift, not learning.")
    print("\n  The on-policy reverse KL -- the actual objective -- is in the training log as")
    print("  opd/reverse_kl_sampled. It cannot be recovered from checkpoints.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")


# -------------------------------------------------------------------------- bench
def _bench_one_model(model_path, benches, k, temperature, top_p, max_tokens, gpu_mem_util, seed):
    """Generate and grade one model against every benchmark.

    Module-level so it can be pickled into an isolated subprocess. vLLM v1 runs its
    engine in a child process that `del llm` does not reliably reap, so loading 17
    checkpoints in one process leaves the GPU occupied and the second model fails with
    "Free memory on device ... is less than desired GPU memory utilization". One process
    per model, and the OS frees everything on exit.
    """
    import os as _os
    import sys as _sys

    _os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _os.path.join(_repo, "verl") not in _sys.path:
        _sys.path.insert(0, _os.path.join(_repo, "verl"))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.trainer.ppo.opd_diagnostics import REPETITION_THRESHOLD, repetition_score
    from verl.utils.reward_score.ttrl_math import reward_func

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_tokens + 1024,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    # Fixed seed so reruns of a checkpoint reproduce, and different checkpoints face the
    # same sampling noise.
    sp = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

    out = {}
    for name, rows in benches.items():
        rendered = []
        for r in rows:
            try:
                rendered.append(
                    tok.apply_chat_template(
                        r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                )
            except TypeError:
                rendered.append(
                    tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
                )

        outs = llm.generate(rendered, sp)
        per_problem, lengths, rep_scores = [], [], []
        trunc, total = 0, 0
        for r, o in zip(rows, outs):
            gt = str(r["reward_model"]["ground_truth"])
            flags = []
            for cand in o.outputs:
                total += 1
                lengths.append(len(cand.token_ids))
                ok = False
                try:
                    res = reward_func(r["data_source"], cand.text, gt)
                    if isinstance(res, dict):
                        res = res.get("score", 0.0)
                    ok = float(res) > 0.5
                except Exception:
                    pass
                flags.append(ok)
                if getattr(cand, "finish_reason", None) == "length":
                    trunc += 1
                sc = repetition_score(cand.text)
                if sc is not None:
                    rep_scores.append(sc)
            per_problem.append(flags)

        lengths.sort()
        out[name] = {
            "per_problem": per_problem,
            "lengths": lengths,
            "trunc": trunc,
            "total": total,
            "rep_mean": (sum(rep_scores) / len(rep_scores)) if rep_scores else 0.0,
            "rep_rate": (
                sum(s >= REPETITION_THRESHOLD for s in rep_scores) / len(rep_scores)
                if rep_scores
                else 0.0
            ),
        }
    return out


def cmd_bench(args):
    """Behavioural benchmark over merged HF checkpoints.

    Two presets, because one set of settings cannot serve both purposes:

    --preset paper    AIME24/25 + AMC23, avg@16, T=0.7, top-p 0.95, 31744 max tokens.
                      Comparable to the Rethinking-OPD numbers. The long allowance is
                      what makes it comparable -- truncating at 8k silently caps the
                      score and the comparison stops meaning anything.

    --preset health   Training-style prompts at the OPD sampling settings (T=1.0,
                      top-p 1.0) and the training response limit. A bounded-length
                      collapse check, NOT a paper-comparable accuracy number.

    Alongside accuracy this reports length percentiles, truncation and repetition,
    because the reported OPD collapse mode is abrupt length inflation -- truncated
    trajectories come to dominate and accuracy hides it until late
    (StableOPD, arXiv 2604.08527).

    The SFT init and the teacher are always run as baseline rows: an accuracy column
    with no step-0 reference cannot be read. Sampling is seeded, and pass@k plus a
    paired bootstrap CI against step 0 are reported -- on AIME24's 30 problems, avg@16
    differences of several points are routinely noise.
    """
    import pyarrow.parquet as pq

    if args.preset == "paper":
        temperature, top_p = 0.7, 0.95
        max_tokens = args.max_tokens or 31744
    else:
        temperature, top_p = 1.0, 1.0
        max_tokens = args.max_tokens or 4096
    print(f"preset={args.preset}  T={temperature}  top_p={top_p}  max_tokens={max_tokens}  k={args.k}\n")

    benches = {}
    for bf in args.benchmarks.split(","):
        name = os.path.basename(os.path.dirname(bf))
        benches[name] = pq.read_table(bf).to_pylist()
        print(f"{name}: {len(benches[name])} problems")

    # Baselines are mandatory, not optional: checkpoint accuracies are only interpretable
    # relative to the SFT init, and the teacher bounds what OPD can reach.
    targets = [(args.sft, "sft", -1)]
    if args.teacher:
        targets.append((args.teacher, "teacher", -2))
    targets += [(d, f"step_{step_of(d)}", step_of(d)) for d in list_steps(args.hf_dir)]

    baseline_correct = {}
    all_rows = []
    for target, label, step in targets:
        print(f"\n=== {label} ===")
        try:
            raw = run_isolated(
                _bench_one_model,
                model_path=target,
                benches=benches,
                k=args.k,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                gpu_mem_util=args.gpu_mem_util,
                seed=args.seed,
            )
        except Exception as e:
            print(f"  skipped: {e}")
            continue

        for name, d in raw.items():
            per_problem, lengths = d["per_problem"], d["lengths"]
            n_prob = max(len(per_problem), 1)

            def _pct(q, _l=lengths):
                return _l[int(q * (len(_l) - 1))] if _l else 0

            row = {
                "checkpoint": label,
                "step": step,
                "benchmark": name,
                f"avg@{args.k}": sum(sum(f) / len(f) for f in per_problem) / n_prob,
                "mean_len": sum(lengths) / max(len(lengths), 1),
                "p50_len": _pct(0.50),
                "p95_len": _pct(0.95),
                "p99_len": _pct(0.99),
                "truncation_rate": d["trunc"] / max(d["total"], 1),
                "repetition_score_mean": d["rep_mean"],
                "repetition_rate": d["rep_rate"],
            }
            # pass@j for j = 1, 4, 8, k -- avg@k alone hides whether the model gained
            # reliability or only lost diversity.
            for j in sorted({1, 4, 8, args.k}):
                if j <= args.k:
                    row[f"pass@{j}"] = sum(any(f[:j]) for f in per_problem) / n_prob

            if label == "sft":
                baseline_correct[name] = per_problem
            elif name in baseline_correct:
                pt, lo, hi = paired_bootstrap(baseline_correct[name], per_problem, seed=args.seed)
                row["delta_avg_vs_sft"] = pt
                row["delta_ci95"] = [lo, hi]

            all_rows.append(row)
            delta = (
                f"  d_vs_sft={row['delta_avg_vs_sft']:+.4f}"
                f" [{row['delta_ci95'][0]:+.3f},{row['delta_ci95'][1]:+.3f}]"
                if "delta_ci95" in row
                else ""
            )
            print(
                f"  {name:<18} avg@{args.k}={row[f'avg@{args.k}']:.4f}  "
                f"len p50/p95={row['p50_len']}/{row['p95_len']}  "
                f"trunc={row['truncation_rate']:.3f}  rep={row['repetition_rate']:.3f}{delta}"
            )

    print("\n" + "=" * 118)
    print(
        f"{'ckpt':>10}{'benchmark':>18}{'avg@' + str(args.k):>10}{'pass@1':>9}{'p50':>8}{'p95':>8}"
        f"{'trunc':>8}{'rep':>7}{'delta vs sft (95% CI)':>28}"
    )
    print("-" * 118)
    for r in sorted(all_rows, key=lambda x: (x["benchmark"], x["step"])):
        d = (
            f"{r['delta_avg_vs_sft']:+.4f} [{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]"
            if "delta_ci95" in r
            else "-"
        )
        print(
            f"{r['checkpoint']:>10}{r['benchmark']:>18}{r[f'avg@{args.k}']:>10.4f}"
            f"{r.get('pass@1', float('nan')):>9.4f}{r['p50_len']:>8}{r['p95_len']:>8}"
            f"{r['truncation_rate']:>8.3f}{r['repetition_rate']:>7.3f}{d:>28}"
        )
    print("\nReading this table:")
    print("  Treat a delta whose CI straddles 0 as no measured change. AIME24 is 30")
    print("  problems, so multi-point avg@16 swings are routinely inside the interval.")
    print("  Rising p95/truncation with flat or falling accuracy is the length-inflation")
    print("  collapse mode -- check that before reading the accuracy column as a result.")
    if args.preset == "health":
        print("  preset=health is a bounded collapse check. These accuracies are NOT")
        print("  comparable to published numbers; use --preset paper for that.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {args.out}")


REPO_ROOT_FOR_REWARD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def paired_bootstrap(base_pp, new_pp, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI on the paired avg@k difference (new - base).

    Resamples PROBLEMS, since the problem set is the unit of uncertainty and both models
    saw the same problems. AIME24 has 30 problems, so this interval is wide -- which is
    the point: multi-point avg@16 swings there are frequently noise.
    """
    rng = random.Random(seed)
    n = min(len(base_pp), len(new_pp))
    if n == 0:
        return 0.0, 0.0, 0.0
    b = [sum(f) / len(f) for f in base_pp[:n]]
    a = [sum(f) / len(f) for f in new_pp[:n]]
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    point = sum(a[i] - b[i] for i in range(n)) / n
    return point, diffs[int((alpha / 2) * (n_boot - 1))], diffs[int((1 - alpha / 2) * (n_boot - 1))]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("weights", help="weight drift from the SFT init (no GPU needed)")
    w.add_argument("--ckpt-dir", required=True)
    w.add_argument("--ref", required=True, help="HF path of the SFT init")
    w.add_argument(
        "--adjacent",
        action="store_true",
        default=True,
        help="also report ||theta_t - theta_{t-1}|| (holds two checkpoints in RAM, ~14GB at 1.7B fp32)",
    )
    w.add_argument("--no-adjacent", dest="adjacent", action="store_false")
    w.add_argument("--out", default="diag_out/weight_drift.json")
    w.set_defaults(func=cmd_weights)

    b = sub.add_parser("build-probe", help="freeze a probe set generated from the SFT init")
    b.add_argument("--sft", required=True)
    b.add_argument("--parquet", default="datasets/dapo-math-17k.parquet")
    b.add_argument("--n-prompts", type=int, default=256)
    b.add_argument("--max-tokens", type=int, default=4096)
    b.add_argument("--temperature", type=float, default=1.0, help="match the training temperature")
    b.add_argument("--gpu-mem-util", type=float, default=0.85)
    b.add_argument("--seed", type=int, default=1234)
    b.add_argument("--out", default="diag_out/probe.jsonl")
    b.set_defaults(func=cmd_build_probe)

    p = sub.add_parser("probe", help="score merged HF checkpoints on the frozen probe")
    p.add_argument("--hf-dir", required=True, help="dir of merged global_step_* HF checkpoints")
    p.add_argument("--sft", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--probe", default="diag_out/probe.jsonl")
    p.add_argument("--max-len", type=int, default=5120)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="diag_out/probe_metrics.json")
    p.set_defaults(func=cmd_probe)

    bn = sub.add_parser("bench", help="accuracy + length/truncation/repetition over merged checkpoints")
    bn.add_argument("--hf-dir", required=True)
    bn.add_argument("--sft", required=True, help="SFT init; always run as the step-0 baseline")
    bn.add_argument("--teacher", default=None, help="teacher; run as an upper-reference row")
    bn.add_argument(
        "--preset",
        choices=["paper", "health"],
        default="paper",
        help="paper: T=0.7 top-p=0.95 max=31744 (comparable to published numbers). "
        "health: T=1.0 top-p=1.0 max=4096 (bounded collapse check, NOT paper-comparable)",
    )
    bn.add_argument(
        "--benchmarks",
        default="datasets/test_data/AIME24/test.parquet,"
        "datasets/test_data/AIME25/test.parquet,"
        "datasets/test_data/AMC23/test.parquet",
    )
    bn.add_argument("--k", type=int, default=16, help="samples per problem (paper: avg@16)")
    bn.add_argument("--max-tokens", type=int, default=0, help="0 = use the preset's value")
    bn.add_argument("--seed", type=int, default=1234, help="fixed sampling seed for comparability")
    bn.add_argument("--gpu-mem-util", type=float, default=0.85)
    bn.add_argument("--out", default="diag_out/bench.json")
    bn.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
