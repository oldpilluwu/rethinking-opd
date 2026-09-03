# OPD experiment configs

Each TOML file is a complete experiment specification for the config-driven
OPD launcher. Start by copying the closest checked-in config; do not edit the
launcher. Use `opd_lightning_a100.sh` for the A100 profiles and
`opd_2x5090.sh` for a two-RTX-5090 profile with hardware validation.

```bash
cp configs/opd/lightning_standard_a100.toml configs/opd/my_experiment.toml
$EDITOR configs/opd/my_experiment.toml
CONFIG_ONLY=1 bash opd_lightning_a100.sh configs/opd/my_experiment.toml
bash opd_lightning_a100.sh configs/opd/my_experiment.toml
```

The 5090 launcher accepts a custom two-GPU config in the same way:

```bash
CONFIG_ONLY=1 bash opd_2x5090.sh configs/opd/my_2gpu_experiment.toml
bash opd_2x5090.sh configs/opd/my_2gpu_experiment.toml
```

It requires `runtime.nodes=1` and `runtime.gpus_per_node=2`. Unless
`CUDA_VISIBLE_DEVICES` is already set, it exposes devices `0,1`. The hardware
preflight is intentionally skipped by `CONFIG_ONLY=1`, so configs can be
resolved without RTX 5090s (the launcher's normal model/data preflight still
requires the configured experiment assets).

`CONFIG_ONLY=1` resolves Hydra and compares every managed value against the
TOML. Training is rejected if a command-line override changes one of those
values. Unmanaged Hydra options can be recorded in `hydra.extra_overrides`.

## Schema

- `experiment`: stable run name, tracking project, checkpoints, logs,
  diagnostics, and validation-output roots. Give each independent run a unique
  `name`.
- `models`: student and teacher paths, Qwen thinking-template mode, FSDP master
  dtype, teacher dtype, gradient checkpointing, and activation offload.
- `data`: train/validation parquet paths, dataset order seed, shuffle, prompt
  and response limits, truncation, and preflight policy. `lightning_dapo`
  enforces the official filtered Lightning DAPO artifact; `paper_dapo`
  enforces the paper repository's processed 17,917-row DAPO artifact and its
  Qwen3 non-thinking 1,024-token subset; `exists` only checks the path; `none`
  permits remote/custom dataset paths.
- `rollout`: engine, responses per prompt, student sampling temperature,
  unscaled/scaled teacher temperature, top-p, and repetition penalty.
- `objective`: `sampled_token` or `top_k` reverse-KL, top-k size and token-set
  strategy, reward weighting, advantage clipping, loss aggregation, entropy,
  task-reward weight, and optional PPO-style KL penalties.
- `optimizer`: learning rate, schedule, warmup, weight decay, Adam betas and
  epsilon, and gradient clipping.
- `training`: exact steps, epochs, prompt batch, optimizer mini-batch, PPO
  epochs, and actor/teacher micro-batches.
- `runtime`: GPU/node topology, tensor and sequence parallelism, dynamic token
  budgets, vLLM memory fraction, FSDP offload, and prefetch settings.
- `checkpoints`: model checkpoint steps and the subset that also retain
  optimizer state. Optimizer steps must also be model-save steps.
- `validation`: enable/frequency and generation sampling values, prompt suffix,
  deterministic evaluator seed, vLLM memory/dtype, and grading method. A
  frequency of `-1` disables verl-side validation; these values are still used
  by `scripts/eval/run_opd_eval.py` for external checkpoint evaluation.
- `reward`: optional format reward and custom task reward function. With
  `token_reward_direct`, task reward is diagnostic unless its estimator/weight
  is explicitly changed.
- `tracking`: logger backends, SwanLab mode, explicit plot steps/frequency, OPD
  text diagnostics, and scheduled full-rollout JSONL dumps. When the `file`
  logger is enabled, every scalar event is flushed to
  `diagnostics_dir/experiment_name/metrics.jsonl`.
- `hydra.extra_overrides`: advanced settings not represented elsewhere. Each
  entry is a complete Hydra `key=value` string.

The seed is applied to dataset order, the vLLM engine, and the rollout RNG. CUDA
kernels can still introduce small hardware/library-dependent numerical
differences, so exact reproduction also requires the same code commit, model
and dataset revisions, package lock, and GPU stack.

Every launched run archives these files in its checkpoint root:

- `experiment.toml`: byte-for-byte source config;
- `experiment.json`: canonical parsed config;
- `resolved_hydra.yaml`: the final framework configuration actually launched.

It also records static environment provenance, five-second GPU telemetry, Ray
logs, SwanLab data, scalar JSONL, and configured rollout dumps under the
experiment's diagnostics directory. Create a checkpoint-free transfer archive
with:

```bash
python scripts/diag/package_run_artifacts.py \
  --config configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100_step50.toml \
  --kind diagnostics
```

Resuming requires the TOML to match the archived source config. Set
`ALLOW_CONFIG_DRIFT=1` only when intentionally changing a resumed experiment;
the replacement config and resolved Hydra state will then be archived.

## External evaluation

The evaluator accepts a base Hugging Face model, a merged checkpoint, or a raw
verl `global_step_N` checkpoint. Raw checkpoints are merged automatically:

```bash
python scripts/eval/run_opd_eval.py \
  --config configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100.toml \
  --model baseline=model/Qwen3-1.7B \
  --model step279=checkpoint/paper_qwen3_1p7b_qwen3_4b_non_thinking_rl_math_topk16_a100/global_step_279 \
  --gpus 0
```

Generation JSONL, per-rollout grades, and `summary.json` are written below the
experiment's `validation_dir`. Existing complete generations are reused unless
`--overwrite` is supplied. Use `--generate-only` and `--grade-only` to split
the GPU generation and CPU grading phases.
