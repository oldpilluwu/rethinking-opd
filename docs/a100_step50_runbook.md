# Ubuntu A100: paper OPD through step 50

This runbook launches the checked-in 50-step prefix of the repository paper's
Qwen3-1.7B / Qwen3-4B-Non-Thinking-RL-Math experiment. It is deliberately not
the Lightning-OPD standard experiment.

## 1. Clone, switch branch, and install the environment

Run on the Ubuntu A100 instance:

```bash
cd /home/ubuntu
git clone https://github.com/oldpilluwu/rethinking-opd.git
cd /home/ubuntu/rethinking-opd
git fetch --all --prune
git switch opd
git pull --ff-only origin opd

bash scripts/setup/setup_a100_ubuntu.sh
source .venv/bin/activate
tmux -V
nvidia-smi
```

The setup uses Python 3.12 in `.venv`; it does not use Conda. It installs tmux,
vLLM 0.11.0, the matching torch/FlashAttention stack, and constrained versions
that avoid the NumPy/SciPy/OpenCV/CuPy and packaging/fsspec conflicts.

## 2. Download immutable model revisions and validate assets

```bash
cd /home/ubuntu/rethinking-opd
source .venv/bin/activate
bash scripts/setup/download_paper_models.sh

CONFIG=configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100_step50.toml
EXP=paper_qwen3_1p7b_qwen3_4b_non_thinking_rl_math_topk16_a100_step50

python scripts/setup/check_a100_env.py \
  --config "$CONFIG" \
  --require-assets \
  --output "diag_out/$EXP/environment-preflight.json"

CONFIG_ONLY=1 bash opd_lightning_a100.sh "$CONFIG"
```

The default teacher download is the immutable public
`Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` upload. The author's G-OPD
repository specifies only the bare local teacher name and does not provide an
official Hub URL. If the original checkpoint is available, set
`TEACHER_MODEL_ID` and `TEACHER_REVISION` before running the download script.

Inspect the resolved YAML path printed by `CONFIG_ONLY`. Do not launch if it is
not on branch `opd`, the environment check fails, or the resolved config differs
from the TOML.

## 3. Launch training in tmux

```bash
cd /home/ubuntu/rethinking-opd
CONFIG=configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100_step50.toml

tmux new-session -d -s opd50 \
  "bash -lc 'cd /home/ubuntu/rethinking-opd && source .venv/bin/activate && bash opd_lightning_a100.sh $CONFIG'"

tmux attach -t opd50
```

Detach with `Ctrl-b`, then `d`. From another SSH shell, monitor the compact
health view and GPU telemetry:

```bash
cd /home/ubuntu/rethinking-opd
source .venv/bin/activate
python scripts/diag/watch_run.py --follow --total-steps 50
```

The `opd50` tmux session exits when training succeeds or fails; the log and
diagnostic files remain available. Use `tmux has-session -t opd50` to test
whether it is still running and `tmux capture-pane -pt opd50` while it exists.

The config records all scalar metrics on every step. At steps
1–5, 10, 15, 20, 30, 40, and 50 it saves the model checkpoint, full decoded
rollouts, and SwanLab plots. Only step 50 includes optimizer state and can be
resumed. The launcher also records five-second GPU utilization, memory,
temperature, power, and clocks, plus Ray logs and the complete package/GPU/git
environment.

## 4. Verify and package diagnostics (no checkpoints)

After training finishes:

```bash
cd /home/ubuntu/rethinking-opd
source .venv/bin/activate
CONFIG=configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100_step50.toml

python scripts/diag/verify_run_outputs.py --config "$CONFIG"
python scripts/diag/package_run_artifacts.py \
  --config "$CONFIG" \
  --kind diagnostics
DIAG_ARCHIVE=$(find opd_artifacts -maxdepth 1 -name '*-diagnostics.tar.gz' -print -quit)
sha256sum "$DIAG_ARCHIVE" | tee "$DIAG_ARCHIVE.sha256"
```

The diagnostics archive contains no checkpoint weights or merged models.

## 5. Sync diagnostics to the local PC before evaluation

Run in local PowerShell, replacing the host:

```powershell
$A100_HOST = "ubuntu@REPLACE_WITH_HOST_OR_IP"
$REMOTE_REPO = "/home/ubuntu/rethinking-opd"
$LOCAL_DEST = "C:\Users\fawwa\projects\rethinking-opd\opd_artifacts"
New-Item -ItemType Directory -Force -Path $LOCAL_DEST | Out-Null

scp "${A100_HOST}:${REMOTE_REPO}/opd_artifacts/*-diagnostics.tar.gz" $LOCAL_DEST
scp "${A100_HOST}:${REMOTE_REPO}/opd_artifacts/*-diagnostics.tar.gz.sha256" $LOCAL_DEST
Get-FileHash "$LOCAL_DEST\*-diagnostics.tar.gz" -Algorithm SHA256
```

## 6. Run the exact configured benchmark suite

Evaluate the initial student and step 50 on AIME24 and AIME25 using the
paper config's `avg@16` settings:

```bash
cd /home/ubuntu/rethinking-opd
source .venv/bin/activate
CONFIG=configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100_step50.toml
EXP=paper_qwen3_1p7b_qwen3_4b_non_thinking_rl_math_topk16_a100_step50

ray stop --force || true
nvidia-smi

python scripts/eval/run_opd_eval.py \
  --config "$CONFIG" \
  --model baseline=model/Qwen3-1.7B \
  --model step50="checkpoint/$EXP/global_step_50" \
  --gpus 0

python scripts/diag/package_run_artifacts.py \
  --config "$CONFIG" \
  --kind benchmarks
BENCH_ARCHIVE=$(find opd_artifacts -maxdepth 1 -name '*-benchmarks.tar.gz' -print -quit)
sha256sum "$BENCH_ARCHIVE" | tee "$BENCH_ARCHIVE.sha256"
```

Generation JSONL, graded JSONL, manifests, and summaries are included. The
temporary merged Hugging Face checkpoint is excluded from the benchmark
archive.

## 7. Sync all benchmark outputs to the local PC

Run in the same local PowerShell session:

```powershell
scp "${A100_HOST}:${REMOTE_REPO}/opd_artifacts/*-benchmarks.tar.gz" $LOCAL_DEST
scp "${A100_HOST}:${REMOTE_REPO}/opd_artifacts/*-benchmarks.tar.gz.sha256" $LOCAL_DEST

Get-FileHash "$LOCAL_DEST\*-benchmarks.tar.gz" -Algorithm SHA256
```

Do not copy `checkpoint/` or `validation_log/.../merged_models/` to the local
PC unless checkpoint weights are explicitly needed later.
