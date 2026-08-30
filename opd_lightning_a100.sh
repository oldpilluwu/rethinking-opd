#!/bin/bash
# =============================================================================
# Config-driven OPD launcher for NVIDIA GPUs.
#
# Usage:
#   bash opd_lightning_a100.sh
#   bash opd_lightning_a100.sh configs/opd/paper_qwen3_1p7b_rl_math_teacher_a100.toml
#   CONFIG_ONLY=1 bash opd_lightning_a100.sh --config configs/opd/lightning_standard_a100.toml
#
# Experiment hyperparameters live in versioned TOML files, not in this launcher.
# Remaining command-line arguments are applied as final Hydra overrides and are
# checked against the TOML for all managed settings before training starts.
# =============================================================================

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

DEFAULT_CONFIG="configs/opd/lightning_standard_a100.toml"
CONFIG_PATH=${OPD_CONFIG:-$DEFAULT_CONFIG}
if [ "${1:-}" = "--config" ]; then
    if [ "$#" -lt 2 ]; then
        echo "ERROR: --config requires a TOML path" >&2
        exit 2
    fi
    CONFIG_PATH=$2
    shift 2
elif [[ "${1:-}" == *.toml ]]; then
    CONFIG_PATH=$1
    shift
fi

CONFIG_LOADER="scripts/config/load_opd_config.py"
python3 "$CONFIG_LOADER" validate "$CONFIG_PATH"

while IFS= read -r -d '' entry; do
    export "$entry"
done < <(python3 "$CONFIG_LOADER" env0 "$CONFIG_PATH")

HYDRA_ARGS=()
while IFS= read -r -d '' entry; do
    HYDRA_ARGS+=("$entry")
done < <(python3 "$CONFIG_LOADER" hydra-args0 "$CONFIG_PATH")

# ---------------------------------------------------------------- environment
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=$SEED
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export HYDRA_FULL_ERROR=1
export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-$HOME/.cache/outlines/$(python3 -c 'import uuid; print(uuid.uuid4())')}
export SWANLAB_MODE=${SWANLAB_MODE:-$CONFIG_SWANLAB_MODE}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$CONFIG_SWANLAB_LOG_DIR}

# ------------------------------------------------------------------- preflight
case "$DATA_PREFLIGHT" in
    lightning_dapo)
        python3 scripts/data/prepare_lightning_dapo.py validate \
            --input "$TRAIN_DATASET" \
            --tokenizer "$ACTOR_MODEL_PATH" \
            --max-prompt-length "$MAX_PROMPT_LENGTH"
        ;;
    paper_dapo)
        python3 scripts/data/validate_paper_dapo.py \
            --input "$TRAIN_DATASET" \
            --tokenizer "$ACTOR_MODEL_PATH" \
            --max-prompt-length "$MAX_PROMPT_LENGTH"
        ;;
    exists)
        if [ ! -f "$TRAIN_DATASET" ]; then
            echo "ERROR: training dataset does not exist: $TRAIN_DATASET" >&2
            exit 1
        fi
        ;;
    none)
        ;;
    *)
        echo "ERROR: unsupported data preflight mode: $DATA_PREFLIGHT" >&2
        exit 1
        ;;
esac

# --------------------------------------------------------------------- resume
RESUME_ARGS=("trainer.resume_mode=disable")
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$RESUME_FROM")
    if [ ! -f "$RESUME_FROM/manifest.json" ]; then
        echo "WARNING: no manifest.json in $RESUME_FROM"
    elif ! grep -q '"resumable": true' "$RESUME_FROM/manifest.json"; then
        echo "ERROR: $RESUME_FROM is marked NOT resumable in its manifest." >&2
        echo "Only steps in checkpoints.optimizer_save_steps carry optimizer state." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------------- logs
mkdir -p "$LOG_DIR" "$DIAGNOSTICS_DIR" "$SWANLAB_LOG_DIR"
mkdir -p "$(dirname "$VERL_FILE_LOGGER_PATH")"
SAFE_EXPERIMENT_NAME=${EXPERIMENT_NAME//[^a-zA-Z0-9_.-]/_}
LOG_FILE="$LOG_DIR/${SAFE_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x
echo "experiment config: $OPD_CONFIG_PATH"
echo "student: $ACTOR_MODEL_PATH"
echo "teacher: $REWARD_MODEL_PATH"
echo "log: $LOG_FILE"

run_verl() {
    python3 -m verl.trainer.main_ppo \
        "${HYDRA_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "$@"
}

# Resolve and compare every managed value to the source TOML before Ray starts.
RESOLVED_CONFIG="${LOG_FILE%.log}_resolved.yaml"
run_verl --cfg job --resolve > "$RESOLVED_CONFIG"
python3 scripts/diag/validate_opd_config.py "$RESOLVED_CONFIG" "$CONFIG_PATH"

if [ "${CONFIG_ONLY:-0}" = "1" ]; then
    echo "resolved config: $RESOLVED_CONFIG"
    exit 0
fi

if [ -z "${RESUME_FROM:-}" ] && [ -d "$CKPT_PATH" ] && [ -n "$(ls -A "$CKPT_PATH" 2>/dev/null)" ]; then
    echo "ERROR: checkpoint directory is not empty:" >&2
    echo "  $CKPT_PATH" >&2
    echo "Set RESUME_FROM=<checkpoint> or choose a new experiment.name." >&2
    if [ "${FORCE_DIRTY_CKPT:-0}" != "1" ]; then
        exit 1
    fi
    echo "FORCE_DIRTY_CKPT=1 set -- continuing into a non-empty directory."
fi

if [ -n "${RESUME_FROM:-}" ] && [ -f "$CKPT_PATH/experiment.toml" ] \
    && ! cmp -s "$CONFIG_PATH" "$CKPT_PATH/experiment.toml"; then
    echo "ERROR: resume config differs from $CKPT_PATH/experiment.toml" >&2
    echo "Use the archived config, or set ALLOW_CONFIG_DRIFT=1 deliberately." >&2
    if [ "${ALLOW_CONFIG_DRIFT:-0}" != "1" ]; then
        exit 1
    fi
    echo "ALLOW_CONFIG_DRIFT=1 set -- archiving the new resume config."
fi

# Archive both human-authored and machine-resolved configs with the run.
mkdir -p "$CKPT_PATH"
cp "$CONFIG_PATH" "$CKPT_PATH/experiment.toml"
python3 "$CONFIG_LOADER" show "$CONFIG_PATH" > "$CKPT_PATH/experiment.json"
cp "$RESOLVED_CONFIG" "$CKPT_PATH/resolved_hydra.yaml"

# Static provenance plus 5-second GPU telemetry make OOMs, thermal throttling,
# utilization gaps, and memory spikes inspectable after the instance is gone.
python3 --version > "$DIAGNOSTICS_DIR/python-version.txt" 2>&1
python3 -m pip freeze > "$DIAGNOSTICS_DIR/pip-freeze.txt"
git rev-parse HEAD > "$DIAGNOSTICS_DIR/git-commit.txt"
git status --short --branch > "$DIAGNOSTICS_DIR/git-status.txt"
uname -a > "$DIAGNOSTICS_DIR/uname.txt"
lscpu > "$DIAGNOSTICS_DIR/lscpu.txt"
nvidia-smi -q > "$DIAGNOSTICS_DIR/nvidia-smi-q.txt"
sha256sum "$TRAIN_DATASET" > "$DIAGNOSTICS_DIR/training-dataset.sha256"
if [ -f model/paper-model-revisions.txt ]; then
    cp model/paper-model-revisions.txt "$DIAGNOSTICS_DIR/model-revisions.txt"
fi

GPU_CSV="$DIAGNOSTICS_DIR/gpu-telemetry.csv"
printf '%s\n' "timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,clocks_sm_mhz" > "$GPU_CSV"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader,nounits -l 5 >> "$GPU_CSV" &
GPU_MONITOR_PID=$!
# Ray builds its plasma store socket as
# <temp-dir>/session_<timestamp>_<pid>/sockets/plasma_store, and an AF_UNIX path
# cannot exceed 107 bytes. Rooting the session under the diagnostics directory
# overruns that limit for long experiment names, so the live session uses a
# short /tmp path and its text logs are copied into the diagnostics tree.
RAY_TEMP_DIR=$(mktemp -d /tmp/ray-opd-XXXXXX)
RAY_LOG_DEST="$(cd "$DIAGNOSTICS_DIR" && pwd)/ray"

collect_ray_logs() {
    [ -d "$RAY_TEMP_DIR" ] || return 0
    while IFS= read -r -d '' logdir; do
        dest="$RAY_LOG_DEST/${logdir#./}"
        mkdir -p "$dest"
        cp -R "$RAY_TEMP_DIR/${logdir#./}/." "$dest/" 2>/dev/null || true
    done < <(cd "$RAY_TEMP_DIR" && find . -type d -name logs -print0)
}

cleanup_run() {
    if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    collect_ray_logs
}
trap cleanup_run EXIT

ray stop --force || true
ray start --head --temp-dir="$RAY_TEMP_DIR"
sleep 5
run_verl
