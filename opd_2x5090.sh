#!/usr/bin/env bash
# Launch the paper OPD recipe on one host with two NVIDIA RTX 5090 GPUs.
#
# Usage:
#   CONFIG_ONLY=1 bash opd_2x5090.sh
#   bash opd_2x5090.sh
#   bash opd_2x5090.sh configs/opd/my_2gpu_experiment.toml
#
# CUDA_VISIBLE_DEVICES may select a different pair, for example:
#   CUDA_VISIBLE_DEVICES=2,3 bash opd_2x5090.sh

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

DEFAULT_CONFIG="configs/opd/paper_qwen3_1p7b_rl_math_teacher_2x5090.toml"
export OPD_CONFIG=${OPD_CONFIG:-$DEFAULT_CONFIG}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

CONFIG_PATH=$OPD_CONFIG
if [ "${1:-}" = "--config" ]; then
    if [ "$#" -lt 2 ]; then
        echo "ERROR: --config requires a TOML path" >&2
        exit 2
    fi
    CONFIG_PATH=$2
elif [[ "${1:-}" == *.toml ]]; then
    CONFIG_PATH=$1
fi

PREFLIGHT_ARGS=(--config "$CONFIG_PATH")
if [ "${CONFIG_ONLY:-0}" = "1" ]; then
    PREFLIGHT_ARGS+=(--skip-hardware)
fi
python3 scripts/setup/check_2x5090.py "${PREFLIGHT_ARGS[@]}"

# The existing launcher is config-driven and supports any NVIDIA GPU topology;
# this wrapper supplies the 5090 profile and performs the hardware checks above.
exec bash "$ROOT_DIR/opd_lightning_a100.sh" "$@"
