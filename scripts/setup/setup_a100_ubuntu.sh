#!/usr/bin/env bash
# Build the repository's CUDA environment on Ubuntu without Conda.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: this setup script expects Ubuntu/Debian (apt-get not found)." >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y \
    build-essential curl git git-lfs jq tmux wget \
    python3-dev libgl1 libglib2.0-0
git lfs install

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12
uv venv --python 3.12 --seed .venv
# shellcheck disable=SC1091
source .venv/bin/activate

export PIP_CONSTRAINT="$ROOT_DIR/constraints/a100-cu12.txt"
python -m pip install --upgrade pip setuptools wheel

# The vendored installer pins vLLM/torch/FlashAttention. The constraint file
# keeps its transitive scientific stack internally compatible.
(
    cd verl
    USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
)

python -m pip install --no-deps -e ./verl
python -m pip install swanlab math-verify latex2sympy2-extended
python -m pip check
python scripts/setup/check_a100_env.py

echo
echo "Environment ready. Activate it with: source .venv/bin/activate"
