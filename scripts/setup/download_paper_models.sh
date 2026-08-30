#!/usr/bin/env bash
# Download immutable model revisions used by the checked-in paper config.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-Qwen/Qwen3-1.7B}
STUDENT_REVISION=${STUDENT_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}

# The author's G-OPD command names this teacher but does not publish a Hub URL.
# This is the currently available full-precision Step500 upload. Override both
# variables if you have the author's original checkpoint.
TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
TEACHER_REVISION=${TEACHER_REVISION:-05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa}

command -v hf >/dev/null 2>&1 || {
    echo "ERROR: hf CLI is unavailable; activate .venv after running setup_a100_ubuntu.sh" >&2
    exit 1
}

mkdir -p model
hf download "$STUDENT_MODEL_ID" \
    --revision "$STUDENT_REVISION" \
    --local-dir model/Qwen3-1.7B
hf download "$TEACHER_MODEL_ID" \
    --revision "$TEACHER_REVISION" \
    --local-dir model/Qwen3-4B-Non-Thinking-RL-Math

printf '%s\n' \
    "student=$STUDENT_MODEL_ID@$STUDENT_REVISION" \
    "teacher=$TEACHER_MODEL_ID@$TEACHER_REVISION" \
    > model/paper-model-revisions.txt

echo "Downloaded immutable model revisions; provenance is in model/paper-model-revisions.txt"
