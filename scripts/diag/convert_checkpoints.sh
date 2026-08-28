#!/bin/bash
# Merge every saved OPD checkpoint into HF format for benchmarking.
#
#   bash scripts/diag/convert_checkpoints.sh checkpoint/<experiment> hf_ckpts
#
# Output goes to <out_dir>/global_step_N/, loadable directly by vLLM. Models are
# written in bf16: eval is done in bf16 anyway, and it halves the disk against the
# fp32 training masters (~3.4GB vs ~6.9GB per checkpoint at 1.7B).
#
# Note the script name -- this fork ships legacy_model_merger.py; there is no
# model_merger.py.
set -euo pipefail

CKPT_DIR=${1:?usage: convert_checkpoints.sh <ckpt_dir> [out_dir] [steps]}
OUT_DIR=${2:-hf_ckpts}
# Optional comma-separated step filter, e.g. "25,50". Converting a checkpoint costs
# ~2 min and ~3.4GB, so there is no reason to convert ones you will not benchmark.
STEPS=${3:-}
MERGER=verl/scripts/legacy_model_merger.py

mkdir -p "$OUT_DIR"

for step_dir in "$CKPT_DIR"/global_step_*; do
    [ -d "$step_dir/actor" ] || continue
    step=$(basename "$step_dir")
    if [ -n "$STEPS" ]; then
        n=${step#global_step_}
        case ",$STEPS," in *",$n,"*) ;; *) continue ;; esac
    fi
    target="$OUT_DIR/$step"

    if [ -f "$target/config.json" ]; then
        echo "skip $step (already merged)"
        continue
    fi

    echo "merging $step ..."
    python "$MERGER" merge \
        --backend fsdp \
        --local_dir "$step_dir/actor" \
        --target_dir "$target"

    # Down-cast the merged weights to bf16 in place.
    python - "$target" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = sys.argv[1]
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, trust_remote_code=True)
m.save_pretrained(p, safe_serialization=True)
try:
    AutoTokenizer.from_pretrained(p, trust_remote_code=True).save_pretrained(p)
except Exception as e:
    print(f"  tokenizer not re-saved: {e}")
PY
done

echo
echo "merged checkpoints in $OUT_DIR:"
ls -1 "$OUT_DIR"
