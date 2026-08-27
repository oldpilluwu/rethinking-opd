#!/bin/bash
# =============================================================================
# Standard (sampled-token) On-Policy Distillation
#   student : Qwen3-1.7B-SFT      teacher : Qwen3-4B (non-thinking)
#   hardware: 1 x A100 80GB       framework: this repo's verl 0.7.0.dev fork
#
# Recipe follows Lightning OPD (arXiv 2604.13010) Table 6 for the OPD stage,
# with lr held at 1e-6 (this repo's value for the 1.7B scale) rather than 2e-6.
#
# "Standard OPD" == LOG_PROB_TOP_K=0: the per-token reward is the negative
# reverse KL on the *sampled* token, applied via policy gradient. Setting
# LOG_PROB_TOP_K>0 would switch to top-k OPD, a different algorithm.
# Diagnostics come from DIAGNOSTIC_TOP_K, which does not touch the reward.
#
# A6000 48GB alternatives are marked  # [48GB]  throughout.
# =============================================================================

set -x

# ---------------------------------------------------------------- environment
ray stop --force
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
# NOTE: CUDA_LAUNCH_BLOCKING is deliberately NOT set. The upstream script sets it
# to 1, which serializes every kernel launch and costs real throughput.

# --- SwanLab run identity -----------------------------------------------------
# Set BOTH of these before a resume, or the diagnostic curves split into two
# disconnected traces at the resume boundary. Record the ID somewhere durable.
export PROJECT_NAME='OPD-Lightning-Repro'
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="<paste-run-id-here-when-resuming>"

# ------------------------------------------------------------------- models
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/Qwen3-1.7B-SFT}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-model/Qwen3-4B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

# --------------------------------------------------------------------- data
# Lightning OPD uses DAPO-Math-17k for the math domain. 150 steps x 64 prompts
# consumes 9,600 of 17,917 rows, so shuffle must be ON or you would only ever
# see the first half of the file in write order.
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
export TEST_DATA_DIR=datasets/test_data
TEST_DATASET="['$TEST_DATA_DIR/AIME24/test.parquet']"   # unused; test_freq=-1

# ---------------------------------------------------------------- recipe
export ADV_ESTIMATOR=token_reward_direct   # do not change for OPD
export ADV_CLIP_RANGE=10.0                 # Lightning Table 6: advantage clip [-10, 10]

export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=4096                # Lightning Table 6
export MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1 ))

export TRAIN_BATCH_SIZE=64                 # 64 prompts x n=4 = 256 sequences = Lightning's global batch
export MINI_BATCH_SIZE=64                  # == train batch -> exactly one update per rollout, strictly on-policy
export N_RESPONSES=4

# DELIBERATE DEVIATION FROM LIGHTNING TABLE 6 (which specifies 0.8).
# This fork computes the student term of the reward from temperature-scaled logits
# (dp_actor.py: logits_rmpad.div_(temperature)), while the teacher is scaled by the
# separate teacher_temperature. At T != 1.0 the reward therefore compares a SHARPENED
# student against an unsharpened teacher, which is not the reverse KL it claims to be.
# Measured on synthetic logits at T=0.8: bias +0.24 nats (~82% of the intended signal)
# and ~25% of tokens get their advantage sign flipped.
# T=1.0 makes div_(1.0) a no-op, so both sides sit at temperature 1.0 and the reward is
# exactly teacher_logp - student_logp. It is also the value the Rethinking-OPD paper
# (Table 2) used with this same 1.7B/4B pair.
# To use 0.8 faithfully you would need the reward to read T=1.0 student log-probs while
# the importance ratio keeps the scaled ones -- a code change, not a config change.
export TEMPERATURE=1.0
export TOP_P=1.0
export TEACHER_TEMPERATURE=1.0

export LR=1e-6                             # 1e-6, not Lightning's 2e-6 (their students are 4B/8B)
export WEIGHT_DECAY=0.1
export ADAM_BETAS="[0.9,0.98]"
export TOTAL_STEPS=150

export LOG_PROB_TOP_K=0                    # 0 = standard / sampled-token OPD
export DIAGNOSTIC_TOP_K=16                 # logging only; does not enter the loss
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export LOSS_AGG_MODE=token-mean
export MODEL_DTYPE=fp32                    # actor MASTER weights. FSDP already computes in
                                           # bf16; bf16 masters would underflow at lr 1e-6.

# ------------------------------------------------------- memory / throughput
export N_GPUS=1
export PARALLEL_SIZE=1
export ACTOR_MAX_TOKEN_LEN=16384           # [48GB] 8192
export TEACHER_MAX_TOKEN_LEN=16384         # [48GB] 8192  (see critic.* note below)
export GPU_MEM_UTIL=0.35                   # [48GB] 0.5
export PARAM_OFFLOAD=False                 # [48GB] True
export OPTIMIZER_OFFLOAD=False             # [48GB] True if OOM

# --------------------------------------------------------------- checkpoints
export SAVE_STEPS="[1,2,3,4,5,10,15,20,25,30,50,75,100,125,150]"
export OPTIMIZER_SAVE_STEPS="[50,150]"     # only these two are resumable

# ------------------------------------------------------------------ smoke mode
# SMOKE=1 runs a few cheap steps to prove the pipeline before committing 5-6 hours.
# Shrinks length/batch/steps and writes to its own checkpoint directory so it can
# never collide with a real run. Everything else -- the objective, the reward path,
# the diagnostics -- is identical, which is the point: a shape mismatch in the
# diagnostics is fatal on the first batch, so this is where it surfaces.
if [ "${SMOKE:-0}" = "1" ]; then
    export MAX_RESP_LENGTH=1024
    export MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1 ))
    export TRAIN_BATCH_SIZE=8
    export MINI_BATCH_SIZE=8
    export TOTAL_STEPS=3
    export SAVE_STEPS="[3]"
    export OPTIMIZER_SAVE_STEPS="[3]"
    export ACTOR_MAX_TOKEN_LEN=4096
    export TEACHER_MAX_TOKEN_LEN=4096
    echo "=== SMOKE MODE: 3 steps, batch 8, 1024 tokens ==="
fi

export PROJECT_PATH=checkpoint
export EXPERIMENT_NAME=${SMOKE:+smoke_}stdopd_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_len${MAX_RESP_LENGTH}-T${TEMPERATURE}-n${N_RESPONSES}-bs${TRAIN_BATCH_SIZE}-lr${LR}-clip${ADV_CLIP_RANGE}
export CKPT_PATH=${PROJECT_PATH}/${EXPERIMENT_NAME}
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log

# ------------------------------------------------------------------- resume
# Only global_step_50 and global_step_150 carry optimizer state. resume_mode=auto
# would follow the tracker file to whichever step was written last -- usually a
# model-only checkpoint -- and fail on the missing optimizer shard. Always name
# the path explicitly.
#
#   RESUME_FROM=${CKPT_PATH}/global_step_50 bash opd_lightning_a100.sh
#
# Rollout sampling is not seeded in this fork, so a resumed run diverges from the
# original after the resume point. Delete any checkpoints above the resume step
# before restarting, or your benchmark set will mix two trajectories.
RESUME_ARGS="trainer.resume_mode=disable"
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_FROM"
    if [ ! -f "$RESUME_FROM/manifest.json" ]; then
        echo "WARNING: no manifest.json in $RESUME_FROM (written by runs after this patch)."
    elif ! grep -q '"resumable": true' "$RESUME_FROM/manifest.json"; then
        echo "ERROR: $RESUME_FROM is marked NOT resumable in its manifest."
        echo "Only steps in OPTIMIZER_SAVE_STEPS carry optimizer state."
        exit 1
    fi
elif [ -d "$CKPT_PATH" ] && [ -n "$(ls -A "$CKPT_PATH" 2>/dev/null)" ]; then
    # The checkpoint manager creates directories but never clears them. Writing a
    # model-only step into a directory that already holds optim_*.pt from a previous
    # run leaves a checkpoint that looks resumable and is not.
    echo "ERROR: checkpoint directory is not empty:"
    echo "  $CKPT_PATH"
    echo
    echo "Use a fresh directory, or set RESUME_FROM=<...>/global_step_50 to resume."
    echo "To reuse this path anyway: FORCE_DIRTY_CKPT=1 bash $0"
    [ "${FORCE_DIRTY_CKPT:-0}" = "1" ] || exit 1
    echo "FORCE_DIRTY_CKPT=1 set -- continuing into a non-empty directory."
fi

mkdir -p logs
LOG_FILE="logs/opd_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "log: $LOG_FILE"

ray start --head
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.adv_clip_range=$ADV_CLIP_RANGE \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.shuffle=True \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.betas="$ADAM_BETAS" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    actor_rollout_ref.actor.optim.eps=1e-8 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']" \
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ACTOR_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.diagnostic_top_k=$DIAGNOSTIC_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    critic.ppo_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    +reward_model.model.dtype=bf16 \
    reward_model.micro_batch_size_per_gpu=1 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_steps="$SAVE_STEPS" \
    trainer.optimizer_save_steps="$OPTIMIZER_SAVE_STEPS" \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.del_local_ckpt_after_load=False \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=False \
    trainer.opd_text_diagnostics=True \
    $RESUME_ARGS

# ---------------------------------------------------------------------------
# NOTES
#
# critic.ppo_max_token_len_per_gpu
#   The teacher's forward budget resolves through
#     reward_model.forward_max_token_len_per_gpu
#       -> critic.forward_max_token_len_per_gpu
#       -> critic.ppo_max_token_len_per_gpu   (default 32768)
#   It does NOT inherit the actor's value, and because
#   reward_model.use_dynamic_bsz resolves to the actor's (True),
#   reward_model.micro_batch_size_per_gpu is never consulted. There is no critic
#   in this run; the key is set purely to bound the teacher's logits tensor.
#   At 32768 tokens x 151,936 vocab that tensor is ~10GB on its own.
#
# reward_model.model.fsdp_config.param_offload
#   Dead config. RewardModelWorker hard-codes CPUOffload(offload_params=True), so
#   the teacher always streams params from CPU. On 80GB this is pure overhead --
#   removing that line in fsdp_workers.py is worth ~30-60 min over the run.
#
# validation
#   verl 0.7.0 under-reports accuracy by 5-7 points, hence test_freq=-1.
#   Benchmark the checkpoints afterwards with scripts/val/eval/.
# ---------------------------------------------------------------------------
