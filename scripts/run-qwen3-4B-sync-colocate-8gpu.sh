#!/bin/bash
# ==============================================================================
# Qwen3-4B 同步训练 — 8× A100 40GB colocate 模式 (对齐 verl 配置)
#
# GPU 分配: 8 卡 colocate (训练+推理共享, offload 切换)
# 训练: TP2 × DP4 (=8卡)
# 推理: 4 引擎 × TP2
#
# 与 verl run_qwen3_4b_megatron_perf_test.sh 对齐:
#   - TP2 (verl: ACTOR_TP=2)
#   - n_samples=16 (verl: ROLLOUT_N=16)
#   - 8 rollout × 8 batch × 16 samples = 128 样本/步
#   - loss_agg: slime 用 sum-of-sample-mean, verl 用 token-mean (数值不同, 梯度方向一致)
#
# slime vs verl 关键差异 (预期影响对比结果):
#   - 推理引擎: SGLang vs vLLM
#   - 推理 CUDA graph: slime 开启, verl enforce_eager=True 关闭
#   - 并行度: slime max_running_requests ≈ 无限制, verl max_num_seqs=32
#   - max_tokens_per_gpu: slime=4096 vs verl=12288 (A100 40G 限制)
# ==============================================================================

# 清理残留进程
pkill -9 sglang 2>/dev/null
sleep 2
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 2
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null

set -ex
export PYTHONBUFFERED=16

# ==================== 路径配置 (请根据实际环境修改) ====================
HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-/workspace/volume/pengxiong/models/qwen3-4B-torch}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/Qwen3-4B_slime}"
PROMPT_DATA="${PROMPT_DATA:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.jsonl}"
# =====================================================================

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${TORCH_DIST_CKPT}"
   --save "${SAVE_DIR}"
   --save-interval 9999        # 对齐 verl: 不做 checkpoint
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --loss-mask-type qwen3

   --rm-type deepscaler

   # 对齐 verl: train_batch_size=8, rollout.n=16
   --num-rollout 8
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --rollout-system-prompt "Please reason step by step, and put your final answer in \boxed{}."

   # 对齐 verl: PPO_MINI_BATCH_SIZE=8
   --global-batch-size 32
   --balance-data

   # 对齐 verl: 不做 eval/val
   --eval-interval 9999
)

# 保留 eval 配置但不触发 (对齐 verl test_freq=9999)
EVAL_ARGS=(
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   # 对齐 verl: TP2, PP1 → DP=4 (8卡)
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # 对齐 verl: full recompute, uniform, 1 layer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # 对齐 verl: PPO_MAX_TOKEN_LEN_PER_GPU=12288
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   # 对齐 verl: grpo, no KL, no entropy
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   # 对齐 verl: clip_ratio_low=0.2, clip_ratio_high=0.28
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   # 对齐 verl: Adam, lr=1e-6, constant, wd=0.1, betas=[0.9,0.98]
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   #--wandb-project slime-vs-verl
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # 4 引擎 × TP2 (=8卡 inference)
   --rollout-num-gpus-per-engine 2
   # A100 40G: 推理和训练共享, 限制推理显存
   --sglang-mem-fraction-static 0.4
   # 对齐 verl: max_num_seqs=32
   --sglang-max-running-requests 32
   # slime 默认开 CUDA graph (verl enforce_eager=True 关闭)
   --sglang-cuda-graph-max-bs 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tb-project-name slime-vs-verl-colocate-8gpu
)

# ==================== 启动 Ray + 提交任务 ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
