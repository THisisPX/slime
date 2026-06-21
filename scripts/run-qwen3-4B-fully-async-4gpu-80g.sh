#!/bin/bash
# ==============================================================================
# Qwen3-4B Fully Async 训练 — 4× A100 80GB (分卡模式, 对齐 verl fully_async)
#
# GPU 分配: 训练 2 卡 (TP2, DP1) + 推理 2 卡 (TP2, 1 engine) = 共 4 卡
# 使用 train_async.py + fully_async_rollout (持久后台 worker 持续生成)
#
# 对齐 verl: examples/grpo_trainer/run_qwen3_4b_megatron_perf_test_async_80g.sh
#   - 4 GPU split: 2 train + 2 infer
#   - TP2 训练, TP2 推理
#   - n=16, max_resp=8192, max_prompt=1024
#   - PPO_MAX_TOKEN_LEN=16384 (80G 充裕)
#   - LR=1e-6 constant, Adam betas=[0.9,0.98], wd=0.1
#   - clip [0.2, 0.28], no KL loss, no entropy
#   - full recompute uniform
#   - TOTAL_ROLLOUT_STEPS=128 → slime num-rollout=16 (× rollout_batch_size=8)
#   - 不做 eval/save (性能测试)
#
# slime vs verl 未对齐差异:
#   - 推理引擎: SGLang (CUDA graph ON) vs vLLM (enforce_eager=False)
#   - reward: deepscaler vs math_dapo (不影响性能)
#   - loss_agg: sum-of-sample-mean vs token-mean (不影响性能)
#   - async 机制: 持久 worker + train_async vs fully_async streaming
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
# =====================================================================

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# fully_async rollout 需要 PYTHONPATH 包含此目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
FULLY_ASYNC_DIR="${SCRIPT_DIR}/../examples/fully_async"

source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${TORCH_DIST_CKPT}"
)

ROLLOUT_ARGS=(
   # fully_async 核心: 持久后台 worker 持续生成
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --loss-mask-type qwen3

   --rm-type deepscaler

   # 对齐 verl: TOTAL_ROLLOUT_STEPS=128
   # slime: num_rollout × rollout_batch_size = 16 × 8 = 128 prompts
   --num-rollout 16
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --rollout-system-prompt "Please reason step by step, and put your final answer in \boxed{}."

   # 必须够大: DP=1 时 128 才能有 8 groups/step
   # 32 只能 2 groups/step, 42% 步全 zero_std → loss≈0
   --global-batch-size 128
   --balance-data
)

PERF_ARGS=(
   # 2 卡训练: TP2 × DP1 (对齐 verl: TRAIN_TP=2)
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
   # 对齐 verl: PPO_MAX_TOKEN_LEN_PER_GPU=16384 (80G 充裕)
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   # 对齐 verl: grpo, no KL, no entropy
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   # 对齐 verl: clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
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
   # 2 卡推理: 1 engine × TP2 (对齐 verl: ROLLOUT_TP=2)
   --rollout-num-gpus-per-engine 2
   # 对齐 verl: ROLLOUT_GPU_MEM_UTIL=0.7 (80G 充裕)
   --sglang-mem-fraction-static 0.7
   # 降低并发避免 KV cache OOM 和长时间运行崩溃
   --sglang-max-running-requests 8
   --sglang-cuda-graph-max-bs 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tb-project-name slime-vs-verl-async-80g

   # fully_async 长时间运行: 放宽健康检查避免误杀引擎
   --rollout-health-check-interval 60
   --rollout-health-check-timeout 120
)

# ==================== 启动 Ray + 提交任务 ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 日志目录 (对齐 verl 的 log 目录结构)
LOG_DIR="${LOG_DIR:-logs/slime-vs-verl-async-80g}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${FULLY_ASYNC_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
