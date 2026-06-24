#!/bin/bash
# ==============================================================================
# Qwen3-4B Async 训练 — 8× A100 40GB (分卡模式, 对齐 verl fully_async)
#
# GPU 分配: 训练 4 卡 (TP2, DP2) + 推理 4 卡 (2 engines × TP2) = 共 8 卡
# 使用 train_async.py (rollout 与 train pipeline 重叠)
#
# 对齐 verl: examples/grpo_trainer/run_qwen3_4b_megatron_perf_test_async_80g.sh
# 适配 40GB: max_tokens 4096, sglang_mem 0.55
#
# 与 4GPU 版本相比的优势:
#   - DP=2 开启优化器分片，每卡显存 ~28GB (vs DP=1 ~47GB)
#   - 2 个 rollout engine 并行推理，生成吞吐翻倍
#   - 每个 rollout 128 样本分流到 2 引擎 → 不崩
#   - 8 groups/step (DP=2 × global_bs=128/16=8)
#
# slime vs verl pipeline 差异:
#   slime train_async: 下一轮 rollout 与当前训练重叠 (buffer=1)
#   verl fully_async:  流式持续生成, 训练异步取结果 (streaming)
#   两者都实现了 rollout↔train 重叠, 差异在粒度和框架实现
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
PROMPT_DATA="${PROMPT_DATA:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
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
)

# ============================================================================
# 训练/数据参数 — 逐项对齐 verl
# ============================================================================

# 对齐 verl: TOTAL_ROLLOUT_STEPS=128
# slime: num_rollout × rollout_batch_size = 16 × 8 = 128 prompts
# 8GPU 有 2 引擎, rollout_batch_size=8 不会压崩引擎 (64 样本/引擎)
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --loss-mask-type qwen3

   --rm-type deepscaler

   --num-rollout 16
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --rollout-system-prompt "Please reason step by step, and put your final answer in \boxed{}."

   # DP=2 (4训练卡): global_bs=128 → 64/DP → 4 groups/DP
   # 8 groups total, zero_std 概率 ~3.2%
   --global-batch-size 128
   --balance-data
)

# 对齐 verl: TP2, PP1, DP=2 (4 GPUs train, 40G)
# 对齐 verl: recompute full uniform, sequence_parallel
# max_tokens: 4096 (40G 限制, 8GPU 之前验证可跑)
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

# 对齐 verl: grpo, no KL, no entropy, clip [0.2, 0.28, 10.0]
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
)

# 对齐 verl: Adam, lr=1e-6, constant, wd=0.1, betas=[0.9,0.98]
OPTIMIZER_ARGS=(
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

# 2 engines × TP2, mem_fraction 0.55 (40G: 22GB KV cache)
# verl gpu_mem_util=0.7 但 vLLM 内存效率更高
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.55
   --sglang-max-running-requests 16
   --sglang-cuda-graph-max-bs 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tb-project-name slime-vs-verl-async-8gpu
)

# ==================== 启动 Ray + 提交任务 ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 日志目录
LOG_DIR="${LOG_DIR:-logs/slime-vs-verl-async-8gpu}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
