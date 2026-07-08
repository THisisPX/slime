#!/bin/bash
# ==============================================================================
# Qwen3-VL-4B-Instruct GEO3K RL 训练 — 4× GPU (分卡模式) — v3 续训
#
# 从 v3 checkpoint (iter_0000099) 继续训练
# 续训逻辑:
#   --load + --ckpt-step 99: 加载 Megatron checkpoint
#   --start-rollout-id 100: 从第 100 步开始 (0-99 已完成)
#   其他参数与 v3 完全一致
#
# GPU 分配: 训练 2 卡 (TP2, DP1) + 推理 2 卡 (TP2, 1 engine) = 共 4 卡
# 使用 train.py (同步 rollout → train → update_weights 循环)
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
pkill -9 redis 2>/dev/null

set -ex
export PYTHONBUFFERED=16

# ==================== 路径配置 (请根据实际环境修改) ====================
HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/Qwen3-VL-4B_slime_geo3k_v3}"
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-/workspace/volume/pengxiong/datasets}"
DATASET_NAME="${DATASET_NAME:-chenhegu/geo3k_imgurl}"
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")
# =====================================================================

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# 下载数据集
mkdir -p "${PROMPT_DATA_DIR}"
if [ ! -d "${PROMPT_DATA_DIR}/${DATASET_LOCAL_NAME}" ]; then
   hf download --repo-type dataset "${DATASET_NAME}" --local-dir "${PROMPT_DATA_DIR}/${DATASET_LOCAL_NAME}"
fi

# ==================== 模型参数 ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# ==================== 参数组装 ====================

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   # 续训: 从 Megatron checkpoint 加载
   --load "${SAVE_DIR}"
   --ckpt-step 99
   --save "${SAVE_DIR}"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_DIR}/${DATASET_LOCAL_NAME}/train.parquet"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --loss-mask-type qwen3

   --rm-type math

   # rollout-batch-size 16 × n-samples 8 = 128 样本/rollout
   # 续训: 从 rollout 100 开始, 总共 500 步 (0-99 已跑)
   --num-rollout 500
   --start-rollout-id 100
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 3072
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
)

# VLM 必需: 告知数据管线图片数据列
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data "${DATASET_LOCAL_NAME}" "${PROMPT_DATA_DIR}/${DATASET_LOCAL_NAME}/test.parquet"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-p 1
)

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
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# 仅使用 TensorBoard，不使用 Wandb

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-mm-attention-backend sdpa
   --sglang-disable-cuda-graph
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --attn-implementation flash_attention_2
   --megatron-to-hf-mode bridge
   --use-tensorboard
   --tb-project-name qwen3-vl-4b-geo3k-4gpu-v3-resume
)

# ==================== 启动 Ray + 提交任务 ====================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head \
   --node-ip-address ${MASTER_ADDR} \
   --num-gpus 4 \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TRITON_PTXAS_PATH\": \"/usr/local/cuda/bin/ptxas\",
    \"TORCH_CUDA_ARCH_LIST\": \"10.0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# 日志目录
LOG_DIR="${LOG_DIR:-logs/qwen3-vl-4b-geo3k-4gpu-v3-resume}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
