#!/bin/bash
# ==============================================================================
# Qwen3-4B 异步训练 — 4× A100 80GB (分卡模式)
#
# GPU 分配: 训练 2 卡 (TP2) + 推理 2 卡 (TP2) = 共 4 卡
# 使用 train_async.py (rollout 与 train 流水线重叠执行)
#
# 与同步训练的关键区别:
#   - 下一轮 rollout 与当前轮训练并行执行
#   - 必须使用分卡模式 (不支持 colocate)
#   - --update-weights-interval 控制权重同步频率
#
# 注意: Qwen3 使用标准 Megatron GPT 架构，不需要 --spec
#       不使用 --use-tis / --calculate-per-token-loss (GLM 专属)
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
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --loss-mask-type qwen3

   --rm-type deepscaler

   --num-rollout 8
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --rollout-system-prompt "Please reason step by step, and put your final answer in \boxed{}."

   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 100
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   # 2 卡训练: TP2 × CP1
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

WANDB_ARGS=(
   #--use-wandb
   #--wandb-project slime-qwen3-4B-async
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # 2 卡推理: 1 个引擎 × TP2
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tb-project-name qwen3-4b-async-4gpu
)

# ==================== 启动 Ray + 提交任务 ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
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
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
