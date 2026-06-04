#!/bin/bash
# ==============================================================================
# GLM-Z1-9B-0414 异步训练 — 4× A100 80GB (分卡模式)
#
# GPU 分配: 训练 2 卡 (TP2) + 推理 2 卡 (TP2) = 共 4 卡
# 使用 train_async.py (rollout 与 train 流水线重叠执行)
#
# 与同步训练的关键区别:
#   - 下一轮 rollout 与当前轮训练并行执行
#   - 必须使用分卡模式 (不支持 colocate)
#   - --update-weights-interval 控制权重同步频率
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
HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/volume/pengxiong/models/GLM-Z1-9B-0414}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-/workspace/volume/pengxiong/models/GLM-Z1-9B-0414_torch_dist}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/GLM-Z1-9B-0414_slime}"
PROMPT_DATA="${PROMPT_DATA:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.jsonl}"
# =====================================================================

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

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

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 100
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 4096
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
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   --calculate-per-token-loss
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
   #--wandb-project slime-glm4-9B-async
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # 2 卡推理: 1 个引擎 × TP2
   --rollout-num-gpus-per-engine 2
   --sglang-cuda-graph-max-bs 16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ==================== 启动 Ray + 提交任务 ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
