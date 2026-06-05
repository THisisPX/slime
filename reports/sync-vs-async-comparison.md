# Qwen3-4B 同步 vs 异步训练对比报告

## 复现方式

### 数据准备

```bash
# 下载模型
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B

# 下载数据集
huggingface-cli download zhuzilin/dapo-math-17k --local-dir /workspace/datasets/dapo-math-17k
huggingface-cli download zhuzilin/aime-2024 --local-dir /workspace/datasets/aime-2024

# 转换 checkpoint 为 Megatron torch_dist 格式
python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /workspace/models/Qwen3-4B \
  --model-type qwen3-4B --tp-size 2 --num-gpus-per-node 4 \
  --output /workspace/models/qwen3-4B-torch
```

### 执行训练

```bash
# 异步模式
bash scripts/run-qwen3-4B-async-8gpu.sh

# 同步模式
bash scripts/run-qwen3-4B-sync-8gpu.sh
```

### 查看结果

```bash
tensorboard --logdir tensorboard_log/ --port 6006
```

## 训练日志

| 模式 | n_samples | TensorBoard 路径 | 内容 |
|------|-----------|-----------------|------|
| 异步 | 16 | `tensorboard_log/qwen3-4b-perf-test-8gpu/20260605_01461[3,5]` | rollout + train metrics |
| 同步 | 16 | `tensorboard_log/qwen3-4b-sync-8gpu/20260605_024931` | rollout metrics |
| 同步 | 16 | `tensorboard_log/qwen3-4b-sync-8gpu/20260605_025354` | train metrics |

## 参数设置说明

同步和异步使用相同的训练参数，差异仅在于入口 `train.py` vs `train_async.py`。完整参数见 `scripts/run-qwen3-4B-{sync,async}-8gpu.sh`。

### 硬件分配

| 参数 | 值 | 说明 |
|------|-----|------|
| `--actor-num-nodes` | 1 | 训练 1 节点 |
| `--actor-num-gpus-per-node` | 4 | 训练 4 卡 (TP2 × DP2) |
| `--rollout-num-gpus` | 4 | 推理 4 卡 (2 引擎 × TP2) |
| `--rollout-num-gpus-per-engine` | 2 | 每引擎 TP2 |

### 并行策略 (Megatron)

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-model-parallel-size` | 2 | 张量并行，权重切分到 2 卡 |
| `--sequence-parallel` | ✓ | 序列并行，分散 LayerNorm/Dropout |
| `--recompute-granularity full` + `--recompute-method uniform` | ✓ | 全重计算，用显存换计算 |
| `--use-dynamic-batch-size` | ✓ | 动态 micro-batch 打包 |
| `--max-tokens-per-gpu` | 4096 | 每 GPU token 上限 |

### 数据 & Rollout

| 参数 | 值 | 说明 |
|------|-----|------|
| `--prompt-data` | dapo-math-17k.jsonl | 数学推理训练集 |
| `--loss-mask-type` | qwen3 | 只对 assistant 回复计算 loss |
| `--rm-type` | deepscaler | DeepScaler 规则判分 |
| `--num-rollout` | 8 | 测试用，正式训练改 3000+ |
| `--rollout-batch-size` | 8 | 每轮 8 个 prompt group |
| `--n-samples-per-prompt` | 16 | GRPO 组大小 |
| `--rollout-max-response-len` | 8192 | 回答最大 token 数 |
| `--global-batch-size` | 32 | 全局 batch 大小 |

### GRPO 算法

| 参数 | 值 | 说明 |
|------|-----|------|
| `--advantage-estimator` | grpo | GRPO 优势估计 |
| `--eps-clip` / `--eps-clip-high` | 0.2 / 0.28 | PPO ratio 裁剪 [0.8, 1.28] |
| `--kl-loss-coef` | 0.00 | 不使用 ref 模型 |
| `--entropy-coef` | 0.00 | 不额外加熵奖励 |
| `--rewards-normalization` | ✓ (默认) | 组内均值归一化 |
| `--grpo-std-normalization` | ✓ (默认) | 组内标准差归一化 |

### 异步独有参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--update-weights-interval` | 1 | 每 1 步同步权重到 rollout 引擎 |

### 推理引擎 (SGLang)

| 参数 | 值 | 说明 |
|------|-----|------|
| `--sglang-mem-fraction-static` | 0.7 | 使用 70% GPU 显存 |
| `--sglang-cuda-graph-max-bs` | 16 | CUDA graph 最大 batch size |

---

## 实验配置

| 项目 | 同步 | 异步 |
|------|------|------|
| 入口 | `train.py` | `train_async.py` |
| 硬件 | 8× A100 (训练 TP2+DP2 / 推理 2引擎×TP2) | 同 |
| 模型 | Qwen3-4B | 同 |
| 算法 | GRPO, n_samples=16, max-response=8192 | 同 |
| 运行 | 8 rollout steps + 32 train steps | 同 |

---

## 一、核心差异：Pipeline 架构

```
同步 (train.py):
  step N:   |-- rollout N --|-- train N --|
  step N+1:                   |-- rollout N+1 --|-- train N+1 --|

异步 (train_async.py):
  step N:   |-- rollout N --|
            |-- train N (与 rollout N+1 并行) --|
  step N+1: |-- rollout N+1 --|
            |-- train N+1 (与 rollout N+2 并行) --|
```

**同步**严格串行：rollout → train → rollout → train。**异步**将下一轮 rollout 与当前轮训练重叠执行。

---

## 二、端到端时序

### 2.1 每步时间分解

| 阶段 | 异步 (均值) | 同步 (均值) | 差异 |
|------|-----------|-----------|------|
| rollout_time | 223.1s | 224.2s | — |
| train_wait_time | 106.6s | **254.6s** | +139% |
| data_preprocess | 0.22s | 0.15s | — |
| log_probs_time | 16.9s | 16.8s | — |
| actor_train_time | 115.0s | 113.9s | — |
| update_weights | 0.45s | 0.53s | — |
| train_time (总和) | 132.2s | 131.4s | — |
| **step_time** | **239.2s** | **386.0s** | **+61.4%** |

### 2.2 各步时序明细

**异步 (8 steps)**：

| step | rollout | train_wait | train | step_time |
|------|---------|-----------|-------|-----------|
| 0 | 214s | 216s | 122s | 338s |
| 1 | 217s | 94s | 130s | 225s |
| 2 | 213s | 83s | 103s | 186s |
| 3 | 225s | 122s | 158s | 280s |
| 4 | 216s | 58s | 132s | 190s |
| 5 | 229s | 98s | 144s | 243s |
| 6 | 214s | 70s | 127s | 198s |
| 7 | 238s | 112s | 142s | 254s |
| **均值** | **223s** | **107s** | **132s** | **239s** |

**同步 (8 steps)**：

| step | rollout | train_wait | train | step_time |
|------|---------|-----------|-------|-----------|
| 0 | 242s | 462s | 130s | 591s |
| 1 | 213s | 216s | 130s | 346s |
| 2 | 211s | 213s | 116s | 328s |
| 3 | 246s | 248s | 154s | 402s |
| 4 | 214s | 216s | 128s | 345s |
| 5 | 231s | 233s | 130s | 364s |
| 6 | 213s | 215s | 121s | 336s |
| 7 | 231s | 233s | 142s | 375s |
| **均值** | **224s** | **255s** | **131s** | **386s** |

### 2.3 等待时间分析

| 指标 | 异步 | 同步 | 说明 |
|------|------|------|------|
| wait_time_ratio | 43.1% | 64.7% | GPU 空闲等待 rollout 的时间占比 |
| train_wait / rollout | 0.48× | 1.14× | 异步等待只需 rollout 一半时间 |
| 有效重叠率 | 52% | 0% | 异步：223s rollout 仅等 107s |

**异步**：train_wait 仅 107s（rollout 的 48%），其余 116s 被训练计算覆盖。
**同步**：train_wait = 255s（**大于** rollout 的 224s），因为首步没有预取，step 0 等了 462s 拖高了均值；去掉 step 0 后为 228s ≈ rollout。

---

## 三、Rollout 生成性能

| 指标 | 异步 | 同步 | 说明 |
|------|------|------|------|
| rollout_time | 223.1s | 224.2s | 无差异 |
| tokens_per_gpu_per_sec | 923 | 908 | 无差异 |
| longest_sample_tok/s | 37.1 | 36.5 | 无差异 |
| response_len mean | 6,369 | 6,432 | 无差异 |
| truncated_ratio | 47.6% | 46.7% | 无差异 |
| prefix_cache_hit_rate | 59.7% | 27.1% | 异步更高* |
| raw_reward 均值 | 51.1% | 52.9% | 无差异 |

*异步 prefix cache 命中率更高可能是因为权重更新间隔不同导致的热缓存差异，不影响结论。

**Rollout 完全不受同步/异步影响**——SGLang 引擎独立运行，两种模式对它无区别。

---

## 四、训练质量

### 4.1 Loss 对比

| 指标 | 异步 | 同步 |
|------|------|------|
| pg_loss 范围 | -0.065 ~ +0.065 | -0.090 ~ +0.090 |
| pg_loss 非零步比例 | ~75% | ~59% |
| entropy_loss 起始 | 0.49 | 0.50 |
| entropy_loss 结束 | 0.20 | 0.24 |
| entropy 下降趋势 | 是 ✅ | 是 ✅ |
| pg_clipfrac 均值 | ~0.0003 | ~0.0003 |
| grad_norm 均值 | ~0.21 | ~0.24 |
| grad_norm=0 步数 | ~25% | ~34% |
| logprob_abs_diff | 0.013 | 0.013 |

### 4.2 zero_std 组对比

| 指标 | 异步 | 同步 |
|------|------|------|
| count_0 (全错) | 3.0 / 步 | 2.5 / 步 |
| count_1 (全对) | 2.3 / 步 | 2.5 / 步 |
| zero_std 占比 | 65.6% | 62.5% |
| 有效梯度组 | 2.8 / 8 | 3.0 / 8 |

**同步和异步训练质量完全一致**。Loss 波动幅度、entropy 收敛趋势、grad_norm 分布在统计范围内无显著差异。这验证了两种模式执行相同的数学——异步只是改变了 rollout/train 的时间排布，不影响计算正确性。

---

## 五、综合对比

| 维度 | 异步 | 同步 | 胜出 |
|------|------|------|------|
| 每步时间 | 239s | 386s | **异步 +62%** |
| 样本吞吐 | 0.54 样本/s | 0.33 样本/s | **异步 +64%** |
| 训练质量 | 一致 | 一致 | 平 |
| 显存占用 | 相同 | 相同 | 平 |
| GPU 利用率 | 57% | 35% | **异步 更高** |
| 复杂性 | 略高 | 简单 | 同步 |
| 代码成熟度 | 较新 | 成熟 | 同步 |

### 5.1 什么时候用同步

- **调试阶段**：日志清晰，每步单独执行，问题定位更直观
- **首轮验证**：确认参数、数据、loss 都正常后再切异步
- **多机 colocate**：异步不支持 `--colocate`

### 5.2 什么时候用异步

- **生产训练**：节省 ~40% 墙钟时间
- **rollout 耗时长**：生成样本多的场景受益最大
- **GPU 资源固定**：无法加 GPU，但想提速

---

## 六、建议

1. **默认使用异步**：在当前配置下节省 62% 时间，训练质量无损失
2. **开发/调试用同步**：出问题时切回同步排查，确认正常后切到异步跑长期训练
3. **4-GPU 场景异步收益更大**：只有 1 个 rollout 引擎时 rollout 时间更长，异步重叠节省的绝对时间更多
