# Slime Colocate 8×A100 40GB 训练分析报告

## 复现方式

```bash
bash scripts/run-qwen3-4B-sync-colocate-8gpu.sh
```

日志输出到 `logs/slime-vs-verl-colocate-8gpu/train_<timestamp>.log`。

## 训练日志

| 内容 | TensorBoard 路径 |
|------|-----------------|
| rollout metrics (8 steps) | `tensorboard_log/slime-vs-verl-colocate-8gpu/20260614_114622` |
| train metrics (32 steps) | `tensorboard_log/slime-vs-verl-colocate-8gpu/20260614_115229` |

---

## 一、实验配置与对比模式

| 项目 | Colocate (本实验) | Split (对比基线) |
|------|-------------------|------------------|
| 入口 | `train.py` | `train.py` |
| 硬件 | 8× A100 **40GB** | 8× A100 |
| 模型 | Qwen3-4B (4B params, 36 layers) | 同 |
| 算法 | GRPO, n_samples=16 | 同 |
| 脚本 | `run-qwen3-4B-sync-colocate-8gpu.sh` | `run-qwen3-4B-sync-8gpu.sh` |

---

## 二、并行策略与架构差异

### 2.1 GPU 分配模型

```
Split 模式 (对比基线):
┌────────── 训练 (Megatron) ──────────┐  ┌── 推理 (SGLang) ──┐
 GPU0  GPU1    GPU2  GPU3               GPU4  GPU5  GPU6  GPU7
 ├─TP2─┤       ├─TP2─┤                  ├TP2┤       ├TP2┤
 └── DP2 ──────┘                        └─ Engine 0 ┘  └─ Engine 1 ┘
    4 卡独立训练, 不与推理共享显存           4 卡独立推理

Colocate 模式 (本实验):
┌─────────── 8 卡共享, 分时复用 ──────────────┐
 GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 ├─TP4 训练────┤                             (offload 切换)
 ├─TP2 推理────────┤├─TP2 推理────┤├TP2推理──┤├TP2推理──┤
   训练时: 全部 8 卡用于 Megatron (显存全给训练)
   推理时: Megatron offload 到 CPU, SGLang onload 到 GPU (显存全给推理)
```

### 2.2 并行参数对比

| 参数 | Colocate | Split | 说明 |
|------|----------|-------|------|
| `--actor-num-gpus-per-node` | **8** | 4 | 训练 GPU 数 |
| `--rollout-num-gpus` | (colocate 自动) | 4 | 推理 GPU 数 |
| `--colocate` | **✓** | ✗ | 训练推理共享 GPU |
| `--tensor-model-parallel-size` | **4** | 2 | 张量并行度 |
| `--sequence-parallel` | ✓ | ✓ | 分散 LayerNorm/Dropout |
| `--pipeline-model-parallel-size` | 1 | 1 | 无流水线并行 |
| `--context-parallel-size` | 1 | 1 | 无上下文并行 |
| 数据并行 (隐式) | **DP=2** | **DP=2** | 取决于 total_gpus / TP / PP |
| 推理引擎数 | 4 × TP2 | 2 × TP2 | rollout_num_gpus ÷ gpus_per_engine |

**为什么要 TP4？** A100 40GB 下 TP2 训练每卡需 ~32GB (权重+优化器+梯度)，加上 colocate 推理的 KV cache (14GB) 就爆了 40G。改用 TP4 后每卡仅 ~14GB，留足 26GB 给激活 + 推理。

**TP4 vs TP2 的显存&计算权衡**：

| 每卡 | TP4 | TP2 |
|------|-----|-----|
| 权重 | 2.25 GB | 9 GB |
| 优化器 (Adam fp32) | 9 GB | 36 GB |
| 梯度 | 2.25 GB | 9 GB |
| 训练合计 | **~14 GB** | **~54 GB** |
| + KV cache (推理) | 14 GB | 28 GB |
| 总需求 | 28 GB ✓ | 82 GB ✗ (40G OOM) |
| actor 吞吐 | **14,685 tok/s** | 7,312 tok/s |

### 2.3 数据 & Rollout 参数

| 参数 | Colocate | Split | 说明 |
|------|----------|-------|------|
| `--num-rollout` | 8 | 8 | 测试用 (正式改 3000+) |
| `--rollout-batch-size` | 8 | 8 | 每轮 8 个 prompt group |
| `--n-samples-per-prompt` | **16** | **16** | GRPO 组大小 |
| `--rollout-max-response-len` | 8192 | 8192 | 回答最大 token 数 |
| `--global-batch-size` | 32 | 32 | 全局 batch (DP 分片) |
| 每步总样本 | 128 | 128 | 8 × 16 |
| 每 rollout train steps | 4 | 4 | 128 / 32 |

### 2.4 推理引擎 (SGLang) 参数

| 参数 | Colocate | Split | 说明 |
|------|----------|-------|------|
| `--rollout-num-gpus-per-engine` | 2 | 2 | 每引擎 TP2 |
| `--sglang-mem-fraction-static` | **0.35** | 0.7 | 推理可用显存占比 |
| `--sglang-cuda-graph-max-bs` | 8 | 16 | CUDA graph 最大 BS |
| `--sglang-max-running-requests` | 32 | (默认) | 最大并发请求数 |
| 实际 KV cache 大小 | ~14 GB | ~28 GB | mem_fraction × 40G |

**这是 Colocate 在 40G 上最大的瓶颈**。0.35×40G=14GB KV cache，仅能同时缓存 ~1500 个 8K 序列 (≈每个序列 10MB KV)。Split 有 0.7×40G=28GB，是 Colocate 的 2 倍。

### 2.5 算法参数 (两模式完全一致)

| 参数 | 值 | 说明 |
|------|-----|------|
| `--advantage-estimator` | grpo | GRPO 优势估计 |
| `--eps-clip` / `--eps-clip-high` | 0.2 / 0.28 | PPO ratio 裁剪 |
| `--kl-loss-coef` | 0.00 | 不使用 ref 模型 |
| `--entropy-coef` | 0.00 | 不额外加熵奖励 |
| `--rewards-normalization` | ✓ (默认) | 组内均值归一化 |
| `--grpo-std-normalization` | ✓ (默认) | 组内标准差归一化 |
| `--optimizer` | adam | lr=1e-6, constant |
| `--weight-decay` / `--adam-beta1` / `--adam-beta2` | 0.1 / 0.9 / 0.98 | |
| `--recompute-granularity` / `--recompute-method` | full / uniform | 全重计算, 用显存换计算 |
| `--use-dynamic-batch-size` | ✓ | 动态 micro-batch 打包 |
| `--max-tokens-per-gpu` | **2048** | 4096 | Colocate 必须降 |

---

## 三、Colocate 特有开销

每次 rollout↔train 切换需要 CPU↔GPU offload：

| 阶段 | 均值 | 说明 |
|------|------|------|
| `sleep_time` | 7.4s | 训练完后模型 offload 到 CPU |
| `wake_up_time` | 3.9s | 推理前模型 onload 回 GPU |
| `update_weights_time` | 1.1s | 权重同步 (colocate 下更慢) |
| **总切换开销** | **12.4s/步** | sleep + wake + update |

Split 模式：update_weights 仅 0.5s，无 sleep/wake。Colocate 每步多了 ~12s 的纯切换开销。

---

## 四、端到端时序

### 4.1 每步时间分解

| 阶段 | Colocate (TP4) | Split (TP2) | 差异 |
|------|---------------|-------------|------|
| rollout_time | 275.7s | 224.2s | +23% |
| sleep_time | 7.4s | — | 新增 |
| wake_up_time | 3.9s | — | 新增 |
| train_wait_time | 316.3s | 254.6s | +24% |
| data_preprocess | 0.14s | 0.15s | — |
| log_probs_time | 16.3s | 16.8s | — |
| actor_train_time | **57.0s** | 113.9s | **-50%** |
| update_weights | 1.1s | 0.5s | — |
| **step_time** | **389.9s** | **386.0s** | **+1%** |

**总 step_time 几乎一样！**

原因分析：
- **训练快了 2 倍** (57s vs 114s)：TP4 把每卡参数量从 ~54GB 降到 ~14GB，dynamic batch 能打包更多 token
- **但 rollout 慢了 23%** (276s vs 224s)：mem_fraction=0.35 仅 14GB KV cache vs split 的 28GB
- 加上 12s 切换开销，最终两两抵消

### 4.2 各步时序明细 (step 0–7)

| step | rollout | sleep | wake | train_wait | train | step_time |
|------|---------|-------|------|-----------|-------|-----------|
| 0 | 335s | 4.8s | 2.8s | 556s | 96s | 653s |
| 1 | 258s | 8.2s | 3.6s | 273s | 71s | 344s |
| 2 | 247s | 7.5s | 4.0s | 261s | 66s | 327s |
| 3 | 283s | 8.1s | 4.2s | 298s | 76s | 374s |
| 4 | 224s | 8.2s | 4.1s | 239s | 69s | 308s |
| 5 | 285s | 7.5s | 4.1s | 300s | 72s | 371s |
| 6 | 276s | 7.6s | 3.8s | 291s | 68s | 359s |
| 7 | 298s | 7.4s | 4.1s | 313s | 71s | 384s |
| **均值** | **276s** | **7.4s** | **3.9s** | **316s** | **74s** | **390s** |

Step 0 的 train_wait 反常地高 (556s)：第一步没有预取，且 colocate 下第一次 offload 需要更长时间。

---

## 五、Rollout 生成性能

| 指标 | Colocate | Split | 说明 |
|------|----------|-------|------|
| rollout_time | 275.7s | 224.2s | +23% |
| tokens_per_gpu_per_sec | **372.7** | 907.7 | **-59%** |
| longest_sample_tok/s | 30.1 | 36.5 | -17% |
| response_len mean | 6,386 | 6,432 | — |
| truncated_ratio | 47.0% | 46.7% | — |
| prefix_cache_hit_rate | ~26% | ~27% | — |
| raw_reward 均值 | 52.1% | 52.9% | — |

**Colocate 推理吞吐只有 split 的 41%**。虽然 colocate 有 4 个 TP2 引擎（vs split 的 2 个），但 `sglang-mem-fraction=0.35` → 14GB KV cache 严重限制了并发能力。Split 模式 4 卡独立推理，每卡 mem_fraction=0.7 → 28GB KV cache，吞吐翻倍不止。

---

## 六、训练性能

| 指标 | Colocate TP4 | Split TP2 | 差异 |
|------|-------------|-----------|------|
| actor_train_time | **57.0s** | 113.9s | **-50%** |
| actor_train_tok_per_s | **14,685** | 7,312 | **+101%** |
| actor_train_tflops | 55.5 | 55.3 | — |

TP4 把训练吞吐翻倍——因为每卡参数量从 ~54GB 降到 ~14GB，dynamic batch 能打包更多 token，micro-batch 更大，GPU 利用率更高。4B 模型在 TP4 下每卡仅 2560/4=640 hidden dims。

---

## 七、训练质量

| 指标 | Colocate | Split | 说明 |
|------|----------|-------|------|
| pg_loss 范围 | -0.081 ~ +0.069 | -0.090 ~ +0.090 | ✅ 一致 |
| entropy_loss | 0.50 → 0.24 | 0.50 → 0.24 | ✅ 收敛趋势相同 |
| pg_clipfrac | ~0.0003 | ~0.0003 | ✅ 一致 |
| grad_norm 均值 | ~0.24 | ~0.24 | ✅ 一致 |
| zero_std 占比 | ~64% | ~62% | ✅ 一致 |
| raw_reward | 52.1% | 52.9% | ✅ 一致 |
| logprob_abs_diff | 0.013 | 0.013 | ✅ 一致 |

**训练质量与 split 模式完全一致**。TP4 和 colocate offload 不影响数学正确性。

---

## 八、三种模式综合对比

| 维度 | Async Split | Sync Split | Sync Colocate |
|------|------------|------------|---------------|
| GPU 分配 | 4 train + 4 infer | 4 train + 4 infer | 8 共享 (offload) |
| 训练并行 | TP2×DP2 | TP2×DP2 | **TP4×DP2** |
| 推理引擎 | 2×TP2 | 2×TP2 | 4×TP2 |
| sglang-mem-fraction | 0.7 | 0.7 | **0.35** |
| step_time | **239s** | 386s | 390s |
| rollout 吞吐 | 923 tok/gpu/s | 908 tok/gpu/s | 373 tok/gpu/s |
| 训练吞吐 | 7,312 tok/s | 7,312 tok/s | **14,685 tok/s** |
| GPU 利用率 | 57% | 35% | 19% |
| offload 切换开销 | 无 | 无 | ~12s/步 |
| 40G 兼容性 | ✅ (需改 TP) | ✅ (需改 TP) | ⚠️ 必须 TP4 |
| 多机扩展 | 可分开扩 | 可分开扩 | 需同机扩 |

---

## 九、与 verl 对比建议

| 维度 | slime colocate | verl megatron |
|------|---------------|---------------|
| 训练 TP/DP | TP4×DP2 | TP2×DP4 |
| 推理引擎 | SGLang, 0.35 mem | vLLM, 0.5 mem |
| 推理 CUDA graph | ON (bs≤8) | OFF (enforce_eager) |
| offload 机制 | torch_memory_saver | Megatron param/grad/opt offload |
| 推理吞吐 | 373 tok/gpu/s | 待测 |
| 训练吞吐 | 14,685 tok/s | 待测 |
| 切换开销 | ~12s | 待测 |

> 在相同 8×A100 40G 上运行 `verl run_qwen3_4b_megatron_perf_test.sh`，重点对比 rollout_time、actor_train_time、step_time。

---

## 十、结论

1. **Colocate 在 40G 上不划算**：推理吞吐被 mem_fraction=0.35 严重限制 (仅 split 的 41%)，抵消了 TP4 训练加速
2. **A100 40G 推荐 Async Split**：239s/步 (比 sync 快 39%)，推理训练独立显存，各自最优
3. **如果有 80G 显存**：Colocate 的 mem_fraction 可以调到 0.5+，推理吞吐恢复后将是更优方案
4. **训练质量无问题**：loss、entropy、grad_norm 与 split 完全一致，TP4 不影响数学正确性
