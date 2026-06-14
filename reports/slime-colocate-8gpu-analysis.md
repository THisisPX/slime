# Slime Colocate 8×A100 40GB 训练分析报告

## 实验配置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-4B (4B params, 36 layers, hidden=2560) |
| 硬件 | 8× A100 40GB |
| 模式 | **Colocate** (训练+推理共享 GPU, TP4×DP2) |
| 推理引擎 | SGLang, 4 引擎 × TP2, mem_fraction=0.35 |
| 算法 | GRPO, n_samples=16, max-response=8192, global-bs=32 |
| 每步生成 | 8 prompt groups × 16 samples = 128 样本 |
| 对比基线 | split sync-8gpu (4 训练+4 推理, TP2×DP2) |
| 日志 | `tensorboard_log/slime-vs-verl-colocate-8gpu/` |

---

## 一、Colocate 特有开销

Colocate 模式每次 rollout↔train 切换需要 CPU↔GPU offload：

| 阶段 | 均值 | 说明 |
|------|------|------|
| `sleep_time` | 7.4s | 训练完后模型 offload 到 CPU |
| `wake_up_time` | 3.9s | 推理前模型 onload 回 GPU |
| `update_weights_time` | 1.1s | 权重同步 (colocate 下更慢) |
| **总切换开销** | **12.4s/步** | |

对比分卡模式（split-8gpu）：update_weights 仅 0.5s，无 sleep/wake。colocate 每步多了 ~12s 的纯切换开销。

---

## 二、端到端时序

### 2.1 每步时间分解

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

**出乎意料：总 step_time 几乎一样！**

原因分析：
- **训练快了 2 倍** (57s→114s)：TP4 把每卡参数量从 35GB 降到 14GB，dynamic batch 能打包更多 token
- **但 rollout 慢了 23%** (276s→224s)：mem_fraction=0.35 仅 14GB KV cache vs split 的 28GB
- 加上 12s 切换开销，最终两两抵消

### 2.2 各步时序明细 (step 0–7)

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

Step 0 的 train_wait 反常地高 (556s)，因为第一步没有预取，训练必须等 rollout 完成，且 colocate 下第一次 offload 需要更长时间。

---

## 三、Rollout 生成性能

| 指标 | Colocate | Split | 说明 |
|------|----------|-------|------|
| rollout_time | 275.7s | 224.2s | +23% |
| tokens_per_gpu_per_sec | **372.7** | 907.7 | **-59%** |
| longest_sample_tok/s | 30.1 | 36.5 | -17% |
| response_len mean | 6,386 | 6,432 | — |
| truncated_ratio | 47.0% | 46.7% | — |
| prefix_cache_hit_rate | ~26% | ~27% | — |
| raw_reward 均值 | 52.1% | 52.9% | — |

**Colocate 推理吞吐只有 split 的 41%**。虽然 colocate 有 4 个 TP2 引擎（vs split 的 2 个），但 `sglang-mem-fraction=0.35` (A100 40G 下仅 14GB KV cache) 严重限制了并发能力。split 模式 4 卡独立推理，每卡 mem_fraction=0.7 (28GB KV cache)，吞吐翻倍不止。

---

## 四、训练性能

| 指标 | Colocate TP4 | Split TP2 | 差异 |
|------|-------------|-----------|------|
| actor_train_time | **57.0s** | 113.9s | **-50%** |
| actor_train_tok_per_s | **14,685** | 7,312 | **+101%** |
| actor_train_tflops | 55.5 | 55.3 | — |

TP4 把训练吞吐翻倍——因为每卡参数量从 35GB 降到 14GB，dynamic batch 能打包更多 token，micro-batch 更大，GPU 利用率更高。4B 模型在 TP4 下每卡仅 2560/4=640 hidden dims，矩阵乘法效率反而更高。

---

## 五、训练质量

| 指标 | Colocate | Split | 说明 |
|------|----------|-------|------|
| pg_loss 范围 | -0.081 ~ +0.069 | -0.090 ~ +0.090 | ✅ 一致 |
| entropy_loss | 0.50 → 0.24 | 0.50 → 0.24 | ✅ 收敛趋势相同 |
| pg_clipfrac | ~0.0003 | ~0.0003 | ✅ 一致 |
| grad_norm 均值 | ~0.24 | ~0.24 | ✅ 一致 |
| zero_std 占比 | ~64% | ~62% | ✅ 一致 |
| raw_reward | 52.1% | 52.9% | ✅ 一致 |
| logprob_abs_diff | 0.013 | 0.013 | ✅ 一致 |

**训练质量与 split 模式完全一致**。TP4 和 colocate offload 不影响数学正确性，两者执行完全相同的梯度计算。

---

## 六、三种模式综合对比

| 维度 | Async Split | Sync Split | Sync Colocate |
|------|------------|------------|---------------|
| step_time | **239s** | 386s | 390s |
| rollout 吞吐 | 923 tok/gpu/s | 908 tok/gpu/s | 373 tok/gpu/s |
| 训练吞吐 | 7,312 tok/s | 7,312 tok/s | **14,685 tok/s** |
| GPU 利用率 | 57% | 35% | 19% |
| 训练质量 | ✅ | ✅ | ✅ |
| 40G 兼容 | ✅ | ✅ | ⚠️ 需 TP4 |
| offload 开销 | 无 | 无 | ~12s/步 |

---

## 七、与 verl 对比建议

| 维度 | slime colocate | verl megatron |
|------|---------------|---------------|
| 训练 TP/DP | TP4×DP2 | TP2×DP4 |
| 推理引擎 | SGLang, 0.35 mem | vLLM, 0.5 mem |
| 推理 CUDA graph | ON (bs≤8) | OFF (enforce_eager) |
| offload 机制 | torch_memory_saver | Megatron param/grad/opt offload |
| 推理吞吐 | 373 tok/gpu/s | 待测 |
| 训练吞吐 | 14,685 tok/s | 待测 |
| 切换开销 | ~12s | 待测 |

> 在相同 8×A100 40G 上运行 verl 的 `run_qwen3_4b_megatron_perf_test.sh`，重点对比 rollout_time、actor_train_time、step_time 三个核心指标。

---

## 八、结论

1. **Colocate 在 40G 上不划算**：推理吞吐被 mem_fraction=0.35 严重限制，抵消了 TP4 训练加速的优势
2. **A100 40G 推荐 Async Split**：239s/步，推理和训练独立分配显存，各自最优
3. **如果有 80G 显存**：Colocate 的 mem_fraction 可以调到 0.5+，推理吞吐恢复后将成为最佳方案
4. **Colocate 训练质量无问题**：loss、entropy、grad_norm 与 split 完全一致
