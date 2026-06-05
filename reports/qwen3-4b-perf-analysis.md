# Qwen3-4B GRPO 性能分析报告 (n_samples=16)

## 实验配置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-4B (4B params, 36 layers, hidden=2560) |
| 硬件 | 8× A100 (训练 4 卡 TP2×DP2 + 推理 4 卡 2引擎×TP2) |
| 算法 | GRPO, eps_clip=[0.2, 0.28], kl-loss-coef=0 |
| 数据 | dapo-math-17k, max-response-len=8192, n-samples-per-prompt=16 |
| batch | global-batch-size=32, rollout-batch-size=8 |
| 每步生成 | 8 prompt groups × 16 samples = **128 样本** |
| 运行 | 8 rollout steps + 32 train steps |

---

## 一、端到端时序

### 1.1 每步时间分解

| 阶段 | 时间 (均值) | 占比 |
|------|------------|------|
| rollout_time | 223.1s | — (异步预取, 与训练重叠) |
| train_wait_time | 106.6s | 44.6% |
| data_preprocess | 0.22s | 0.1% |
| log_probs_time | 16.9s | 7.1% |
| actor_train_time | 115.0s | 48.1% |
| update_weights | 0.45s | 0.2% |
| **step_time** | **239.2s** | **100%** |

**关键观察**：

- `wait_time_ratio` = 43.1%，说明 **pipeline 重叠率良好**：rollout 需 223s，训练需 158s (train_time)，但异步预取后实际等待仅 107s，接近一半的 rollout 时间被训练覆盖
- **训练是瓶颈** (115s actor_train + 17s log_probs = 132s 计算)，超过了 rollout 等待 (107s)
- `data_preprocess` 和 `update_weights` 几乎可忽略 (< 0.5s)

### 1.2 各步时序 (step 0–7)

| step | rollout_time | train_wait | actor_train | log_probs | step_time |
|------|-------------|-----------|-------------|-----------|-----------|
| 0 | 214.2s | 215.5s | 102.1s | 20.0s | 338.3s |
| 1 | 216.5s | 94.4s | 113.7s | 15.9s | 224.6s |
| 2 | 212.7s | 82.9s | 87.9s | 14.9s | 186.4s |
| 3 | 224.6s | 121.9s | 139.9s | 17.8s | 280.3s |
| 4 | 215.8s | 58.1s | 114.8s | 16.5s | 190.0s |
| 5 | 229.3s | 98.1s | 126.7s | 17.1s | 242.5s |
| 6 | 214.1s | 70.3s | 110.8s | 15.7s | 197.5s |
| 7 | 238.2s | 111.7s | 124.1s | 17.6s | 253.9s |
| **均值** | **223.1s** | **106.6s** | **115.0s** | **16.9s** | **239.2s** |

step 0 的 train_wait 反常地高 (215.5s，与 rollout 等长)，因为第一步没有预取，训练必须等 rollout 完成。后续步异步预取生效，wait_time 显著降低。

---

## 二、Rollout 生成性能

### 2.1 吞吐

| 指标 | 均值 |
|------|------|
| rollout_time | 223.1 s |
| tokens_per_gpu_per_sec | 922.5 |
| effective_tokens_per_gpu_per_sec | 922.5 |
| longest_sample_tokens_per_sec | 37.1 |

**分析**：2 个 TP2 引擎并行生成 128 条回答，平均 922.5 tok/gpu/s (4 卡合计 ~3690 tok/s)。但 `longest_sample_tokens_per_sec` 仅 37.1 tok/s——最慢的单条生成速度极低。这是因为 128 条回答中存在命中 8192 上限的"最长路径"，它阻塞了整个 rollout 的完成，形成长尾效应。

### 2.2 生成质量

| 指标 | 均值 | 范围 |
|------|------|------|
| response_len mean | 6,369 | 5,220–7,244 |
| response_len median | 7,258 | 4,902–8,192 |
| response_len max | **8,192** (所有步均触及) |
| response_len min | 2,105 | 1,388–3,689 |
| truncated_ratio | **47.6%** | 29.7%–63.3% |
| total_lengths (prompt+response) | 6,498 | 5,390–7,393 |

**核心问题：8192 token 不够。** 所有 8 步的 max response length 都触及 8192 上限。接近一半 (47.6%) 的回答被截断。数学 CoT 推理需要大量 token 来展开思考过程，截断直接导致部分回答因答案不完整被判定错误。建议将 `rollout-max-response-len` 提升至 16384。

### 2.3 Prefix Cache 效果

| 指标 | 均值 |
|------|------|
| prefix_cache_hit_rate | 59.7% |
| avg_cached_tokens_per_sample | 89.4 |

n_samples=16 时 SGLang radix cache 发挥显著作用：每个 prompt 的 16 条回答共享前缀 token，平均节约 89 token/prompt。cache 命中率波动较大 (17.6%–90.3%)，未命中的情况可能是因为 rollout batch 间切换了不同的 prompt。

### 2.4 重复检测

| 指标 | 值 |
|------|-----|
| repetition_frac | 0% (所有步均为 0) |

推理过程未检测到 token 重复退化，说明 Qwen3-4B 生成质量稳定，没有陷入循环输出的问题。

---

## 三、训练性能

### 3.1 计算效率

| 指标 | 均值 |
|------|------|
| actor_train_tflops | 55.3 |
| actor_train_tok_per_s | 7,312 |
| log_probs_tflops | 124.6 |

Qwen3-4B 在 4×A100 TP2+DP2 下达到 55 TFLOPS (actor train)。结合 log_probs 的 125 TFLOPS 和 `wait_time_ratio`=43%，GPU 空闲时间来自 rollout 等待。若拉大 gap (减少 wait)，整体利用率还有提升空间。

### 3.2 端到端效率

| 指标 | 值 |
|------|-----|
| step_time | 239.2 s |
| 每步样本数 | 128 |
| 样本/秒 | **0.54** |
| 每样本 tokens | ~6,544 |
| **tokens/秒 (端到端)** | **~3,506** |

---

## 四、训练质量

### 4.1 Loss 曲线

| step | train/loss | pg_loss | entropy_loss | ppo_kl | pg_clipfrac | grad_norm |
|------|-----------|---------|-------------|--------|-------------|-----------|
| 0 | 0.0000 | 0.0000 | - | - | - | 0.257 |
| 2 | 0.0001 | 0.0001 | 0.445 | -8.7e-5 | 2.5e-4 | 0.271 |
| 5 | **-0.0071** | **-0.0071** | 0.329 | -3.7e-5 | 6.5e-4 | **0.559** |
| 7 | **0.0343** | **0.0343** | 0.302 | 1.3e-4 | 2.8e-4 | 0.218 |
| 10 | 0.0174 | 0.0174 | 0.331 | 1.1e-4 | 2.7e-4 | 0.254 |
| 12 | **-0.0547** | **-0.0547** | 0.335 | 0 | 0 | 0.133 |
| 14 | 0.0630 | 0.0630 | 0.410 | 8.7e-5 | 4.9e-4 | 0.204 |
| 18 | 0.0291 | 0.0291 | 0.363 | -1.0e-4 | 2.4e-4 | 0.250 |
| 20 | 0.0409 | 0.0409 | 0.272 | 0 | 0 | 0.288 |
| 24 | 0.0343 | 0.0343 | 0.407 | 0 | 0 | 0.276 |
| 26 | -0.0350 | -0.0350 | 0.336 | -2.1e-5 | 4.9e-5 | 0.046 |
| 29 | **-0.0653** | **-0.0653** | 0.363 | 4.5e-5 | 3.1e-4 | 0.246 |
| 31 | 0.0000 | 0.0000 | 0.256 | 1.7e-4 | 0 | 0 |

**关键观察**：

- **pg_loss 已经非零**：在 -0.065 ~ +0.065 范围内波动，GRPO 的梯度在流动 ✅
- **entropy_loss 下降趋势**：从 0.49 逐步降至 0.20，模型从高熵随机生成收敛到更确定性的策略 ✅
- **ppo_kl 极小** (10⁻⁴–10⁻⁵)：kl_coef=0 未使用 ref 模型，(old_logprob - new_logprob) 仅由训练状态漂移引起，正常
- **pg_clipfrac 极低** (0.0002–0.0007)：模型更新幅度很小 (lr=1e-6)，ppo_kl 最小，ratio 几乎全在 [1-0.2, 1+0.28] 内
- **grad_norm 偶尔为 0**：对应 zero_std 的 step (见下文)

### 4.2 组内方差 (zero_std)

| 指标 | 均值 |
|------|------|
| zero_std/count_0 (16 条全错) | 3.0 groups/step |
| zero_std/count_1 (16 条全对) | 2.3 groups/step |
| zero_std 总占比 | **65.6%** (5.3/8) |
| 有效梯度组 | **2.8/8** (34.4%) |

即使 n_samples=16，仍有 65.6% 的 groups 内 reward 完全相同 (全对或全错)。这是因为 Qwen3-4B 能力有限 (~51% 正确率)，对许多题目要么全会要么全不会。这些组的 advantage=0，不贡献梯度。grad_norm 偶尔为 0 正好与此对应。

原始 reward (**raw_reward**) 均值为 51.1%，说明模型有一半概率做出正确答案——但对于 GRPO 训练，reward 信号的数值分布比均值更重要。当前配置下 34.4% 的 groups 有混合结果，能持续提供有效梯度驱动训练。

### 4.3 策略漂移

| 指标 | 均值 |
|------|------|
| train_rollout_logprob_abs_diff | 0.013 |

训练前后的 log probability 差异仅 0.013——模型策略更新幅度极小。与 pg_clipfrac 极低一致，32 步训练 (lr=1e-6, 每步仅 128 样本) 还不足以产生显著的策略改变。这是正常现象，RL 训练通常需要数百到上千步才出现明显的策略转移。

---

## 五、瓶颈分析与建议

### 当前瓶颈

| 瓶颈 | 严重度 | 说明 |
|------|--------|------|
| Response 截断 | 🔴 高 | 47.6% 回答触及 8192 上限，直接降低有效 reward |
| Rollout 长尾 | 🟡 中 | longest_sample_tok/s 仅 37.1，最慢样本阻塞整体 rollout |
| 模型能力 | 🟡 中 | 51% 正确率 → 65.6% 组无梯度 |

### 建议

1. **提升 response 上限**：`--rollout-max-response-len` 8192 → 16384。能解决截断问题，但 rollout_time 预计增加 30–50%
2. **增大 rollout batch**：`--rollout-batch-size` 8 → 16，提升生成并行度，缓解长尾问题
3. **继续迭代**：当前仅 32 train steps，loss 波动正常。需要持续观察数百步后的 loss 趋势和 entropy 收敛情况
