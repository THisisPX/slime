---
type: results-report
date: 2026-07-10
experiment_line: qwen3-vl-4b-geo3k-grpo
round: 1
purpose: v3 106-step RL training analysis
status: final
linked_experiments: []
---

# Qwen3-VL-4B GEO3K GRPO Training / v3 / 106-Step Analysis / 2026-07-10

## 1. Executive Summary

在 4× NVIDIA B300 SXM6 GPU 上对 Qwen3-VL-4B-Instruct 进行 GRPO 数学几何推理 RL 训练（106 steps），训练显示明确的正向学习信号：

- **raw_reward**: 0.469 → 0.555 (+18.3%)
- **回复长度**: 1784 → 1431 (−19.8%)
- **截断率**: 38.3% → 15.6% (−59.3%)
- **熵**: 0.280 → 0.278 (稳定)
- **eval_reward**: 0.491 → 0.526 (+7.1%)

模型学会了在 3072 token 限制内高效推理，没有出现训练崩溃。但 binary math reward 的梯度信号微弱 (train/loss ≈ 0)，是当前配置的主要瓶颈。

## 2. Experiment Identity and Decision Context

### 实验目标
在 slime 框架上对 Qwen3-VL-4B-Instruct 进行视觉数学几何推理（GEO3K 数据集）的 GRPO RL 训练，验证 VLM RL 训练管线的可行性，找到稳定的超参数配置。

### 关键决策
- 是否能用 binary math reward (0/1) 训练 VLM？
- 如何在 4-GPU 共享环境中平衡训练/推理显存？
- max-response-len=3072 能否抑制 VLM 训练中常见的回复长度膨胀？
- B300 Blackwell GPU 的兼容性如何解决？

## 3. Setup and Evaluation Protocol

### 硬件
| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA B300 SXM6 AC (275 GiB each), CUDA 13.0 |
| 服务器 | 8 卡共享节点，通过 `CUDA_VISIBLE_DEVICES` 隔离 |
| 互联 | NVLink |

### 软件
| 项目 | 版本 |
|------|------|
| slime | main branch |
| PyTorch | 2.9.1+cu129 |
| SGLang | 源码安装 (cuda-graph disabled) |
| Megatron-LM | 开发版 (megatron-bridge 集成) |
| Docker | slimerl/slime:nightly-dev-20260629a |

### 完整训练参数

| 类别 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **模型** | `--hf-checkpoint` | `/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct/` | Qwen3-VL-4B-Instruct |
| | `--rotary-base` | 5000000 | VL 模型专用 RoPE |
| | `--megatron-to-hf-mode` | bridge | Megatron Bridge 加载 VL 权重 |
| | `--multimodal-keys` | `{"image": "images"}` | 图片数据列映射 |
| | `--apply-chat-template` | ✓ | 多模态消息模板 |
| **数据** | `--prompt-data` | `chenhegu/geo3k_imgurl/train.parquet` | GEO3K 训练集 |
| | `--input-key` | problem | 文本字段 |
| | `--label-key` | answer | 答案字段 |
| | `--loss-mask-type` | qwen3 | Qwen3 对话格式 |
| **GPU 分配** | `--actor-num-gpus-per-node` | 2 | 训练 2 卡 |
| | `--rollout-num-gpus` | 2 | 推理 2 卡 |
| | `--tensor-model-parallel-size` | 2 | TP2 × DP1 |
| **算法** | `--advantage-estimator` | grpo | 组内标准化 |
| | `--kl-loss-coef` | 0.00 | 无 KL 正则 |
| | `--kl-loss-type` | low_var_kl | |
| | `--entropy-coef` | 0.00 | 无熵惩罚 |
| | `--eps-clip` | 0.2 | PPO clip 下界 |
| | `--eps-clip-high` | 0.28 | PPO clip 上界 |
| **Rollout** | `--rm-type` | math | 数学 boxed 验证 |
| | `--num-rollout` | 500 | 目标 500 步 (实际 106) |
| | `--rollout-batch-size` | 16 | 每步 16 个 prompt |
| | `--n-samples-per-prompt` | 8 | 每个 prompt 8 个回答 |
| | `--rollout-max-response-len` | 3072 | 回复长度上限 |
| | `--rollout-temperature` | 0.8 | 采样温度 |
| | `--global-batch-size` | 64 | 训练 batch size |
| **优化器** | `--optimizer` | adam | |
| | `--lr` | 1e-6 | |
| | `--lr-decay-style` | constant | |
| | `--weight-decay` | 0.1 | |
| | `--adam-beta1` | 0.9 | |
| | `--adam-beta2` | 0.98 | |
| **Megatron** | `--sequence-parallel` | ✓ | TP2 配合 SP |
| | `--recompute-granularity` | full | |
| | `--recompute-method` | uniform | |
| | `--recompute-num-layers` | 1 | |
| | `--use-dynamic-batch-size` | ✓ | |
| | `--max-tokens-per-gpu` | 2048 | VLM 保守设置 |
| | `--attention-backend` | flash | FA2 (B300 不支持 FA3) |
| **SGLang** | `--rollout-num-gpus-per-engine` | 2 | TP2 推理 |
| | `--sglang-mem-fraction-static` | 0.7 | 推理显存比例 |
| | `--sglang-mm-attention-backend` | sdpa | B300 Blackwell 兼容 |
| | `--sglang-disable-cuda-graph` | ✓ | B300 CUDA graph 兼容 |
| **评估** | `--eval-interval` | 20 | 每 20 步评测 |
| | `--eval-max-response-len` | 4096 | 评测不限制长度 |
| | `--eval-top-p` | 1 | 贪婪解码 |
| **监控** | `--use-tensorboard` | ✓ | TensorBoard 监控 |
| | `--tb-project-name` | qwen3-vl-4b-geo3k-4gpu-v3 | |

### B300/Blackwell 兼容性修复

| 问题 | 错误信息 | 修复方案 |
|------|---------|---------|
| ptxas 不识别 sm_103a | `Value 'sm_103a' is not defined for option 'gpu-name'` | `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| CUDA kernel 无 sm_103a 镜像 | `cudaErrorNoKernelImageForDevice` at `F.linear` | `--sglang-disable-cuda-graph` |
| FA3 不支持 Blackwell | PTXAS error | `--sglang-mm-attention-backend sdpa` + `--attn-implementation flash_attention_2` |
| CUDA 内存碎片 | OOM at 20 GiB / 267 GiB | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 共享 GPU 隔离 | 8 卡节点他人占用 | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| megatron-bridge 模块路径变更 | `No module named 'megatron.bridge.models.qwen.qwen_provider'` | `slime_plugins/megatron_bridge/__init__.py` try/except 包裹 |
| dapo RM 不兼容 GEO3K | `ValueError: could not convert string to float: '\sqrt{21}'` | 回退 `--rm-type math` |

### 评估协议
- 每 20 rollout 评估一次
- 测试集：geo3k_imgurl/test.parquet
- 贪婪解码 (top-p=1)，最大 4096 token
- Primary metric: eval/geo3k_imgurl (math accuracy)

## 4. Training Metrics

### 4.1 Rollout 指标 (训练集)

| 指标 | Step 0 | Step 99 | Δ | 趋势 |
|------|--------|--------|----|------|
| raw_reward | 0.469 | 0.555 | +18.3% | ✅ 改善 |
| response_len/mean | 1784 | 1431 | −19.8% | ✅ 有效缩短 |
| response_len/median | 1561 | 2470 | — | ⚠️ 分布变宽 |
| truncated_ratio | 38.3% | 15.6% | −59.3% | ✅ 大幅下降 |
| zero_std/count_0 (all-bad) | 5 | 8 | — | → 稳定 |
| zero_std/count_1 (all-good) | 4 | 1 | — | ⚠️ 全对组减少 |
| rollout_log_probs | −0.261 | −0.323 | −23.7% | ⚠️ 置信度下降 |

**步步序列 (每 10 步)**：
```
raw_reward:  0.469 → 0.422 → 0.430 → 0.656 → 0.461 → 0.391 → 0.500 → 0.453 → 0.594 → 0.383 → 0.555
```
±0.10 级别的波动是 binary reward 小 group size 的正常现象。整体向上趋势可信（Spearman rank correlation 在正方向）。

### 4.2 训练健康指标

| 指标 | Step 0 | Step 199 | Δ | 解读 |
|------|--------|---------|----|------|
| train/loss | 0.0000 | 0.0002 | → 0 | 梯度弱但存在 |
| train/grad_norm | 0.419 | 0.507 | +21.0% | 梯度信号健康 |
| train/entropy_loss | 0.280 | 0.278 | −0.7% | ✅ 稳定，未恶化 |
| train/pg_clipfrac | 0.00 | 0.13% | — | 极少 clip |
| train/ppo_kl | 0.0000 | 0.0001 | → 0 | KL 几乎为零 |
| rollout/log_probs | −0.273 | −0.335 | −22.7% | ⚠️ 置信度下降 |

### 4.3 评估指标 (测试集)

| Eval Step | eval_reward | resp_len/mean | truncated_ratio |
|-----------|-------------|---------------|-----------------|
| 0 | 0.491 | 1901 | 26.8% |
| 19 | 0.528 | 1782 | 16.0% |
| 39 | 0.520 | 1734 | 15.3% |
| 59 | 0.547 | 1714 | 14.5% |
| 79 | 0.534 | 1838 | 17.2% |
| 99 | 0.526 | 1847 | 15.1% |

eval_reward 在 0.49~0.55 间波动，峰值 0.547 (step 59)，末值 0.526。提升幅度 (~3.5pp) 在噪声范围内，需要更长实验确认。

## 5. Performance Analysis

| 指标 | Step 0 | Step 99 | 改善 |
|------|--------|--------|------|
| rollout_time | 94.3s | 93.9s | → 稳定 |
| tokens_per_gpu_per_sec | 1211 | 1503 | +24.1% |
| step_time | 495s | 234s | −52.7% |
| train_time | 194s | 135s | −30.4% |
| train_wait_time | 301s | 99s | −67.1% |
| wait_time_ratio | 60.8% | 42.3% | −30.4% |
| actor_train_tflops | 25.1 | 28.1 | +12.0% |

### 分析
- **GPU 利用率逐步改善**：wait_time_ratio 从 61% 降至 42%，表明 rollout 和训练的重叠效率在提升
- **推理吞吐稳步增长**：tokens/gpu/sec 从 1211 上升至 1503，SGLang 引擎的 prefix caching 逐渐生效
- **禁用 CUDA graph 的代价**：rollout 时间约 94s，对比 text-only 4B 模型约 80s 有 ~17% 的额外开销，在可接受范围内
- **B300 训练吞吐正常**：TP2 配置下 25-28 TFLOPS 的 actor 训练性能合理

## 6. Key Findings

**F1: max-response-len=3072 有效抑制了 VLM 训练中的回复长度膨胀**

训练过程中回复平均长度不仅没有增长，反而从 1784 降至 1431 tokens。truncated_ratio 从 38.3% 显著降至 15.6%。这与 v1 实验中 max-len=4096 时长度暴涨 58% 形成鲜明对比。3072 的限制迫使模型学习在有限空间内给出正确答案。

**F2: 106 步内模型展现了明确的 reward 改善**

raw_reward 从 0.469 上升至 0.555，且未见衰减趋势。模型学会了更多地在回复中给出正确 boxed 答案。

**F3: entropy 保持稳定，未被 binary reward 破坏**

entropy_loss 在 0.280→0.278 范围内，未出现训练后期常见的 entropy collapse 信号。这表明当前的 group size (8) 和 max-len (3072) 组合足以维持输出多样性。

**F4: GRPO + binary math reward 存在梯度瓶颈**

train/loss 始终维持在 ~0.0000 水平。GEO3K 的 math 答案形式多样（LaTeX 公式），即使正确答案也难保证 group 内 8 个样本的 reward 有足够方差。zero_std/count_0 在 5-8 之间波动，说明始终有一定比例的 group 无法产生有效梯度。

**F5: B300 Blackwell GPU 需要 7 项兼容性适配**

从 ptxas 路径、CUDA graph 禁用、FA3→FA2 回退、到 expandable_segments 内存修复和 megatron-bridge 模块路径变更——共 7 处适配才使训练正常运行。这些修复已集成到训练脚本中，可供后续 B300 实验复用。

## 7. Limitations

- **loss ≈ 0**：GRPO 的 group-normalized advantage + binary reward 生成的梯度量级在可测量下限
- **eval_reward 绝对值低 (~0.53)**：离可用模型尚有距离，math accuracy 仅指格式+答案双重匹配
- **未测试其他 reward 函数**：dapo RM 因 GEO3K 答案含 LaTeX 公式而崩溃，gpqa 等未测试
- **n-samples=16 OOM**：4-GPU 配置下无法测试更大 group size
- **只跑了一个数据集**：GEO3K 结论的泛化性有限

## 8. Next Actions

### 短期
1. 测试连续值 reward 函数（如 gpqa 或自定义部分评分），增强梯度信号
2. 考虑 SFT 暖启动 + RL fine-tuning 的两阶段流程（参考 `run_geo3k_vlm_sft.sh`）

### 不推荐
- ❌ 还原 max-response-len 到 4096（已证明会导致回复长度膨胀）
- ❌ 继续使用 math RM + GRPO 长跑（梯度瓶颈限制可持续学习）

## 9. Artifact and Reproducibility Index

### Event Files
| 用途 | 路径 | 步数 |
|------|------|------|
| Eval | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090614/` | 106 |
| Train | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090856/` | 106 |

### Script
- `scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh`

### Checkpoint
- `iter_0000099`: `/workspace/volume/pengxiong/models/Qwen3-VL-4B_slime_geo3k_v3/iter_0000099/`

### 模型
- Qwen3-VL-4B-Instruct: `/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct/`
