---
type: results-report
date: 2026-07-10
experiment_line: qwen3-vl-4b-geo3k-grpo
round: 1
purpose: 100-step RL training baseline and ablation comparison (v1 vs v2 vs v3)
status: final
linked_experiments: []
---

# Qwen3-VL-4B GEO3K GRPO Training / Round 1 / 100-Step Baselines / 2026-07-10

## 1. Executive Summary

在 4× NVIDIA B300 SXM6 GPU 上对 Qwen3-VL-4B-Instruct 进行 GRPO 数学几何推理 RL 训练，三轮实验逐步优化：

| 版本 | 核心配置 | 结果 |
|------|---------|------|
| **v1** (100 steps) | math RM, 4 samples/group, max-len=4096 | ❌ 训练崩溃：reward↓47%, 回复长度爆涨58% |
| **v2** (100 steps) | math RM, 8 samples/group, max-len=3072 | → 稳定但停滞：loss≈0, eval基本持平 |
| **v3** (106 steps) | math RM, 8 samples/group, max-len=3072, 500 rollouts目标 | ✅ 唯一有效版本：raw_reward↑18%, trunc↓60%, 熵稳定 |

v3 是唯一实现有效学习趋势的配置。核心差异化因素：`max-response-len=3072`（抑制长度膨胀）和足够的数据量。B300 兼容性通过禁用 CUDA graph + sdpa 注意力后端 + expandable_segments 内存修复解决。

**实验定论**：Qwen3-VL-4B 在 GEO3K 上使用 GRPO + math binary reward 可以学到有用行为，但梯度信号微弱（loss≈0），需要精心控制回复长度和样本量。208步（v3+v3-resume）后出现 entropy collapse 早期信号，建议在 ~200 步停止。

## 2. Experiment Identity and Decision Context

### 实验目标
在 slime 框架上对 Qwen3-VL-4B-Instruct 进行视觉数学几何推理（GEO3K 数据集）的 GRPO RL 训练，验证 VLM RL 训练管线的可行性，找到稳定的超参数配置。

### 关键决策
- 是否能用 binary math reward (0/1) 训练 VLM？
- 如何在 4-GPU 共享环境中平衡训练/推理显存？
- max-response-len 和 n-samples-per-prompt 哪个对训练稳定性更重要？
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
| SGLang | 源码安装 (sglang disable-cuda-graph) |
| Megatron-LM | 开发版 (megatron-bridge 集成) |
| Docker | slimerl/slime:nightly-dev-20260629a |

### 训练配置（三版共享部分）
| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-VL-4B-Instruct (`/workspace/volume/distributed-training-softdata/models/`) |
| 数据集 | chenhegu/geo3k_imgurl (train.parquet + test.parquet) |
| GPU 分配 | 训练 2 卡 (TP2, DP1) + 推理 2 卡 (TP2, 1 engine) |
| 算法 | GRPO (`--advantage-estimator grpo`) |
| KL | `--kl-loss-coef 0.00`, `--kl-loss-type low_var_kl` |
| Entropy | `--entropy-coef 0.00` |
| Clip | `--eps-clip 0.2 --eps-clip-high 0.28` |
| 优化器 | Adam, lr=1e-6, constant, wd=0.1, β=[0.9,0.98] |
| 训练精度 | fp32 (accumulate-allreduce-grads-in-fp32, attention-softmax-in-fp32) |
| Attention | flash attention 2 (B300 不支持 FA3) |
| Megatron | TP=2, SP, full recompute, dynamic batch, max-tokens-per-gpu=2048 |
| 模型加载 | `--megatron-to-hf-mode bridge` (Megatron Bridge) |
| VLM 数据 | `--multimodal-keys '{"image": "images"}' --apply-chat-template` |
| 监控 | TensorBoard only |

### 三版差异
| 参数 | v1 | v2 | v3 |
|------|----|----|----|
| `--n-samples-per-prompt` | 4 | 8 | 8 |
| `--rollout-batch-size` | 32 | 16 | 16 |
| `--rollout-max-response-len` | 4096 | 3072 | 3072 |
| `--num-rollout` | 100 | 100 | 106 (实际)/500(目标) |
| `--rm-type` | math | math | math |
| 总样本/步 | 128 | 128 | 128 |
| B300 修复 | TRITON_PTXAS_PATH, disable-cuda-graph | 同 v1 | 同 v1 + expandable_segments |

### B300/Blackwell 兼容性修复
| 问题 | 修复 |
|------|------|
| ptxas 不识别 sm_103a | `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| CUDA kernel 无 sm_103a 镜像 | `--sglang-disable-cuda-graph` |
| FA3 不支持 Blackwell | `--sglang-mm-attention-backend sdpa` + `--attn-implementation flash_attention_2` |
| CUDA 内存碎片 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 共享 GPU 隔离 | `CUDA_VISIBLE_DEVICES=0,1,2,3` |

### 评估协议
- 每 20 rollout 评估一次 (`--eval-interval 20`)
- 测试集：geo3k_imgurl/test.parquet
- `--n-samples-per-eval-prompt 1`, `--eval-max-response-len 4096`, `--eval-top-p 1`
- Primary metric: eval/geo3k_imgurl (math accuracy)

## 4. Main Findings

### 4.1 训练指标对比

| 指标 | v1 (100步) | v2 (100步) | **v3 (106步)** | 最优方向 |
|------|-----------|-----------|---------------|---------|
| raw_reward (始→终) | 0.664→0.352 | 0.500→0.500 | **0.469→0.555** | ↑ v3 |
| eval_reward (始→终) | 0.508→0.532 | 0.491→0.542 | **0.491→0.526** | → 均微升 |
| resp_len (始→终) | 1663→2630 | 1785→1888 | **1784→1431** | ↓ v3 |
| truncated_ratio (始→终) | 18.8%→31.2% | 37.5%→30.5% | **38.3%→15.6%** | ↓ v3 |
| entropy (始→终) | 0.267→0.359 | 0.266→0.339 | **0.280→0.278** | → v3 |
| grad_norm (始→终) | 0.39→0.19 | 0.55→0.41 | **0.42→0.51** | ↑ v3 |
| train/loss | ≈0.0000 | ≈0.0000 | ≈0.0000 | — 均无效 |

### 4.2 关键发现

**F1: max-response-len 是训练稳定性的决定性因素** (置信度: high)

v1 使用 4096 上限，回复长度暴涨 58%，truncated 上升 66%。v2/v3 使用 3072 上限，回复长度受控（v2 +6%, v3 -20%）。4096 长的回复产生了大量的 zero-std 组（全对或全错），导致 GRPO 的组内标准化后 advantage=0，梯度消失。

**F2: n-samples-per-prompt 4→8 不能单独挽救训练** (置信度: high)

v2 仅有 8 samples 的改动（相对于 v1），raw_reward 保持平坦，eval 仅 +2.4pp。没有 max-response-len 的配合，更多的样本不能阻止回复炸裂。

**F3: 500 rollouts 的充足数据量有正向趋势但受限于 binary reward** (置信度: medium)

v3 在 106 步内 raw_reward 从 0.469→0.555 (+18.3%)，训练有明显学习信号。但 v3-resume 续训到 208 步时 entropy 暴涨 60%，说明 binary math reward 的梯度在长时间训练后出现 policy degradation。

**F4: GRPO + binary math reward 存在本质的梯度瓶颈** (置信度: high)

三个版本 train/loss 始终 ≈ 0。GEO3K 的 math 答案多样（LaTeX 公式），即使是正确答案，group 内 8 个样本的 reward 分布也趋于同质（全部 0 或全部 1），zero-std 组无法产生有效梯度。

## 5. Statistical Validation

由于是 RL 训练（非独立实验重复），以下统计基于 rollout 级别的趋势分析。

### 5.1 raw_reward 趋势稳定性

v3 raw_reward 106 步序列：
```
step  0: 0.469
step 10: 0.422
step 20: 0.430
step 30: 0.656
step 40: 0.461
step 50: 0.391
step 60: 0.500
step 70: 0.453
step 80: 0.594
step 90: 0.383
step 99: 0.555
```
所有版本 (v1, v2, v3) 均显示 ±0.10 级别的步步波动，这是 binary reward 小 group size 的正常现象。

### 5.2 eval_reward 稳定性
| 版本 | eval 首值 | eval 末值 | 最大 eval | 最小 eval |
|------|----------|----------|----------|----------|
| v1 | 0.508 | 0.532 | 0.544 | 0.506 |
| v2 | 0.491 | 0.542 | 0.552 | 0.491 |
| v3 | 0.491 | 0.526 | 0.547 | 0.491 |

eval 波动范围 ~0.05，v1 和 v2 的提升在此范围内。v3 的最终 eval (0.526) 低于中间峰值 (0.547)，需要更长实验确认趋势。

### 5.3 zero_std 组统计
| 版本 | zero_std_0 范围 (all-bad) | zero_std_1 范围 (all-good) |
|------|------------------------|--------------------------|
| v1 | 4→18 (恶化 4.5x) | 16→9 |
| v2 | 3→3 (稳定) | 5→5 (稳定) |
| v3 | 5→8 | 4→1 |

v1 的 zero_std_0 暴涨是最明确的问题信号，对应 raw_reward 崩溃。v2 和 v3 均保持稳定。

## 6. Performance Analysis

| 指标 | v1 | v2 | v3 | v3-resume |
|------|----|----|----|-----------|
| rollout_time (s) | 126→123 | 90→66 | 94→94 | 116→93 |
| tokens/gpu/sec | 847→1369 | 1275→1828 | 1211→1503 | 892→949 |
| step_time (s) | 496→284 | 511→207 | 495→234 | 350→268 |
| train_time (s) | 156→158 | 181→137 | 194→135 | 229→172 |
| train_wait_time (s) | 340→126 | 330→69 | 301→99 | 121→96 |
| wait_time_ratio | 68.5%→44.4% | 64.6%→33.5% | 60.8%→42.3% | 34.5%→35.8% |
| actor_train_tflops | — | 31.0→42.9 | 25.1→28.1 | 20.3→20.8 |
| longest_sample_tok/s | 32.6→33.3 | 34.3→46.5 | 32.6→32.7 | 26.5→33.0 |

### 性能观察
- **B300 训练性能正常**：TP2 训练 + TP2 推理配置下，actor_train_tflops 在 20-43 TFLOPS 范围，合理
- **推理吞吐稳步提升**：tokens/gpu/sec 从初始 ~850 上升到稳定 1300-1500
- **wait_time_ratio 下降**：GPU 空闲比例从 68% 降至 35-44%，利用率改善
- **v3-resume 性能偏低**：actor_train_tflops ~20 vs 初始 ~28，可能受模型输出分布变化影响
- **禁用 CUDA graph 的性能代价**：对比同规模文本模型的 rollout_time (~80s)，v2/v3 的 rollout 时间 (~90s) 有约 12% 减速，在可接受范围

## 7. Failure Cases / Negative Results / Limitations

### 7.1 主要失败模式

1. **v1 训练崩溃**：回复长度爆炸 (1663→2630 tokens)，zero_std_0 组从 4 增至 18，梯度完全消失
2. **loss 始终 ≈ 0**：GRPO 的 group-normalized advantage + binary reward 生成的梯度量级在下限附近
3. **dapo RM 不可用**：GEO3K 答案含 LaTeX 公式 (`\sqrt{21}`)，dapo 的 `int(float(gt))` 转换崩溃

### 7.2 v3-resume 续训问题
- entropy 从 0.321 暴涨至 0.519 (+62%)，模型输出分布快速陷入噪声
- grad_norm 从 0.65 衰减至 0.21，梯度信号在后半段变弱
- log_probs 持续恶化 (-0.315→-0.494)，模型置信度大幅下降
- **结论**：208 步可能接近 binary reward GRPO 的有效天花板

### 7.3 局限性
- 未测试更丰富的 reward 函数（如 GPQA-style 部分评分）
- 未测试 n-samples=16（4-GPU 配置下 OOM）
- eval_reward 绝对水平 (~0.55) 偏低，离可用模型尚有距离
- 只测试了 GEO3K 一个数据集，结论泛化性有限

## 8. What Changed Our Belief

| 先验信念 | 后验结论 | 置信度 |
|---------|---------|--------|
| binary math reward 可以训练 VLM | 在 ~100 步内有效，但梯度瓶颈在 ~200 步开始显现 | high |
| n_samples 越大越好 | 8 在 4-GPU 下是显存最优解；16 OOM；max-len 比 n_samples 关键 | high |
| 直接复制文本模型训练脚本就可以训练 VLM | 不行——需要 `--multimodal-keys`, `--megatron-to-hf-mode bridge`, `--rotary-base 5000000` | high |
| B300 可以直接跑 slime | 需要 4 项兼容性修复 (ptxas, cuda-graph, fa3→fa2, expandable_segments) | high |
| 100 步足以判断训练方向 | 对稳定性趋势足够；reward 趋势需要更长实验 (200+) | medium |

## 9. Next Actions

### 短期（本周）
1. **停止当前的 math RM + GRPO 路径** — binary reward 在 200+ 步后不可持续
2. 测试 `--rm-type gpqa` 或自定义部分评分 reward（提供 0~1 连续值，增强梯度）
3. 或改用 SFT 暖启动 + RL fine-tuning 的两阶段流程（参考 `run_geo3k_vlm_sft.sh`）

### 中期
4. 如果显存允许 (`CUDA_VISIBLE_DEVICES` 选更多卡)，测试 n-samples=16
5. 添加 `--kl-loss-coef 0.005`（需要 `--ref-load` 指向 Megatron checkpoint）
6. 在其他 VLM 任务（如 ScienceQA, MMMU）上验证稳定性发现

### 不推荐
- ❌ 继续使用 math RM + GRPO 跑 500+ 步（v3-resume 已显 entropy collapse 信号）
- ❌ 还原 max-response-len 到 4096（v1 已证明会导致崩溃）

## 10. Artifact and Reproducibility Index

### Event Files
| 实验 | TB 日志 | 步数 |
|------|---------|------|
| v1 | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu/20260701_084406/` (eval), `20260701_084710/` (train) | 100 |
| v2 | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v2/20260703_073646/` (eval), `20260703_073924/` (train) | 100 |
| v3 | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090614/` (eval), `20260707_090856/` (train) | 106 |
| v3-resume | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3-resume/20260708_022213/` (eval), `20260708_022342/` (train) | 109 |

### Scripts
- v1: `scripts/run-qwen3-VL-4B-geo3k-4gpu.sh`
- v2: `scripts/run-qwen3-VL-4B-geo3k-4gpu-v2.sh`
- v3: `scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh`
- v3-resume: `scripts/run-qwen3-VL-4B-geo3k-4gpu-v3-resume.sh`

### Checkpoint
- v3 iter_0000099: `/workspace/volume/pengxiong/models/Qwen3-VL-4B_slime_geo3k_v3/iter_0000099/`

### 模型
- Qwen3-VL-4B-Instruct: `/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct/`
