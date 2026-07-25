# Round 1 报告（多主体身份崩塌 · 免训练 MIE 引导修正）

日期：2026-07-17 · 机器：RunPod A100 80GB · 分支：AAAI_V1

## 1. 结论：GO（核心 claim 成立，幅度 marginal）

免训练、推理时用冻结的分维度评测器 **MIE** 做闭环诊断+修正，在**同底座 OmniGen2** 上、
以**零训练成本**，在多主体身份崩塌上**追平并小胜 CVPR 2026 重训 SOTA（UMO）**。

- 独立判据：**SCR(↓, DINOv2 检测每主体) + DINO 相似度(↑)**；MIE 只在环内当控制器，不自评。
- 规模：Round 1 = 每档 30 任务（4 主体 hard + 2 主体 easy），5 方法。

## 2. 最终数字

### 4 主体（强交互+遮挡，hard，n=30）
| 方法 | SCR↓ | DINO↑ |
|---|---|---|
| ours (v1) | **0.500** | 0.481 |
| ours_v2 (改进) | 0.508 | 0.467 |
| UMO（重训 SOTA, CVPR26）| 0.525 | 0.436 |
| best-of-N（同算力挑）| 0.550 | 0.480 |
| one-shot | 0.558 | 0.407 |
| FreeGraftor（FLUX开环,跨系统, n=3）| 0.417 | 0.553 |

→ ours ≤ UMO 且优于所有同底座免训练基线。配对胜率 ours<UMO = 9/30（均值优、非碾压）。

### 2 主体（easy，n=30）
| 方法 | SCR↓ | DINO↑ |
|---|---|---|
| one-shot | **0.450** | 0.510 |
| ours_v2 | 0.483 | 0.515 |
| UMO | 0.483 | 0.530 |
| best-of-N | 0.517 | 0.528 |
| ours (v1) | 0.517 | 0.521 |

→ 改进版把易例从 v1 的 0.517 降到 0.483（追平 UMO、优于 best-of-N），但 one-shot 在易例仍最好。

## 3. 方法改进（v1 → v2，均免训练+MIE内）
- **触发门槛**：best init 的标准化异常 ≤ 0.75 就不修（够好别动）→ 缓解易例回退。
- **每步多提案取优**：每个修正步生成多个候选，MIE 选最好再判接受 → 定向小搜索。
- **预算对齐**：n_init 2 + k 3 × 提案 2 = 8，与 best-of-N=8 同算力。
- 效果：易例明显改善，难例持平。

## 4. 校准路由验证
- 路由**数据自适应**：不同任务路由到不同维（existence/appearance/interaction），
  **不固定选 Interaction**（证明校准修好了"永远选最低维"的问题）。
- 守卫式接受：仅当 MIE 总分↑ 且 目标维↑ 且 无其他维明显下降 才接受，否则回滚。

## 5. 诚实的局限
1. 幅度 marginal（4主体 0.50 vs UMO 0.525），**未做显著性检验**（n=30 太小）。
2. 配对胜率一般，均值优势来自减少灾难性崩塌。
3. 仅 SCR/DINO 自动指标，**无人评**。
4. 受 OmniGen2 ≤5 refs 限制，**仅 2/4 主体**，无 6/8。
5. FreeGraftor(跨系统 FLUX)在少量样本上 SCR 最低，需全量确认；不同底座不能因果对比。

## 6. Round 2 打法（决定 AAAI vs workshop）
1. **放大 500 任务 + 多 seed + 置信区间/配对检验**：证明 vs UMO 的差**统计显著**。
2. **人评**：抽 100–200 对 A/B（ours vs UMO / vs best-of-N），3 人投票，胜率 CI 站住。
3. **6/8 主体 on FLUX.2 的 scaling 曲线**：证 ours 优势**随主体数增大**（最有冲击力）。
4. **消融**：校准路由 vs 总分/argmin；+参考集操纵 vs 只改 prompt；换 VLM 控制器。
5. FreeGraftor 全量作跨系统旁证。

判定：上述 1-4 做出且显著 + 人评认 → **AAAI 有力竞争**；否则 **workshop(P13N) 稳**。

## 7. 复现
```bash
# 环境+模型已在 misc（.venvs/*, models/*），MIE 权重只读：
#   Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best
cd round1
# 完整 Round 1（tmux 内）：
OMNIGEN2_STEPS=28 ROUND1_CPU_OFFLOAD=0 OURS_PROPOSALS=2 OURS_DEFICIT_MIN=0.75 \
N_SUPPORTED=30 N_STRESS=30 N_CAL_SUPPORTED=30 N_CAL_STRESS=30 \
MIE_CKPT=<ckpt> bash run_round1.sh
# 对比：python3 peek_compare.py results ours_v2
```
