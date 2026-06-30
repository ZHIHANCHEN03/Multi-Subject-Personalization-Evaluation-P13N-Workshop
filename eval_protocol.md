# 评测协议（Phase 0.5，预注册）

> 配套 [engineering_plan.md](engineering_plan.md)。本协议在**开训之前**定死；训练后不得修改，以免事后挑指标。
> 目的：判定 **η>0（方法）是否真的优于 η=0（baseline）**。

---

## 1. 评测集

- 来源：gold 集的 **prompt + 参考图**（与训练 curated 集无泄漏）。
- 固定抽取一个子集：每个 N∈{2,4,6,8} 各取若干条，**task_id 清单写死**，之后不换。
- 输出物：`eval_subset.json`（task_id + prompt + subject_refs）。

## 2. 生成

- 对每个 prompt，用 flex2 + 各 η 的 LoRA 各生成一张：η ∈ {0, 0.3, 0.5, 0.7}。
- **配置全部锁死，只让 η 变**：sampler / steps / CFG / 分辨率 / negative prompt 固定；
  **每个 prompt 用同一个种子**，4 个 η 共用 → 差异只来自模型权重。
- 输出物：`gen/{eta}/{task_id}.png`。

## 3. 裁判（GPT-4V）

- **配对比较**：同 prompt+种子，η>0 的图 vs η=0 的图，判哪张更好。
- 每对**同时输出**：总体偏好（A/B）+ 三维诊断 **E / A / I**（二元，照搬训练标签的 rubric）。
- **去偏置**：A/B 顺序随机化，正反各判一次取一致结果。
- 与训练用的 Gemini **换家族**，避免循环。
- 输出物：`judge/{eta}_vs_0.jsonl`。

## 4. 质量护栏（别省）

- 测一个画质指标（**FID**，或 aesthetic / CLIP-IQA 任一）。
- 作用：证明"绑定变好"不是拿画质换来的。η>0 的画质不应显著劣于 η=0。

## 5. 报告与判定

- **两张表**：
  - 整体表：各指标 × η∈{0,0.3,0.5,0.7}。
  - 按 N 分桶表：N=2/4/6/8 分别看（主打多主体，增益预计集中在高 N）。
- **判定规则（事先定死）**：η>0 相对 η=0 的**配对胜率，统计显著 > 50%**（配对检验 / 胜率 CI 下界 > 50%）。
  - 主方法 = η 曲线上的最优档；峰不在 0.5 不算问题，改标签即可。

---

## Reference 指标（顺手测，非判定依据）

DINO/CLIP-I（Appearance）、TIFA/CompBench（Existence/Interaction）、PickScore/ImageReward（总体偏好）。
作分维度旁证，**不参与最终判定**。

## 通过条件（本阶段算完成）

- `eval_subset.json` 固定；`evaluate.py` 能在 baseline 上端到端跑通并吐出上述数字（dry-run）。
- 协议冻结，进入训练（Phase 3/4）。