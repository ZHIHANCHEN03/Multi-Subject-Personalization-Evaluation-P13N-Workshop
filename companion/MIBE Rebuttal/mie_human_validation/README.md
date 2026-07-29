# MIE × human validation — package for the MIBE rebuttal

**What this is.** We scored the PP1 reference-extension study with the trained MIE
evaluator and compared it against two fresh human-annotation batches. The point is
to answer, with numbers, the reviewer objection that MIE's generalization evidence
is too narrow.

**Data status: complete.** Both annotation batches are the full export as of
2026-07-27 — 648 items × exactly 2 annotators × 2 batches, 8 annotators, 2592 votes.
An earlier partial export is kept in `archive_partial/` for provenance only.

---

## Read in this order

| file | 用途 |
|---|---|
| **`FINAL_REPORT.md`** | 全部结果。先读这个。§2 是禁止引用的数字，§7 是相比部分数据的变化 |
| **`REVIEWER_MAPPING.md`** | 这项研究**答了哪些 review 点、没答哪些**，逐条对应 |
| **`REBUTTAL_PLAN.md`** | 17 条 review 诉求的分类 + 优先级 + 每条怎么执行 |

---

## The three numbers to quote

| | value | why it survives scrutiny |
|---|---|---|
| **AUC = 0.815** [0.755, 0.868] | MIE 的分差 → 人类偏好，pp2（两个生成器都未见过） | 常数预测器在 AUC 上恒为 0.500，所以失衡杀不掉它 |
| **existence r = +0.942** (κ 0.924) | MIE 的 existence 概率 vs 人类标注比例，GPT-Image-1.5 | 常数预测器**根本产生不了**逐项相关 |
| **82.5 → 98.5 → 100%** | 一致率按 \|Δtotal\| 分桶 | 常数预测器没有 margin，做不出单调性。证明分数的**大小**有意义，不只是符号 |

## The one number to never quote

**MIE 与人类多数票的一致率：pp2 89.1%、pp1 99.0%。**

两批人评都极度一边倒，一个「永远选多数方」的常数预测器能拿 **92.3%** 和 **99.8%** —— 它在两批上都赢过 MIE。

| batch | MIE | always-majority | MIE − baseline |
|---|---|---|---|
| pp1 | 99.0% | 99.8% | **−0.8 pp** |
| pp2 | 89.1% | 92.3% | **−3.2 pp** |

swx9 的置信度是 5，他会算这个基线。上面三个数字之所以全部用 AUC 和逐维相关表达，就是为了这一点。

（可以引用的相关数字：MIE 在**人类偏好少数方**的 41 项 pp2 item 上救回 **24.4%**，而常数基线在那里按定义恒为 0%。）

---

## Reproduce

无需 GPU、无需模型权重、无需联网，CPU 几秒：

```bash
python3 analyze_mie_vs_human.py --scores data/mie_scores.json \
    --pp1 data/anno_pp1.csv --pp2 data/anno_pp2.csv --drop_split \
    --out results/mie_vs_human_dropsplit.json
```

`--drop_split` 丢弃两位标注员偏好不一致的项（pp1 25 项，pp2 112 项）。这条规则对**偏好类**指标没有任何影响（两人时不一致即平票，本来就被排除），只影响逐维相关，幅度 ≤ 0.04。去掉该 flag 即可对比，两份结果都在 `results/`。

## Layout

```
README.md                 本文件
FINAL_REPORT.md           完整结果
REVIEWER_MAPPING.md       答了哪些 review 点
REBUTTAL_PLAN.md          17 条诉求分类 + 行动计划
analyze_mie_vs_human.py   分析脚本（唯一的计算入口）
data/
  anno_pp1.csv            nano_banana vs MOSAIC，1296 票
  anno_pp2.csv            Flux.2 vs GPT-Image-1.5，1296 票
  mie_scores.json         MIE 对全部 2586 个 cell 的打分
results/
  mie_vs_human_dropsplit.json   报告里的全部数字（机器可读）
  mie_vs_human.json             同上，不施加 drop 规则
archive_partial/          早先的部分导出及其结果 —— 仅作溯源，勿引用
```

## Caveats a co-author should know before quoting anything

1. **Appearance 维度测不了。** 标注员之间的 κ 只有 0.19–0.34，人自己都不一致，
   任何相关系数都被这个上限压死。这是标注协议的局限，**不是** MIE 的失败，也
   **不能**拿来当挡箭牌。原样说出来。
2. **MIE 高估 Flux.2 的 existence**：0.829 对人类的 0.562，差 +0.267，而且是四个
   生成器里唯一一个 existence 相关掉下来的（0.528 vs 其余 0.85–0.94）。这是最容易
   被抓的点，主动交代。
3. **pp2 有 17% 的项被丢弃**（112/648，两人偏好分歧）。每项只有 2 个标注员，没有
   第三方打破平局。
4. **两批都不是势均力敌的对局**（pp2 92/8，pp1 99.8/0.2），所以 AUC 的 CI 比理想
   情况宽。最高价值的后续是找一对人类接近 50/50 的组合 —— 按 MIE 自己的打分，
   `gpt15 vs nano_banana` 是最接近的（62/38，三种参考源下一致）。
