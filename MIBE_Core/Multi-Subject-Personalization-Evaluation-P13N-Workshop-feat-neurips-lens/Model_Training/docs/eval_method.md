# LENS 评测方法与论文证明路径

本文档的目标不是罗列一堆指标名，而是回答一个更核心的问题：

**如何系统性地证明 LENS 比现有指标更适合多主体个性化生成评估，并让论文具备冲击 NeurIPS 的说服力。**

---

## 1. 核心主张

论文必须围绕以下 3 个主张展开：

1. **现有指标在多主体个性化生成任务上存在结构性盲区。**
2. **LENS 与人类偏好对齐得更好。**
3. **LENS 不只是会打分，还能输出可解释的错误诊断。**

换句话说，论文的贡献不应该被表述为：

- “我们又微调了一个 Qwen 模型”

而应该被表述为：

- “我们发现现有评测系统在多主体个性化生成上失效，并提出了一个新的 benchmark + evaluator 组合来修复这个评价危机。”

---

## 2. 要证明什么

要让 Reviewer 相信 LENS 足够强，至少要证明以下 4 点：

1. **Human Alignment 更高**
2. **Error Diagnosis 更强**
3. **在主体数增加时退化更慢**
4. **在计算成本上可实际部署**

这 4 点分别对应论文中的 4 组实验。

---

## 3. Baseline 设计

Baseline 不能只选弱的，否则 Reviewer 会认为结论不可信。建议分成 4 类：

### 3.1 文本-图像对齐类

- `CLIP Score`
- `PickScore`
- `ImageReward`

作用：

- 证明“只看 prompt + gen”的指标在多主体 identity / bleeding / swapping 上不够。

### 3.2 参考图-生成图相似类

- `DINOv2`
- `ArcFace`（主要针对 human identity）

作用：

- 证明“只看 ref + gen”的指标也不够。
- `ArcFace` 是很重要的局部基线，因为它在人脸 identity 上很强，但它不理解 prompt，也无法评估 interaction。

### 3.3 文本规则 / VQA 类

- `TIFA`
- `T2I-CompBench++`

作用：

- 证明纯文本 faithful evaluator 对局部串色和身份绑定无能为力。

### 3.4 高成本 Oracle 类

- `GPT-4V / GPT-4o / Qwen-VL-Max` zero-shot

作用：

- 作为高成本上界，证明 LENS 的意义不是“超过所有闭源大模型”，而是：
- **以低成本、可批量、可解释的方式接近甚至部分达到高成本 VLM 的判断能力。**

---

## 4. 主实验指标

## 4.1 主指标：Preference Accuracy

定义：

- 在测试集上，对每个 pair 预测哪张图更好；
- 判断该预测是否与人类偏好一致。

这是最核心的主指标。

建议主表至少包含：

- `Preference Accuracy`
- `Kendall tau`
- `Spearman rho`

说明：

- 如果当前 test set 主要是 pairwise 二选一，`Preference Accuracy` 最直观。
- 如果后续有连续偏好分数或更复杂排序，可以补充 `Kendall tau` / `Spearman`。

### 为什么这个指标重要

因为 LENS 首先必须证明自己是一把“更接近人类判断的尺子”。

如果连这个都不比 CLIP / DINO 好，那么诊断再漂亮也很难说服 Reviewer。

---

## 4.2 诊断指标：Category Accuracy / F1

对每张生成图分别预测 3 个维度：

- `Existence`
- `Appearance`
- `Interaction`

建议汇报：

- `Existence Accuracy`
- `Appearance Accuracy`
- `Interaction Accuracy`
- `Macro Accuracy`
- `Macro F1`
- `Exact Match Accuracy`

### 为什么这些指标重要

这是 LENS 和大多数 black-box metric 最大的差异点。

CLIP、DINO、ImageReward 只能告诉你“好不好”，但不能告诉你：

- 是不是少人了
- 是不是串色了
- 是不是动作/角色交换了

而 LENS 可以把错误拆成正交维度，这正是论文的独特卖点。

---

## 4.3 严格指标：Exact Match

定义：

- 一张图的 3 个维度全都预测正确，才算这张图正确。

这比单维 accuracy 更严格。

### 为什么值得报告

因为它能证明 LENS 不是“在每个维度上碰巧猜对一些”，而是能给出整体一致的诊断结果。

---

## 5. 最重要的分析实验

## 5.1 Subject Count Scaling

按主体数量分组：

- `N=2`
- `N=4`
- `N=6`
- `N=8`

对每组分别报告：

- `Preference Accuracy`
- `Macro F1`

### 预期故事

- 在 `N=2` 时，传统指标可能还凑合；
- 在 `N=6/8` 时，CLIP / DINO / ArcFace 的缺陷会迅速暴露；
- LENS 退化更慢。

这张图最能说明：

**LENS 解决的不是普通 T2I 评价，而是多主体复杂组合下的系统性失效。**

---

## 5.2 Error-Type Breakdown

按错误类型拆：

- `Existence`
- `Appearance`
- `Interaction`

比较各 baseline 在不同错误类型上的表现。

### 预期解释

- `CLIP`：可能对全局语义还行，但对 `Appearance` 和 `Interaction` 弱
- `DINOv2`：对局部视觉相似有帮助，但不理解语义关系
- `ArcFace`：对 face identity 强，但无法处理 object、interaction 和多主体关系
- `LENS`：因为联合使用 `prompt + refs + gen`，并用 diagnostic supervision 训练，所以三项最均衡

这张表能把“为什么 LENS 更强”讲清楚，而不是只给一个总体分数。

---

## 5.3 Ratio Type / Scene Type Breakdown

按数据集属性分组：

- `all_human`
- `all_object`
- `equal`
- `human_heavy`
- `object_heavy`

如果还有 scene type，也可以按：

- `neutral`
- `occlusion`
- `interaction`

### 为什么重要

这能证明 LENS 的优势不是局限在某一种特定分布里。

比如：

- ArcFace 在 `all_human` 可能强；
- 但在 `object_heavy` 就失效；
- LENS 如果在不同 composition 下都稳，就更有说服力。

---

## 6. 推荐的论文主表设计

## 表 1：Overall Comparison

列：

- CLIP
- DINOv2
- ArcFace
- PickScore / ImageReward
- GPT-4V (可选)
- LENS

行：

- Preference Accuracy
- Kendall tau
- Macro Category Accuracy
- Macro F1
- Exact Match

目的：

- 证明整体上 LENS 最接近人类，并且具备诊断能力。

## 表 2：By Subject Count

列：

- N=2
- N=4
- N=6
- N=8

行：

- 各模型的 Preference Accuracy

目的：

- 证明 LENS 在高主体数量下更稳。

## 表 3：By Error Type

列：

- Existence
- Appearance
- Interaction

行：

- 各模型 Accuracy / F1

目的：

- 证明 LENS 的优势来自真正的结构化诊断，而不是偶然的总体优势。

## 图 1：Capacity Decay Curve

X 轴：

- subject count

Y 轴：

- Preference Accuracy 或 Macro F1

目的：

- 用一张最直观的图展示现有指标在复杂场景下崩塌，而 LENS 退化更慢。

## 图 2：Diagnostic Error Distribution

可以用堆叠柱状图或雷达图表示：

- 每个模型在哪类错误上失败最多

目的：

- 提供可解释的机制分析。

---

## 7. ArcFace 在论文里该怎么放

ArcFace 很值得加，但不要把它写成“总 baseline”，而要写成：

- **identity-specialized baseline**

你要用 ArcFace 来证明：

- 即便一个指标在“人脸 identity”上很强，
- 它仍然不能替代一个同时理解 `prompt + refs + gen` 的 evaluator。

ArcFace 的局限：

- 主要适用于 human face
- 不适用于 object-heavy 数据
- 不理解 prompt
- 不评估 interaction
- 对多人遮挡、姿态变化、风格化生成较脆弱

所以 ArcFace 是一个很好的“强局部基线”，但它反而能帮助你证明：

**单维 identity metric 不足以完成多主体个性化生成评估。**

---

## 8. 如何论证 LENS 比其他指标“更好”

论文里不要只写：

- “LENS 的 accuracy 更高”

要写成：

1. **CLIP / PickScore / ImageReward 的失败原因**
   - 只看 `prompt + gen`
   - 容易受 bag-of-words shortcut 影响
   - 无法验证 reference identity

2. **DINOv2 / ArcFace 的失败原因**
   - 只看 `ref + gen`
   - 不理解 prompt
   - 难以评估 interaction / role binding

3. **LENS 的优势来源**
   - 同时使用 `prompt + refs + gen`
   - 用 pairwise preference 学会“谁更好”
   - 用 category supervision 学会“为什么更好”

也就是说，**LENS 的优势必须被解释为架构与监督设计带来的系统性优势，而不是参数偶然更多。**

---

## 9. Silver + Golden Dataset 在论文里的作用

如果你只发一个 evaluator，论文说服力有限。  
真正让它有 NeurIPS 级别潜力的是：

- **Silver Set**
  - 大规模
  - 自动标注
  - 提供训练可扩展性

- **Golden Set**
  - 纯人工
  - 高质量
  - 提供可信测试基准

这个组合的贡献是：

1. 给社区一个新的 benchmark
2. 给社区一个新的 evaluator
3. 给社区一个新问题定义：多主体个性化生成的评价危机

也就是说，你的论文不是单点贡献，而是：

- `Dataset + Evaluation Protocol + Diagnostic Metric`

这比只发模型更像 Datasets & Benchmarks 赛道会喜欢的工作。

---

## 10. 做完这些是否足够说明 LENS 很强

如果你把以下内容都做出来：

1. 和 CLIP / DINOv2 / ArcFace / PickScore / GPT-4V 做完整对比
2. 在 Preference Accuracy 上显著更强
3. 在 category diagnosis 上有清晰优势
4. 在 `N=2/4/6/8` 上展示退化曲线
5. 在 error type 上做可解释分析
6. 证明 Silver + Golden 的数据集构造是干净的、无数据泄漏的

那么你就**足够证明 LENS 是一个有实质价值的新评测指标**。

注意，这不等于“自动稳中”，但已经足够构成一篇强论文的实验核心。

---

## 11. 能不能中 NeurIPS

### 11.1 可以有冲击力的条件

这篇工作有 NeurIPS 潜力，如果满足：

1. **问题真实且重要**
   - 多主体个性化生成确实是现有评价体系盲区

2. **基线完整**
   - 不是只对比弱 baseline

3. **实验闭环**
   - train / val / test 清楚
   - metric 清楚
   - ablation 清楚
   - 分析清楚

4. **结果显著**
   - 不只是比基线高 1-2 个点
   - 而是系统性优于现有指标

5. **故事讲得像评价危机，而不是模型工程**
   - “现有尺子坏了，我们提出新尺子”
   - 而不是“我们又训了一个 Qwen”

### 11.2 是否稳中

不能说“稳中”。

因为 NeurIPS 是否接收，除了想法，还取决于：

- 实验是否完整
- 实现是否稳定
- 结果是否显著
- 论文写作是否足够锋利
- Reviewer 是否认同这是一个足够普适的问题

### 11.3 是否有 oral / best paper 潜力

只有在以下情况下才有资格谈 oral 甚至 best paper：

1. 结果对现有指标形成**明确压制**
2. Benchmark 被证明填补了真实社区空白
3. LENS 的诊断能力非常有视觉冲击力
4. 实验图表和 case study 极其强
5. 论文叙事从“一个 metric”上升到“评价范式更新”

所以更准确的说法是：

- **有 oral / best paper 的理论潜力**
- **但这取决于实验结果是否真的形成“范式级”证据**

---

## 12. 最后一句话

如果这篇论文最终能成立，它最强的一句话不是：

- “LENS 比 CLIP 高了几个点”

而是：

- **在多主体个性化生成这个重要子问题上，社区现有的评测尺子是坏的，而 PrismBench + LENS 提供了一套更接近人类判断、可解释、可扩展的新评价协议。**

这才是最有希望打动 NeurIPS Reviewer 的叙事核心。
