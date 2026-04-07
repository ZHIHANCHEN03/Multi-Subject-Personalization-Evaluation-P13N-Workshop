# PrismBench & LENS：数据与模型架构

本文档定义了 **PrismBench** 数据集的正式数据结构，该数据集旨在训练用于多主体个性化评估的 **LENS** (Localized Entanglement Navigation and Scoring) 指标模型。

> **🎯 为什么需要 PrismBench 和 LENS？**
> 现有的评估体系在多主体场景下已完全失效。**CLIP** 存在严重的“词袋效应”，它能认出“钢铁侠”和“美国队长”的元素，但无法区分主体间的动作错位（Swapping）与特征泄漏（Bleeding）；**DINO** 难以应对极端密度（$N \ge 4$）下的主体同质化与实体崩溃（Collapse）。
> LENS 的设计初衷，就是通过一套严密的 MECE 诊断分类树，精准打击 CLIP 和 DINO 的评估盲区，重新定义多主体生成的 Benchmark 标杆。

## 1. 数据组成 (Data Composition)
为了将 LENS 训练为行业标准的诊断型指标模型，PrismBench 采用了以下规模和策略：

- **银集 (Silver Set, 自动打标训练集)**: 约 **100,000 (10w)** 个图像对。
  - 数据引擎：使用 **GPT** 自动生成多主体描述与组合 Prompt，并生成对应的 Reference Images。
  - 生成模型对决：好的图片来自 **[Gemini (Nano Banana 2)](https://gemini.google/overview/image-generation/)**，差的图片来自 **[MOSAIC](https://github.com/bytedance-fanqie-ai/MOSAIC)**。
  - 由高级 AI 教师 (例如 `Qwen3.5-35B-A3B-FP8`) 进行 VLM 伪标签打标。
  - 主体数量分布：$N \in \{2, 4, 6, 8\}$（剔除单主体，专注特征纠缠）。
- **金集 (Golden Set, 人工打标测试/验证集)**: 约 **10,000 (1w)** 个图像对。
  - 由人类专家严格标注。
  - 同样覆盖 $N \in \{2, 4, 6, 8\}$ 的极端密度失败场景，用于验证 LENS 模型的 Zero-Shot（零样本）鲁棒性。

## 2. 图像预处理 (拼接 Stitching)
与其将多个独立的图像分别喂给 VLM，我们将它们**拼接 (stitch)**成一个单一的网格。
- **参考图像 (Reference Images)**: 直接来源于 MOSAIC 开源的 SemAlign-MS-Subjects200K 数据集。纯白背景（以隔离身份特征）。
- **生成图像 (Generated Images)**: 由 Prompt 驱动的复杂背景（用于测试空间注意力的特征纠缠）。

**拼接格式：**
```text
+-------------------+-------------------+
|                   |                   |
|   Reference A     |   Reference B     |
|                   |                   |
+-------------------+-------------------+
|                   |                   |
|   Generated Img 1 |   Generated Img 2 |
|                   |                   |
+-------------------+-------------------+
```

## 3. JSON 标签格式 (孪生网络成对学习)

### 3. JSON 标签格式 (孪生网络成对学习)

### JSON Schema 结构：
```json
[
  {
    "task_id": "0001",
    "prompt": "A photo of [Subject A], [Subject B] walking in a cyberpunk city.",
    "subject_count": 2,
    "stitched_image_path": "./data/images/stitch_0001.jpg",
    "preference_score_A": 0.9, 
    "preference_score_B": 0.2,
    "category_scores_A": {
      "class_4_collapse": 0.0,
      "class_3_swapping": 0.0,
      "class_2_bleeding": 0.5,
      "class_1_misalignment": 0.0
    },
    "category_scores_B": {
      "class_4_collapse": 1.0,
      "class_3_swapping": 0.0,
      "class_2_bleeding": 0.0,
      "class_1_misalignment": 0.0
    },
    "metadata": {
      "source": "GPT Automated Subject Generation"
    }
  }
]
```

### 层级化分类体系与三级打分制 (3-Tier Scoring)：

这个严格的决策树保证了 MECE (相互独立，完全穷尽) 的分类。为了解决模棱两可的生成情况，我们不使用非黑即白的二分类，而是采用 **三级打分制**：
*   **`1.0` (是 / Yes)**: 存在明确的该类错误。
*   **`0.5` (可能 / Maybe)**: 存在轻微瑕疵、部分遮挡难以判断。
*   **`0.0` (否 / No)**: 完全没有该类错误。

按以下顺序评估：

1. **Class 4 - 实体崩溃 (Entity Collapse)**: 画面中是否恰好存在请求的 $N$ 个不同的主体？ (如果“否” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：DINOv2, YOLO (目标检测), SCR (你的CVPR指标)。
2. **Class 3 - 语义错位 (Semantic Swapping / Misbinding)**: 核心身份是否被分配了属于彼此的错误的动作/角色？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：VQA (视觉问答), T2I-CompBench。
   - **核心防线**：强调“元素生成出来了，但给错人了”。
3. **Class 2 - 特征泄漏 (Attribute Bleeding)**: 核心身份和动作是否正确，但局部特征（颜色、配饰、肢体特征）在主体之间泄漏？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：**目前学术界空白**（这正是 LENS 最大的独家贡献）。
4. **Class 1 - 文本遗漏与语境不对齐 (Prompt Misalignment / Omission)**: 主体和特征没串味，但生成的动作是否彻底丢失了（如要A吃饭但画面没饭），或者全局背景/画风忽略了 Prompt 的要求？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：CLIP Score (CLIP-T), ImageReward, PickScore。
   - **核心防线**：强调“根本没生成出来”的文本截断或遗忘。
5. **Class 0 - 完美对齐 (Perfect Alignment)**: 上述四项得分全为 `0.0`，即为完美对齐。
   - **对标指标**：Human Preference Score (人类主观评价)。

## 4. 多任务学习 (MTL) 架构
LENS 使用孪生网络 (Siamese Network)，让 Image A 和 Image B 独立通过一个共享的 VLM backbone。

1. **排序损失 (Score Head, 分数头)**: 使用 `preference_label` 来拉高更受偏好图像的分数。
2. **诊断损失 (Classification Head, 分类头)**: 使用 `classification_label` 强制 backbone 的交叉注意力机制（cross-attention）明确聚焦在主体边界上。**这起到了强大的正则化作用，防止模型像 CLIP 那样走“虚假的背景捷径”。**