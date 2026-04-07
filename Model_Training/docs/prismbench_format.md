# PrismBench & LENS：数据与模型架构

本文档定义了 **PrismBench** 数据集的正式数据结构，该数据集旨在训练用于多主体个性化评估的 **LENS** (Localized Entanglement Navigation and Scoring) 指标模型。

> **🎯 为什么需要 PrismBench 和 LENS？**
> 现有的评估体系在多主体场景下已完全失效。**CLIP** 存在严重的“词袋效应”，它能认出“钢铁侠”和“美国队长”的元素，但无法区分主体间的动作错位（Swapping）与特征泄漏（Bleeding）；**DINO** 难以应对极端密度（$N \ge 4$）下的主体同质化与实体崩溃（Collapse）。
> LENS 的设计初衷，就是通过一套严密的 MECE 诊断分类树，精准打击 CLIP 和 DINO 的评估盲区，重新定义多主体生成的 Benchmark 标杆。

## 1. 数据组成 (Data Composition)
为了将 LENS 训练为行业标准的诊断型指标模型，PrismBench 采用了以下规模和策略：

- **银集 (Silver Set, 自动打标训练集)**: 约 **100,000 (10w)** 个图像对。
  - 数据引擎：采用多模型协作。由 **GPT-4o (DALL-E)** 负责生成高质量、纯白背景的主体 Reference Images，并由 **Claude** 负责撰写连贯、高难度的场景 Prompt。
  - 生成模型对决：最后由 **[Gemini (Nano Banana 2)](https://gemini.google/overview/image-generation/)** 生成“好图”，由 **[MOSAIC](https://github.com/bytedance-fanqie-ai/MOSAIC)** 等开源基线生成“差图”。
  - 由高级 AI 教师 (例如 `Qwen3.5-35B-A3B-FP8`) 进行 VLM 伪标签打标。
  - 主体数量分布：$N \in \{2, 4, 6, 8\}$（剔除单主体，专注特征纠缠）。
- **金集 (Golden Set, 人工打标测试/验证集)**: 约 **10,000 (1w)** 个图像对。
  - 由人类专家严格标注。
  - 同样覆盖 $N \in \{2, 4, 6, 8\}$ 的极端密度失败场景，用于验证 LENS 模型的 Zero-Shot（零样本）鲁棒性。

## 2. 图像预处理 (拼接 Stitching)
与其将多个独立的图像分别喂给 VLM，我们将它们**拼接 (stitch)**成一个单一的网格。
- **参考图像 (Reference Images)**: 由 **GPT-4o / DALL-E 3** 自动生成的高质量单主体图像。**纯白背景**（以隔离身份特征，防止背景干扰）。
- **生成图像 (Generated Images)**: 由 Claude 编写的复杂 Prompt 驱动，包含真实世界或风格化的**复杂背景**（用于测试空间注意力的特征纠缠和背景对齐能力）。

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
      "class_5_omission": 0.0,
      "class_4_distortion": 0.0,
      "class_3_swapping": 0.0,
      "class_2_bleeding": 0.5,
      "class_1_misalignment": 0.0
    },
    "category_scores_B": {
      "class_5_omission": 1.0,
      "class_4_distortion": 0.0,
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

1. **Class 5 - 核心实体缺失与同质化 (Subject Omission & Homogenization)**: 画面中是否恰好存在请求的 $N$ 个**独立的**核心参考主体？ (如果“否” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：DINOv2, YOLO (目标检测), SCR (你的CVPR指标)。
   - **核心防线**：仅清点“提供的参考主体”。文本里随口提的普通道具（如汉堡、剑）如果丢了，不在此类，归入 Class 1。
   - **示例**：
     - `1.0分`：明确少人（要参考主体猫和狗，只有狗，猫完全消失）；克隆（要参考主体A和B，生成了两个一模一样的A）；物种消失（要A人和B猫，画了两个人，猫没了）。
     - `0.5分`：严重遮挡（某主体只露出半只手或半张脸）；极度模糊无法确认身份。
2. **Class 4 - 实体结构扭曲与崩坏 (Subject Distortion & Mutilation)**: 在主体存在的前提下，其基础生物/物理结构是否发生了严重扭曲或畸变？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：FID, IS (Inception Score), BRISQUE。
   - **核心防线**：专门惩罚把人画成“怪物”的模型。不包含串色问题（Class 2）。
   - **示例**：
     - `1.0分`：肢体变异（长了三个胳膊、六根手指且极其明显）；身体残缺（只有一个悬空的头，没有身干）；脸部如融化的蜡像。
     - `0.5分`：轻微比例失调（腿异常短小）；身体连接处轻微不对齐。
3. **Class 3 - 语义错位 (Semantic Swapping / Misbinding)**: 核心身份是否被分配了属于彼此的错误的动作/角色？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：VQA (视觉问答), T2I-CompBench。
   - **核心防线**：强调“元素生成出来了，但给错人了”。
   - **示例**：
     - `1.0分`：动作给错（要A骑马B牵马，变成B骑马A牵马）；衣服穿错（要A穿红B穿蓝，变成A蓝B红）；道具拿错（要A拿剑，变成B拿剑）；位置关系互换（A在桌上B在椅上，反过来了）。
     - `0.5分`：动作不标准（要“背靠背”，但看起来像“并排站”）；道具归属不清（剑放两人中间，看不出谁拿）。
4. **Class 2 - 特征泄漏 (Attribute Bleeding)**: 核心身份和动作是否正确，但局部特征（颜色、配饰、肢体特征）在主体之间泄漏？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：**目前学术界空白**（这正是 LENS 最大的独家贡献）。
   - **示例**：
     - `1.0分`：颜色渗透（A的红衣服在B的蓝衣服上染了一大块红斑）；肢体融合（握手时手部长成一团带有双人肤色的肉块）；配饰传染（A戴眼镜，B没戴，但B脸上有眼镜框）。
     - `0.5分`：接缝处颜色轻微渗透；疑似环境光影反射导致的模糊串色。
5. **Class 1 - 文本遗漏与动作不对齐 (Prompt Misalignment / Omission)**: 主体和特征没串味，但 Prompt 要求的核心动作、交互或**纯文本指定的普通道具（非定制主体）**是否彻底丢失了？ (如果“是” $\rightarrow$ 得分 `1.0` 或 `0.5`)。
   - **对标指标**：CLIP Score (CLIP-T), ImageReward, PickScore。
   - **核心防线**：核心人物都完美在场，专门惩罚**动作的无视**和**非参考道具的遗漏**。
   - **示例**：
     - `1.0分`：动作彻底遗漏（要求“A和B握手”，但两人只是毫无接触地并排站着，变成木头人）；纯文本道具缺失（要求“A吃汉堡”，A在场但画面里根本没有汉堡，A空手站着）；状态丢失（要求“A躺在地上睡觉”，但A是睁眼站着的）。
     - `0.5分`：动作含糊（要求“A吃汉堡”，A手里拿着汉堡但并没有做出“吃”的动作，仅持有）；次要道具遗漏（要求“A戴着帽子弹吉他”，吉他在但帽子没画）。
6. **Class 0 - 完美对齐 (Perfect Alignment)**: 上述五项得分全为 `0.0`，即为完美对齐。
   - **对标指标**：Human Preference Score (人类主观评价)。
   - **示例**：要求“钢铁侠和美国队长在握手，钢铁侠左手拿着公文包”，画面中两人（N=2）都完美呈现，未畸变，衣服颜色独立无污染，正在握手，且钢铁侠拿着包。完美达成目标。

## 4. 多任务学习 (MTL) 架构
LENS 使用孪生网络 (Siamese Network)，让 Image A 和 Image B 独立通过一个共享的 VLM backbone。

1. **排序损失 (Score Head, 分数头)**: 使用 `preference_label` 来拉高更受偏好图像的分数。
2. **诊断损失 (Classification Head, 分类头)**: 使用 `classification_label` 强制 backbone 的交叉注意力机制（cross-attention）明确聚焦在主体边界上。**这起到了强大的正则化作用，防止模型像 CLIP 那样走“虚假的背景捷径”。**