# PrismBench & LENS：数据与模型架构

本文档定义了 **PrismBench** 数据集的正式数据结构，该数据集旨在训练用于多主体个性化评估的 **LENS** (Localized Entanglement Navigation and Scoring) 指标模型。

> **🎯 为什么需要 PrismBench 和 LENS？**
> 现有的评估体系在多主体场景下已完全失效。**CLIP** 存在严重的“词袋效应”，它能认出“钢铁侠”和“美国队长”的元素，但无法区分主体间的动作错位（Swapping）与特征泄漏（Bleeding）；**DINO** 难以应对极端密度（$N \ge 4$）下的主体同质化与实体崩溃（Collapse）。
> LENS 的设计初衷，就是通过一套严密的 MECE 诊断分类树，精准打击 CLIP 和 DINO 的评估盲区，重新定义多主体生成的 Benchmark 标杆。

## 1. 数据组成 (Data Composition)
为了将 LENS 训练为行业标准的诊断型指标模型，同时确保在学术研究中的最高投资回报率 (ROI)，PrismBench 采用了以下“甜点级 (Sweet Spot)” 规模和策略：

- **总数据量**: 最终约 **53,000 (5.3w)** 个图像对。
- **银集 (Silver Set, 自动打标训练集)**: 初始约 **65,000 (6.5w)** 个图像对，过滤后保留 **50,000 (5w)** 个图像对。
  - 数据引擎：采用多模型协作。由 **GPT-4o (DALL-E)** 负责生成高质量、纯白背景的主体 Reference Images，并由 **Claude** 负责撰写连贯、高难度的场景 Prompt。
  - 生成模型对决 (控制变量配对)：固定使用 **[Gemini (Nano Banana 2)](https://gemini.google/overview/image-generation/)** 作为强模型锚点生成“好图”，使用 **MOSAIC** 等特定策略基线作为弱模型锚点生成“差图”。初始构建约 **65,000 对** 候选图像对，经过严格漏斗清洗后保留 **50,000 对**。
  - 由高级 AI 教师 (例如 `Qwen-VL-Max` 或 `GPT-4o`) 结合 CoT (思维链) 与硬锚点清洗机制进行高质量 VLM 伪标签打标。
  - 主体数量分布：$N \in \{2, 4, 6, 8\}$（剔除单主体，专注特征纠缠）。
- **金集 (Golden Set, 人工打标测试/验证集)**: 初始约 **4,000 (4k)** 个图像对，过滤后保留 **3,000 (3k)** 个图像对。
  - 由领域专家采用 **双盲标注 (Double-Blind) + 冲突仲裁 (Tie-breaker)** 的机制严格标注，彻底消除自动化偏见 (Automation Bias)。
  - 同样覆盖 $N \in \{2, 4, 6, 8\}$ 的极端密度失败场景，用于验证 LENS 模型的 Zero-Shot（零样本）鲁棒性。

## 2. 图像预处理 (拼接 Stitching vs. 独立推理)

在 LENS 的生命周期中，**训练 (Train) 阶段**和**推理/评估 (Inference) 阶段**对输入图像的格式要求是**完全不同**的。这种设计是为了兼顾“大规模自动打标的成本”与“模型实际应用的灵活性”。

### 2.1 训练阶段 (Train / Data Annotation) 的输入：Stitched Grid
在生成 10w 条训练集（给 Teacher VLM 打标）时，我们采用的是**拼接网格 (Stitched Grid)** 格式。
*   **为什么这么做？** 
    1. 节约大模型 API Token 成本：将 Reference 和两张生成的图拼在一起，一次 API 调用就能得出所有分数。
    2. 强化对比学习：Teacher VLM 可以在同一张大图里直接对比好图（Image A）和差图（Image B）的细节差异。
*   **网格像素要求 (Grid Resolution)：**
    为了让 VLM 能够清晰地辨认局部特征泄漏（Appearance）和主体缺失（Existence），单个子图必须统一为 `512x512` 像素。
    - **N=2 时**: 采用 **2x2 Grid (1024x1024 px)**，对应 `2 张 Reference + 2 张 Generated Image`。
    - **N=4 时**: 采用 **2 列 x 3 行网格 (1024x1536 px)**，对应 `4 张 Reference + 2 张 Generated Image`。
    - **N=6 时**: 采用 **2 列 x 4 行网格 (1024x2048 px)**，对应 `6 张 Reference + 2 张 Generated Image`。
    - **N=8 时**: 采用 **2 列 x 5 行网格 (1024x2560 px)**，对应 `8 张 Reference + 2 张 Generated Image`。
    - **统一原则**: 始终固定每个子图为 `512x512`，按“先排完全部 Reference，再排 Image A / Image B”的顺序进行纵向扩展。

*(示例：N=2 时的基础网格 1024x1024)*
```text
+-------------------+-------------------+
|   512x512 px      |   512x512 px      |
|   Reference A     |   Reference B     |
+-------------------+-------------------+
|   512x512 px      |   512x512 px      |
|   Generated Img 1 |   Generated Img 2 |
+-------------------+-------------------+
```

### 2.2 推理/评估阶段 (Inference) 的输入：独立图像序列 (Sequential Input)
当 LENS 训练完成，作为一个开源评估工具发布在 Hugging Face 给社区使用时，我们**不强制要求**用户把图片拼成 Grid！
*   **输入格式：** 列表形式的独立图片 `[Ref_1, Ref_2, ..., Ref_N, Generated_Image]`。
*   **为什么这么做？**
    1. **解耦评估**：在实际应用中，用户通常是拿 LENS 来评估单张生成的图片，而不是成对对比。
    2. **用户友好 (User-Friendly)**：如果要求开源社区每次评测前还要写复杂的 Python PIL 代码去拼长图，这会极大增加工具的使用门槛。
*   **内部处理逻辑 (In LENS Forward Pass)**：
    当用户传入一个图片列表时，LENS 模型内部会自动利用 Qwen3.5-9B 的 Multi-Image 处理能力，将它们编码为独立的视觉 Token 序列：
    `<image> (Ref 1) <image> (Ref 2) ... <image> (Generated) + Text Prompt`
    然后，LENS 附加的 MLP Heads（Score Head 和 Classification Head）会直接输出针对这一张 Generated Image 的打分和 3D 诊断向量 (Existence, Appearance, Interaction)。

## 3. JSON 标签格式 (孪生网络成对学习)

### 3. JSON 标签格式 (孪生网络成对学习)

这里有一个必须严格保持的术语约定：**`subject` 指参考主体身份（Reference Subject）**，**`image` 指生成图像**。因此：
- `subject_refs` 是参考主体列表；
- `image_A_path` / `image_B_path` 是两张待比较的生成图像；
- `annotator_results` 保存每位标注员对 **Image A / Image B** 的原始打分结果；Silver Set 通常有 2 条 VLM Teacher 结果，Golden Set 通常有 2 条人类双盲结果；
- 在任意一位 annotator 内部，`category_scores_A` / `category_scores_B` 都直接就是 **Image A / Image B 这两张生成图像** 的最终 3 维诊断分数。

### JSON Schema 结构：
```json
[
  {
    "task_id": "0001",
    "prompt": "A photo of [Subject A], [Subject B] walking in a cyberpunk city.",
    "subject_count": 2,
    "subject_refs": [
      {
        "id": "Subject A",
        "image_path": "./data/refs/cat_01.jpg"
      },
      {
        "id": "Subject B",
        "image_path": "./data/refs/dog_02.jpg"
      }
    ],
    "image_A_path": "./data/generated/gemini_0001.jpg",
    "image_B_path": "./data/generated/mosaic_0001.jpg",
    "annotator_results": [
      {
        "annotator_id": "teacher_vlm_01",
        "preference": "A",
        "category_scores_A": {
          "existence": 1,
          "appearance": 0,
          "interaction": 0
        },
        "category_scores_B": {
          "existence": 0,
          "appearance": 0,
          "interaction": 0
        }
      },
      {
        "annotator_id": "teacher_vlm_02",
        "preference": "A",
        "category_scores_A": {
          "existence": 1,
          "appearance": 0,
          "interaction": 1
        },
        "category_scores_B": {
          "existence": 0,
          "appearance": 0,
          "interaction": 0
        }
      }
    ],
    "metadata": {
      "source": "GPT Automated Subject Generation"
    }
  }
]
```

### 层级化二元分类体系 (Binary Cascading Taxonomy)：

为了将人类认知负荷降到最低，并为模型提供最干净的正交梯度，我们采用了**极致的二元打分制 (1=Pass, 0=Fail)**：
*   **`1` (Pass / 完美)**: 在该维度上没有发现任何瑕疵。
*   **`0` (Fail / 有瑕疵)**: 只要存在该维度的瑕疵（无论是轻微还是严重），一律判定为失败。

**独立正交评估机制 (Independent Orthogonal Evaluation)**：
标注采用完全独立的评估逻辑。**这三个维度相互解耦，无论前一个维度是否失败，都需要继续对整张生成图像完成后续维度的判断。**
*   *理论依据*：即使生成图像已经出现主体缺失（Existence=0），我们仍然可以继续判断这张图像在外观保真度（Appearance）和交互语义（Interaction）上是否也存在问题。这最大程度保留了整张图像的诊断信息。

按以下维度独立评估：

1. **Existence (存在性 - 无缺失/无克隆)**: 画面中是否恰好存在请求的 $N$ 个**独立的**核心参考主体？
   - **`1` (Pass)**: 人数完美，没有遗漏，没有克隆。
   - **`0` (Fail)**: 明确少人、同质化克隆、严重遮挡导致无法辨认。
2. **Appearance (独立外观 - 无畸变/无串色)**: 对整张生成图像进行判断：所有请求的参考主体在图中是否都保持了正确的物理结构，并且**没有**发生衣服/颜色/材质的局部特征串染 (Attribute Bleeding)？
   - **`1` (Pass)**: 结构完美，颜色纯净，没有任何特征泄漏。
   - **`0` (Fail)**: 肢体变异、脸部融化、明确的局部串色（美队衣服染了红色）、轻微比例失调。
3. **Interaction (交互对齐 - 关系与语义无误)**: 对整张生成图像进行判断：参考主体之间的关系、动作和道具归属是否完全符合 Prompt？
   - **`1` (Pass)**: 交互完美，动作完全符合文本描述。
   - **`0` (Fail)**: 动作张冠李戴（A骑马变成B骑马）、衣服互换、动作遗漏（变成木头人）、次要文本道具丢失。

### 终局偏好选择 (Binary Preference Choice)
在完成诊断后，系统要求每位标注员给出一个简单的二元偏好选择：**Image A 更好 还是 Image B 更好？**（即 `preference: "A"` 或 `preference: "B"`）。这些原始结果会直接存入 `annotator_results`；Silver Set 通常包含 2 条 VLM Teacher 标注结果，而 Golden Set 包含 2 条人类双盲标注结果。只有当标注结果足够一致时，样本才会进入最终可训练集合。

## 4. 多任务学习 (MTL) 架构
LENS 使用孪生网络 (Siamese Network)，让 Image A 和 Image B 独立通过一个共享的 VLM backbone。

1. **排序损失 (Score Head, 分数头)**: 使用 `preference` 标签 (`"A"` 或 `"B"`)，通过二元交叉熵或 Margin Ranking Loss 来训练 Reward Model，强制模型学习人类在两张图中的相对偏好。
2. **诊断损失 (Classification Head, 分类头)**: 使用通过一致性过滤后保留下来的 `annotator_results[*].category_scores_A/B` (3D Binary Vector) 作为训练监督，强制 backbone 明确聚焦在主体的存在、外观和交互上。对于 Silver Set，这意味着两路 Teacher 需要先达成足够一致；对于 Golden Set，这意味着两位人工标注员也需要先满足双盲一致性要求。通过标准的 BCE Loss (Binary Cross Entropy) 独立计算 3 个维度的损失，起到了强大的正则化作用。
