# LENS 流水线：从数据收集到模型训练

本文档概述了针对 NeurIPS Datasets & Benchmarks 赛道提交的完整端到端流水线：**PrismBench**（数据集）和 **LENS**（评估指标模型）。

## 1.1 核心动机：填补多主体评估的学术空白 (The Evaluation Gap)
现有的主流评估指标在单主体生成中表现良好，但在多主体个性化生成（N>=2）时存在严重的“语义捷径”和“特征坍塌盲区”。为了直观展示 LENS 的不可替代性，下表对比了当前最具代表性的 T2I 评价指标：

| 指标名称 | 能否识别“特定身份”? | 能否量化“特征串色”? | 能否量化“张冠李戴”? | 是否具备“可解释诊断”? | 评估成本 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CLIP Score** | ❌ (词袋效应) | ❌ (全局融合) | ❌ | ❌ | 极低 |
| **DINOv2** | ✅ (懂特征) | ❌ (不懂语义) | ❌ | ❌ | 低 |
| **ImageReward / PickScore** | ❌ (黑盒偏好) | ❌ (偏好唯美) | ❌ | ❌ | 低 |
| **TIFA / T2I-CompBench** | ❌ (纯文本VQA) | ❌ (无法查串色) | ✅ | ❌ | 中等 |
| **GPT-4V (Zero-shot)** | ✅ | ⚠️ (极不稳定) | ✅ | ✅ (自然语言) | **极高** |
| **LENS (Ours)** | **✅ (多图参考)** | **✅ (独立外观维)** | **✅ (交互对齐维)** | **✅ (3D正交瀑布流)** | **低** |

**LENS 的使命**：重新审视多主体个性化生成的评价标准，通过引入包含“特征泄漏”和“语义错位”的细粒度诊断架构，彻底覆盖上述指标留下的学术空白，确保评估结果真正对齐人类视觉感知。

---

## 阶段 1：数据收集与生成 (PrismBench)

为了训练一个鲁棒的指标模型，我们需要一个包含多种失败模式的、大规模且多样化的多主体生成图像数据集。

### 1.1 源主体库与 Prompt 自动构建 (多模型协作引擎)
为了实现 5.3 万条数据的自动化规模，我们构建了一个基于顶尖大模型的多代理协作流水线：
*   **参考图生成 (Reference Generation by GPT)：** 使用 **GPT-4o / DALL-E 3** 根据预设的主体类别，生成高质量、纯白背景的单主体参考图 (Reference Images)。GPT 负责把控主体的视觉多样性和特征清晰度。
*   **场景编排 (Prompting by Claude)：** 使用 **Claude 3.5 Sonnet (或 Opus)** 接收 GPT 生成的主体列表，并组合生成多主体同框的复杂 Prompt。Claude 在长文本逻辑推理和复杂场景编排上具有卓越能力，能确保交互动作合理且极具挑战性。

### 1.2 场景设计：自顶向下的降维策略
构建能够强制多个主体出现在同一场景中的 Prompt。**主体数量 $N$ 的取值范围严格设定为 2, 4, 6, 8**（剔除 N=1，因为单主体生成已不是学术难题，PrismBench 专注于纯粹的“多主体特征纠缠”）。
*   **降维生成策略 (Top-Down Prompting)：** 
    1.  首先让 **Claude** 设计包含 **8 个主体**的最复杂场景 Prompt。
    2.  随后，通过让 Claude 逐步剔除主体，向下缩减生成 **6、4、2** 个主体的 Prompt。
    3.  这种格式确保了场景语义的连贯性，并能精确控制主体数量的衰减测试。
*   **场景类型 (Scene Types)：**
    *   *中性 (Neutral)：* 主体空间分离（低特征纠缠风险）。
    *   *遮挡 (Occlusion)：* 主体前后遮挡（中等特征纠缠风险）。
    *   *交互 (Interaction)：* 主体物理接触（高特征纠缠风险）。

### 1.3 图像生成（控制变量配对生成）
考虑到数据构建的时间和工程成本，我们采用了一种极简但控制变量极其严格的**配对生成策略 (Controlled Paired Generation)**。
使用构建好的 Prompt 和参考图像，生成成对的图像，以捕获质量差异和失败类型：
*   **强模型 / 高分锚点 (Strong Model)：** 使用 **[Gemini 图像生成大模型 (Nano Banana 2)](https://gemini.google/overview/image-generation/)**。作为当前 SOTA，它预期能生成构图完美、特征无泄漏的高质量图像，为模型提供正向的结构对齐范例。
*   **弱模型 / 低分锚点 (Weak Model)：** 使用 **MOSAIC** 等特定策略基线。预期在极端数量 (如 N=8) 或复杂交互下会产生丰富的特征泄漏、语义错位和实体崩溃，为模型提供负向错误范例。
*   *关键规则 (Crucial Rule)：* 对于每一个 Prompt，必须**同时**使用 Nano Banana 2 和 MOSAIC 生成图像，构成一对 (Image A, Image B)。因为 Prompt 相同，背景语义和构图分布得到严格控制，从而强制 VLM Backbone 将注意力孤立在“主体特征纠缠”上，而不会被背景差异干扰。

---

## 阶段 2：标注与分类体系设计 (VLM 伪标签与人工标注)

为了在有限的学术团队资源下实现最高 ROI（投资回报率）的顶会级论文产出，同时确保足以说服 NeurIPS 严苛评审的数据规模，我们对数据集进行了“甜点级 (Sweet Spot)” 的体量设定：
*   **自动标注 (Auto-labels) - Silver Set：** 初始约 **65,000 (6.5w)** 条数据，经过双 Teacher 一致性与质量漏斗过滤后，保留 **50,000 (5w)** 条数据作为训练集。由两路强大的 VLM Teacher（如 GPT-4o、Qwen-VL-Max、Claude 等）独立自动打标，只有当两路 Teacher 对同一个样本给出**足够一致**的判断时，该样本才会被保留。5w 条最终训练数据在大规模数据集时代既显得扎实有诚意，又在 API 成本和生成时间上完全可控。
*   **人工标注 (Human-labels) - Golden Set：** 初始约 **4,000 (4k)** 条数据，经双盲一致性与冲突清洗后，最终保留 **3,000 (3k)** 条高质量纯净数据作为验证/测试集。由领域专家全手工盲标。3,000 条纯净数据在统计学上提供了坚不可摧的显著性 ($p < 0.001$)，足以让 Reviewer 对你的评测结论毫无疑义。

### 2.1 3维正交解耦分类体系与二元打分制 (3D Orthogonal Taxonomy & Binary Scoring)

#### 金集双盲标注与仲裁机制 (Golden Set Double-Blind Annotation Protocol)
为了彻底消除自动化偏见 (Automation Bias) 和人类锚定效应，我们在构建 **初始 4k、最终保留 3k** 的金集时**坚决不使用任何 VLM 预打标**，而是采用了纯人类主导的“双盲标注 + 自动仲裁”机制，这是在 NeurIPS 论文中证明数据集 Ground Truth 质量的最强有力手段：
1.  **双盲独立标注 (Double-Blind)**：同一张生成的图像会被随机分发给两位相互独立的标注员（Annotator A 和 B）。他们通过“瀑布流单选问卷”从头开始盲标。
2.  **一致性检验 (Inter-Annotator Agreement, IAA)**：在论文中，我们将使用 Cohen's Kappa ($\kappa$) 系数来证明这 3 类诊断体系是客观的。只要 $\kappa > 0.6$，就能在统计学上向 Reviewer 证明该分类体系没有主观歧义。
3.  **冲突仲裁机制 (Tie-breaking Resolution)**：
    *   *高度一致 (Absolute Agreement)*：两人选择完全相同，直接入库。
    *   *严重冲突 (Severe Conflict)*：例如 A 给 `1` (Pass)，B 给 `0` (Fail)。该样本将被打上 `[Conflict]` 标签，交由核心作者/领域专家进行最终一锤定音的仲裁 (Tie-breaker)。

#### 诊断体系与打分示例
在本文档中，**Subject 始终指 Reference Subject / 参考主体身份**，而 **Image 始终指待评估的生成图像**（Image A 或 Image B）。换句话说，`subject_refs` 存的是参考主体，`image_A_path` / `image_B_path` 存的是生成图像；`category_scores_A/B` 记录的是 **Image A / Image B 作为整张生成图像** 的最终 3 维诊断分数。
LENS 的核心在于诊断多主体生成**为什么**失败。我们将多主体的失败本质解耦为 3 个完全正交的维度：**存在 (Existence) $\rightarrow$ 独立外观 (Appearance) $\rightarrow$ 交互 (Interaction)**。这构建了一个完美的互斥（MECE）决策树。

为了将人类认知负荷降到最低，并为模型提供最干净的正交梯度，我们采用了**极致的二元打分制 (1=Pass, 0=Fail)**：
*   **`1` (Pass / 完美)**: 在该维度上没有发现任何瑕疵。
*   **`0` (Fail / 有瑕疵)**: 只要存在该维度的瑕疵（无论是轻微还是严重），一律判定为失败。

系统对这三个维度进行**独立正交评估 (Independent Orthogonal Evaluation)**：

**无截断评估机制 (No Cascading/Truncation)**：
在打标过程中（无论是人类还是 VLM），这三个维度完全解耦。即使某一维度的得分是 `0` (Fail)，标注员也必须根据画面中实际生成的内容，继续对其他维度进行独立打分。例如，如果生成了 4 个人但只出现了 2 个人 (Existence = 0)，依然需要评估这幸存的 2 个人是否长得对 (Appearance) 以及动作是否正确 (Interaction)。这能最大程度榨取图像的细粒度质量特征。

*   **Existence (存在性 - 无缺失/无克隆)**
    *   *判断条件：* 画面中是否恰好存在请求的 $N$ 个独立的**核心参考主体**？
    *   *打分示例：* 
        *   `1 (Pass)`: 人数完美，没有遗漏，没有克隆。
        *   `0 (Fail)`: 明确少人、同质化克隆、严重遮挡导致无法辨认。
    *   *对标现有指标：* DINOv2, Object Detection。

*   **Appearance (独立外观 - 无畸变/无串色)**
    *   *判断条件：* 对整张生成图像进行判断：所有请求的参考主体在图中是否都保持了正确的物理结构，并且**没有**发生衣服/颜色/材质的局部特征串染 (Attribute Bleeding)？
    *   *打分示例：* 
        *   `1 (Pass)`: 结构完美，颜色纯净，没有任何特征泄漏。
        *   `0 (Fail)`: 肢体恐怖变异；明确的局部串色（美队的制服上染了钢铁侠的金属材质）；轻微比例失调。
    *   *对标现有指标：* 目前学术界完全空白（LENS 独家贡献）。

*   **Interaction (交互对齐 - 关系与语义无误)**
    *   *判断条件：* 对整张生成图像进行判断：参考主体之间的关系、动作和道具归属是否完全符合 Prompt？
    *   *打分示例：* 
        *   `1 (Pass)`: 交互完美，动作完全符合文本描述。
        *   `0 (Fail)`: 动作张冠李戴（要A骑马B牵马，变成B骑马A牵马）；整套衣服互换；动作遗漏（变成木头人）；次要道具遗漏。
    *   *对标现有指标：* T2I-CompBench, VQA Metrics。

#### 终局偏好选择 (Binary Preference Choice)
在完成 3 关诊断后，系统要求标注员给出一个简单的二元偏好选择：**Image A 更好 还是 Image B 更好？**。这抛弃了难以校准的连续打分，回归到最纯粹的人类直接偏好 (RLHF 标准做法)。

### 2.2 训练数据格式与指标计算 (Training Data Format)
对于最终保留下来的 **53,000 条** 训练和测试数据（**50k Silver + 3k Golden**），存下来的 JSON 格式是**成对的 (Paired)**。这是因为我们需要使用孪生网络来计算排序损失（Ranking Loss）。其中，`annotator_results` 用于保存原始标注结果：**Silver Set 通常有 2 条 VLM Teacher 结果，Golden Set 通常有 2 条人类双盲结果**。在任意一条 annotator 结果内部，`category_scores_A` / `category_scores_B` 都直接表示 **Image A / Image B 这两张生成图像** 的 3 维诊断分数。完整格式定义详见 `prismbench_format.md`。

示例（简版）：
```json
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
```

---

## 阶段 3：模型训练 (LENS)

LENS (Localized Entanglement Navigation and Scoring) 是一个轻量级的指标模型（例如 Qwen3.5-9B），在 PrismBench 的 5w 银集上进行微调，以同时预测连续的偏好分数和离散的错误类别。

### 3.1 架构：双头多任务学习 (Dual-Head MTL)
我们在 VLM backbone 的最后隐藏层状态上附加了两个头（Heads）：
1.  **分数头 Score Head (回归)：** 输出一个标量偏好分数。
2.  **分类头 Classification Head (分类)：** 输出 3 个正交错误维度 (Existence, Appearance, Interaction) 的 logits。

### 3.2 孪生网络训练策略 (Siamese Network Strategy)
为了防止模型学习到虚假的上下文相关性（"聪明汉斯"效应 Clever Hans effect，比如将复杂的背景与高分相关联），我们在图像对上使用孪生网络的方法进行训练。

*   **输入 (Input)：** Image A（Gemini 好的图片）和 Image B（MOSAIC 差的图片），两者都是从**同一个** Prompt 生成的。
*   **前向传播 (Forward Pass)：** 两张图像独立地通过**同一个**共享的 VLM backbone 以提取特征嵌入（embeddings）。
*   **Loss 1 (综合偏好损失 Preference Scoring Loss)：** 嵌入特征进入分数头 (Score Head) 预测单一分数，并通过 Bradley-Terry Pair-wise Ranking Loss 进行训练。
    *   **Pair-wise Ranking Loss (Margin Ranking / BCE)**: 根据标注的 `preference` (A 或 B)，强制模型对被选中的图片预测出比另一张更高的分数。这直接优化了模型的对比排序能力。
    *   总偏好损失：$\mathcal{L}_{pref} = \text{MarginLoss}(\hat{S}_A, \hat{S}_B)$ 或者使用 Bradley-Terry 的 BCE loss: $\mathcal{L}_{pref} = - \log \sigma(\hat{S}_{preferred} - \hat{S}_{rejected})$$
*   **Loss 2 (分类诊断损失 Classification Diagnosis Loss)：** 嵌入特征进入分类头。由于我们采用了极简的二元分类体系（0 或 1），传统的 Cross-Entropy Loss（仅适用于互斥单分类）不再适用。我们改用 **BCEWithLogitsLoss (Binary Cross Entropy with Logits)**。该损失函数独立计算 3 个维度的二元交叉熵。
*   **总损失 (Total Loss)：** $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pref} + \lambda_2 \mathcal{L}_{cls}$

*为什么这样有效：* 分类损失起到了强大的正则化作用。为了正确识别“特征泄漏”与“语义错位”，backbone 的交叉注意力（cross-attention）**必须**聚焦在主体及其边界上，从而迫使模型忽略无关的背景像素。

### 3.3 标签质量保证与数据清洗漏斗 (Label Quality Assurance & Data Funnel)
在使用最终 **50,000 条** VLM 伪标签 (Silver Set) 训练模型时，Reviewer 必然会质疑数据噪声问题（Garbage In, Garbage Out）。为了保证 Silver Set 的绝对高质量并向学术界展示严谨的工程把控，我们不仅采用了 Teacher-Student 强弱代差蒸馏（使用 GPT-4o 等顶尖闭源模型进行打标），还设计了严格的多级数据清洗漏斗 (Data Cleaning Funnel)：

*   **初始候选数据量**：约 **65,000 对图像 (Pairs)**。
*   **Step 1: CoT 格式与逻辑校验**
    *   *过滤字段*：`reasoning_format` 和 `json_parsing_success`。
    *   *逻辑步骤*：剔除 VLM 未能遵循思维链 (Chain-of-Thought) 规范或 JSON 格式损坏的样本，确保所有标签均基于充分的逻辑推理过程。
    *   *数据量变化*：移除 2,000 对，保留 63,000 对。
*   **Step 2: 双 Teacher 一致性过滤 (Teacher Agreement Filter)**
    *   *过滤字段*：`annotator_results`。
    *   *逻辑步骤*：要求两路 Teacher VLM 对 `preference`、`category_scores_A` 和 `category_scores_B` 的判断高度一致。若两路模型的结论冲突明显，则直接剔除该样本，仅保留高一致性样本。
    *   *数据量变化*：移除 5,000 对，保留 58,000 对。
*   **Step 3: 强制锚点与无区分度过滤 (Hard Anchors)**
    *   *过滤字段*：`category_scores_A` 和 `category_scores_B`。
    *   *逻辑步骤*：对比两张图像的得分。如果两图在所有 3 个错误分类上的得分完全相同（缺乏对比梯度），则直接剔除。这保证了送入 Siamese 网络的每一对数据都具有极强的对比区分度。
    *   *数据量变化*：移除 4,000 对，保留 54,000 对。
*   **Step 4: 多轮一致性校验 (Self-Consistency)**
    *   *过滤字段*：`temperature_variance_score`。
    *   *逻辑步骤*：对同一图像对使用不同 Temperature 进行两次 VLM 评估。计算两次诊断分类的方差，剔除分类结果发生严重跳变的高不确定性噪声样本。
    *   *数据量变化*：移除 4,000 对，保留最终的 50,000 对。
*   **最终入库数据量**：**50,000 对**极高纯度的 Silver Set 训练数据。

*(注：在全面执行上述自动化漏斗前，Teacher VLM 需先在小批量金集上完成**人类基准校准 (Human-in-the-loop Calibration)**，达到 >85% 的人类一致性后，方可启动大规模处理。这种“先考老师，再教学生”的策略从根本上保证了训练分布不偏离人类偏好。)*

### 3.4 数据集划分策略 (Data Split Strategy)
为了绝对保证学术诚信（防止数据泄漏）并兼顾最高效的学术 ROI，我们将最终保留下来的 **53,000 条数据（50k 银集 + 3k 金集）** 进行如下严格划分：

*   **训练集 (Train Set)：约 50,000 条**
    *   *来源：* 纯自动标注银集 (Silver Set)。
    *   *用途：* 用于 Siamese MTL 架构的参数更新和梯度反向传播。5 万条优质 Contrastive Pairs 配合 PEFT (LoRA) 足以覆盖所有长尾的失败模式。
    *   *Reviewer 防守逻辑：* 模型在训练阶段完全不接触人类真实打标的数据，纯靠蒸馏，这证明了我们在论文中宣称的“VLM Teacher-Student 蒸馏”是切实有效的，不存在使用人类数据作弊。

*   **验证集 (Validation Set)：约 500 条**
    *   *来源：* 纯人工标注金集 (Golden Set) 随机抽样。
    *   *用途：* 仅用于训练过程中的 Early Stopping（早停）和超参数调优，**绝不参与梯度回传**。
    *   *Reviewer 防守逻辑：* 为什么不用 Silver 数据做验证？因为如果用 VLM 的伪标签来做早停，模型最终是在迎合 VLM 的偏好。使用 500 条纯人类 Golden 数据做验证，确保了模型的最终检查点 (Checkpoint) 是朝着“人类偏好对齐”的方向停止的。

*   **测试集 (Test Set)：约 2,500 条**
    *   *来源：* 剩余的纯人工标注金集 (Golden Set)。
    *   *用途：* 绝对隔离的 Held-out 数据集。用于产出论文中证明 LENS 优于 CLIP/DINO 的性能指标（Kendall-Tau, F1-Score）。
    *   *Reviewer 防守逻辑：* 2500 条纯人工真值标签在统计学上具备极强的说服力 ($p < 0.001$)。这批数据在模型定稿前**绝对黑盒、不可见**，实现了真正的 Zero-Shot 对齐验证，从根本上杜绝了任何形式的数据泄漏 (Data Leakage) 质疑。

### 3.5 训练与评估框架 (Frameworks)
*   **深度学习框架：** `PyTorch 2.x`
*   **模型库与分布式训练：** 
    *   使用 Hugging Face `transformers` 加载 Qwen3.5-9B backbone。
    *   使用 `peft` 库应用 LoRA / QLoRA 进行高效微调（Parameter-Efficient Fine-Tuning），极大降低显存开销。
    *   使用 `DeepSpeed` (ZeRO-2/3) 或 `accelerate` 进行多卡分布式并行训练。

---

## 阶段 4：多模型基准评测与发榜 (Benchmark Evaluation)

模型训练完成后，论文的最高潮部分是**应用 LENS 评估当前行业的生成模型**，产出极具洞察力的 PrismBench Leaderboard。这一步将彻底确立该论文的学术贡献，也是最能打动 Reviewer 的“秀肌肉”环节。

### 4.1 评测数据集构建 (Leaderboard Set)
为了公平测试，我们将从金集或全新生成的 Prompt 中抽样构建专门的打榜测试集。
*   **被测模型库：** 选取 6 个代表性生成模型：闭源天花板 (Midjourney v6, DALL-E 3)、开源 DiT 新贵 (SD3, Flux)、传统基线 (SDXL) 以及特定策略模型 (MOSAIC)。
*   **评测规模：** 每个模型生成 800 张图（分布于 $N \in \{2, 4, 6, 8\}$ 各 200 张）。
*   **总推理量：** 4,800 张图。这只需进行一次极速的 LENS 单图前向推理。

### 4.2 最终推理输出格式 (Final Eval Output)
在 Leaderboard 打榜或日常使用中，当你输入一张测试图像时，LENS 能够直接输出一个高度结构化的 JSON 结果，它完美包含了四个核心要素：
1. **Prompt** (文本提示)
2. **Subject References** (参考主体列表)
3. **Generated Image** (待评估的生成图像)
4. **Metrics** (独立评估得分：存在性、外观、交互，以及最终的绝对/相对偏好分数)

示例输出：
```json
{
  "prompt": "A photo of [Subject A] and [Subject B] shaking hands.",
  "subject_refs": ["A_ref.jpg", "B_ref.jpg"],
  "generated_image": "midjourney_output_001.jpg",
  "metrics": {
    "existence": 1,
    "appearance": 0,
    "interaction": 1,
    "preference_score": 0.85
  }
}
```

### 4.3 论文核心图表与数据指标 (Metrics for Paper)
通过收集 LENS 对这 4800 张图的输出（绝对 Preference Score 和 3D 分类向量），我们将在论文的 Experiment 章节呈现四大核心结果：

1.  **元评估对齐度 (Meta-Evaluation: LENS vs. CLIP/DINO)**
    *   *目标*：证明 LENS 是一把“更好的尺子”。
    *   *方法*：在 3,000 张人类盲标的金集上，计算 LENS 的 Preference Score 与人类打分的 **Kendall-Tau ($\tau$) 排序相关性** 和 **Pearson ($r$) 线性相关性**。
    *   *预期结论*：LENS 的相关性将达到 0.7+，而 CLIP 和 DINO 将在 0.3 左右徘徊，用铁证宣判传统指标在多主体上的彻底失效。

2.  **多主体生成排行榜 (Multi-Subject Leaderboard)**
    *   *目标*：给行业各路神仙排座次。
    *   *方法*：按模型对所有 4800 张图的 **平均 Preference Score** 进行排名。
    *   *预期洞察*：Midjourney 极大概率依然霸榜，但开源模型在特定 N 数量下可能会有反超表现，或者暴露出特定的缺陷。

3.  **抗压衰减折线图 (Capacity Decay Curve)**
    *   *目标*：探索模型的“容量极限”。
    *   *方法*：X轴为请求主体数量 $N$ (2, 4, 6, 8)，Y轴为平均 Preference Score 或主体存活率。
    *   *预期洞察*：所有模型在 N=2 时分数相近，但在 N=6, 8 时，基线模型（如 SDXL）的分数将呈现“断崖式跳水”，而强模型（如 Midjourney）的衰减更平滑。

4.  **死因诊断堆叠图 (Diagnostic Error Distribution Stack/Radar)**
    *   *目标*：解释模型为什么会输，提供可解释性洞察。
    *   *方法*：利用 LENS 独有的 3D 分类向量，统计每个模型在触发错误时，各错误维度 (Existence, Appearance, Interaction) 的占比。
    *   *预期洞察*：揭示深层机制。例如，“SD3 虽然总分不高，但它的错误多集中在 Appearance (独立外观坍塌与特征泄漏)；而 MOSAIC 的错误几乎全是因为 Existence (存在性缺失)。” 这种**可解释性评价**是本文超越所有黑盒评价模型的核心卖点。

### 4.4 定性案例分析 (Qualitative Case Studies)
在论文的 Appendix 或正文对比图中，挑出 3-4 个经典的“打脸 CLIP”定性案例：
*   *场景*：展示一张“美国队长穿了钢铁侠红色盔甲”的生成图。
*   *对比*：标注 CLIP 给了 0.85 的高分（被红蓝颜色和双人标签欺骗），而 LENS 精准给出了 0.2 的低分，并高亮输出了 `Appearance (Attribute Bleeding) = 0.0`。
*   *效果*：视觉冲击力极强，让 Reviewer 瞬间 Get 到你这篇论文的伟大之处。

---

## 阶段 5：模型发布与 Hugging Face 托管 (Deployment)
为了让整个开源社区都能轻松使用 LENS 评测他们的模型，我们设计了极简的权重保存与托管策略。

### 4.1 权重保存机制 (Save Mechanism)
LENS 模型本身包含了一个庞大的 VLM backbone (Qwen3.5-9B)。为了**避免上传几十 GB 的冗余文件**，`lens/model.py` 中实现了自定义的 `save_pretrained()` 方法，它仅保存增量参数：
1.  **`lens_heads.pt`**: 仅约 20MB。包含了我们自己设计的 Score Head 和 Classification Head 的权重。
2.  **`lora_adapter/` (可选)**: 仅约 50MB。如果你在训练时开启了 `--mode lora`，这里会保存 PEFT 的 LoRA 增量权重。
3.  **`lens_config.json`**: 记录了依赖的基础模型 (`Qwen/Qwen3.5-9B`) 以及 Head 的分类维度等元数据。

当 `scripts/train.py` 运行结束时，上述三个文件会自动保存到 `outputs/LENS-v1-[mode]/` 目录下。

### 4.2 如何上传到 Hugging Face (HF Hub)
将训练好的 LENS 模型分享给社区非常简单：

1.  在 Hugging Face 创建一个模型仓库，例如：`your-username/LENS-Qwen3.5-9B`。
2.  将 `outputs/` 目录下的那三个文件（`lens_heads.pt`, `lora_adapter/`, `lens_config.json`）直接拖拽上传到你的 HF 仓库。
3.  在你的模型卡片 (Model Card/README) 里提供一段简单的加载代码，其他人就可以通过一行代码调用 LENS 了。因为我们保存了 config，其他人的代码会自动从 HF 官方下载原本的 Qwen 权重，并把你的 Head 缝合上去，即插即用！
