# PrismBench: VLM (Teacher) Annotation Prompt

本文档提供了用于自动化标注 PrismBench (Silver Set) 的完整多模态大模型 (VLM) Prompt。

为了保证标签的绝对高质量并避免“幻觉 (Hallucination)”，我们采用了 **“思维链 (Chain-of-Thought, CoT)”** 策略。该 Prompt 强制要求 VLM 在给出最终的 JSON 打分之前，必须先进行“诊断分析”，逐一核对 5 个错误维度（Class 5 到 Class 1），从而确保它在进行 3 级打分 (`1.0`, `0.5`, `0.0`) 时的逻辑一致性。

> **推荐使用的 Teacher VLM**: GPT-4o, Claude 3.5 Sonnet, 或 Qwen-VL-Max 72B

---

## 📌 完整的 VLM Prompt (英文版)

*在实际调用中，请将方括号 `[...]` 中的内容替换为当前样本的真实数据。输入给 VLM 的图像是一张 `2x2` 的拼接图 (Stitched Grid)。*

```text
You are an expert, meticulous evaluator for Text-to-Image generation models. Your task is to diagnose complex multi-subject image generations.

### INPUT INFORMATION:
- **Prompt:** "{prompt}"
- **Core Reference Subjects:** {subjects_str}
- **Image Input:** A 2x2 grid image.
  - Top row: Reference Images of the Core Subjects (solid white background).
  - Bottom row: Generated Image 1 (Left) and Generated Image 2 (Right). Both were generated using the EXACT same Prompt.

### YOUR TASK:
You must evaluate **Generated Image 1** and **Generated Image 2** independently against the Reference Images and the Prompt.

**Step 1: Diagnostic Reasoning (Chain of Thought)**
Before assigning scores, you MUST write a brief diagnostic analysis for BOTH Image 1 and Image 2. Follow this strict hierarchical decision tree (from most severe to least severe):

*   **Class 5 (Subject Omission & Homogenization):** Are all the requested Core Reference Subjects present? Is any subject completely missing or cloned (e.g., two identical dogs instead of a dog and a cat)?
*   **Class 4 (Subject Distortion & Mutilation):** Assuming the subjects are present, are their core biological/physical structures severely distorted or mutilated (e.g., missing limbs, melted faces, three arms)?
*   **Class 3 (Semantic Swapping):** Are the core subjects assigned to the wrong actions, roles, clothing, or props described in the prompt (e.g., A is riding the horse instead of B)?
*   **Class 2 (Attribute Bleeding):** Are local attributes (colors, textures, accessories) leaking or blending across subjects (e.g., the red color of A's shirt bleeding onto B's blue shirt)?
*   **Class 1 (Prompt Misalignment):** Assuming no severe errors above, did the image completely fail to generate the requested interactions, or did it omit non-reference text-only props (e.g., the prompt asked for them to "shake hands" but they are just standing still)?

**Step 2: Scoring**
Based on your reasoning, assign a 3-tier score for each Class:
- **`1.0` (Yes)**: A clear, undeniable occurrence of this error.
- **`0.5` (Maybe)**: A minor flaw, ambiguous occurrence, or severe occlusion making it hard to judge.
- **`0.0` (No)**: This specific error is absolutely NOT present.

Finally, assign an overall **Preference Score (0.0 to 1.0)** for both Image 1 and Image 2. The image with fewer and less severe errors should receive a higher score.

### OUTPUT FORMAT REQUIREMENTS:
You must structure your response EXACTLY as follows. First provide your `<reasoning>`, then provide the `<json_output>`. DO NOT output markdown code blocks around the JSON.

<reasoning>
[Write your step-by-step diagnostic reasoning for Image 1 and Image 2 here, evaluating Class 5 down to Class 1]
</reasoning>

<json_output>
{
    "preference_score_A": [float between 0.0 and 1.0 for Image 1],
    "preference_score_B": [float between 0.0 and 1.0 for Image 2],
    "category_scores_A": {
        "class_5_omission": [1.0, 0.5, or 0.0],
        "class_4_distortion": [1.0, 0.5, or 0.0],
        "class_3_swapping": [1.0, 0.5, or 0.0],
        "class_2_bleeding": [1.0, 0.5, or 0.0],
        "class_1_misalignment": [1.0, 0.5, or 0.0]
    },
    "category_scores_B": {
        "class_5_omission": [1.0, 0.5, or 0.0],
        "class_4_distortion": [1.0, 0.5, or 0.0],
        "class_3_swapping": [1.0, 0.5, or 0.0],
        "class_2_bleeding": [1.0, 0.5, or 0.0],
        "class_1_misalignment": [1.0, 0.5, or 0.0]
    }
}
</json_output>
```

---

## 💡 Prompt 设计亮点 (防御 Reviewer 质疑)

1.  **引入 `<reasoning>` 标签 (Chain of Thought)**：这解决了大模型“偷懒”或“随机猜分”的问题。强制 VLM 在输出分数前先进行文本推理，能大幅提升复杂场景下的打标准确率。在最终解析数据时，我们只需要用正则表达式提取 `<json_output>` 中的内容即可。
2.  **明确的层级约束 (Hierarchical Decision Tree)**：在 Prompt 中明确指出了“从最严重到最轻微”的评估顺序（Class 5 $\rightarrow$ Class 1），并在每一类的描述中加入了互斥防线的提示（例如在 Class 4 强调物理结构，在 Class 2 强调颜色材质），防止 VLM 在多个类别上重复扣分。
3.  **清晰的 3-Tier 定义**：明确告诉 VLM 什么时候该用 `0.5`（ambiguous occurrence / severe occlusion），避免它滥用中间分数。