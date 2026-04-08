# PrismBench: VLM (Teacher) Annotation Prompt

本文档提供了用于自动化标注 PrismBench (Silver Set) 的完整多模态大模型 (VLM) Prompt。

为了保证标签的绝对高质量并避免“幻觉 (Hallucination)”，我们采用了 **“思维链 (Chain-of-Thought, CoT)”** 策略。该 Prompt 强制要求 VLM 在给出最终的 JSON 打分之前，必须先进行“诊断分析”，逐一核对 3 个正交错误维度（Existence, Appearance, Interaction），从而确保它在进行纯二元打分 (`1`, `0`) 时的逻辑一致性。

> **推荐使用的 Teacher VLM**: GPT-4o, Claude 3.5 Sonnet, 或 Qwen-VL-Max 72B

---

## 📌 完整的 VLM Prompt (英文版)

*在实际调用中，请将方括号 `[...]` 中的内容替换为当前样本的真实数据。输入给 VLM 的图像是一张按样本主体数量动态扩展的拼接图 (Stitched Grid)：N=2 时是 `2x2`，N=4 时是 `2x3`，N=6 时是 `2x4`，N=8 时是 `2x5`。*

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

Terminology rule:
- **Subject** = a reference identity from the top-row reference images
- **Image** = a generated image to be evaluated (Image 1 / Image 2)
- Your job is to judge each generated image as a whole against the prompt and all reference subjects
- `category_scores_A` / `category_scores_B` are the score containers for Image 1 / Image 2
- This prompt outputs the result of **one annotator / one VLM pass**; the final stored dataset may later wrap multiple annotators into `annotator_results`

**Step 1: Diagnostic Reasoning (Chain of Thought)**
Before assigning scores, you MUST write a brief diagnostic analysis for BOTH Image 1 and Image 2. Follow this strict hierarchical decision tree (from most foundational to most detailed):

*   **Existence (1=Pass, 0=Fail):** Are all requested Core Reference Subjects correctly present in the generated image? (Fail if any reference subject is missing, cloned, or severely occluded).
*   **Appearance (1=Pass, 0=Fail):** Does the generated image preserve the correct physical structure and appearance for all requested reference subjects? (Fail if there are missing limbs, melted faces, or any attribute bleeding/color swapping).
*   **Interaction (1=Pass, 0=Fail):** Does the generated image correctly depict the requested interactions, actions, and non-reference props? (Fail if actions are assigned to the wrong person, or if they are just standing still instead of interacting).

**Step 2: Scoring (Per-Image)**
Based on your reasoning, assign a binary score (`1` or `0`) for each dimension.
You must score each generated image as a whole.
- For Existence: `1` ONLY if the generated image contains all requested reference subjects correctly. `0` otherwise.
- For Appearance: `1` ONLY if the generated image preserves correct appearance for all requested reference subjects. `0` otherwise.
- For Interaction: `1` ONLY if the generated image correctly depicts the requested interaction semantics. `0` otherwise.

**Independent Evaluation Rule:**
You must evaluate all three dimensions independently based on whatever is visible in the generated image. Even if Existence=0, you should still evaluate Appearance and Interaction for that same generated image.

Finally, make a binary **Preference Choice ("A" or "B")**.
Compare Image 1 (A) and Image 2 (B). Which image is overall better aligned with the prompt and reference images? Choose "A" or "B". If both are terrible, choose the one that is slightly less terrible.

### OUTPUT FORMAT REQUIREMENTS:
You must structure your response EXACTLY as follows. First provide your `<reasoning>`, then provide the `<json_output>`. DO NOT output markdown code blocks around the JSON.

<reasoning>
[Write your step-by-step diagnostic reasoning for Image 1 and Image 2 here, evaluating Existence down to Interaction]
</reasoning>

<json_output>
{
    "preference": ["A" or "B"],
    "category_scores_A": {
        "existence": [1 or 0],
        "appearance": [1 or 0],
        "interaction": [1 or 0]
    },
    "category_scores_B": {
        "existence": [1 or 0],
        "appearance": [1 or 0],
        "interaction": [1 or 0]
    }
}
</json_output>
```

---

## 💡 Prompt 设计亮点 (防御 Reviewer 质疑)

1.  **引入 `<reasoning>` 标签 (Chain of Thought)**：这解决了大模型“偷懒”或“随机猜分”的问题。强制 VLM 在输出分数前先进行文本推理，能大幅提升复杂场景下的打标准确率。在最终解析数据时，我们只需要用正则表达式提取 `<json_output>` 中的内容即可。
2.  **明确的正交评估 (Orthogonal Evaluation)**：在 Prompt 中明确指出了“Existence, Appearance, Interaction”是相互独立的。即使某人消失了，只要画面里还有人，就要评价剩下的外观和交互。这防止了 VLM 的“级联归零”问题。
3.  **清晰的 Binary 定义**：强制 VLM 只使用 `1` 和 `0`，避免了连续分数带来的校准困难和认知过载。
