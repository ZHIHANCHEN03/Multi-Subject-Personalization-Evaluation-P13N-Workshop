# LeakBench: Dataset Requirements & Format

This document defines the data structures and requirements for the **LeakBench** dataset, designed to train the **LeakGuard** metric model for multi-subject personalization evaluation.

## 1. Data Composition
To hit the NeurIPS Datasets & Benchmarks Track, we use a hybrid data collection strategy ("Distillation for Training, Human for Testing"):

- **Silver Set (Training)**: ~20,000 image pairs. Generated using diverse base models (SDXL, Midjourney, etc.) and pseudo-labeled by an advanced Vision-Language Model (e.g., GPT-4o or Claude 3.5 Sonnet).
- **Golden Set (Testing/Validation)**: ~1,000 image pairs. Strictly annotated by human experts. Used as the final benchmark to prove LeakGuard aligns better with human judgment than CLIP or DINOv2.

## 2. Image Preprocessing (Stitching)
Instead of feeding multiple independent images into the VLM (which consumes massive context window and confuses attention), we **stitch** the images into a single grid before feeding them to the model.

**Stitching Format:**
```
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

## 3. JSON Label Format (Pairwise Preference + Classification)

We use a **Pairwise Ranking** approach rather than absolute scoring (0-100), as humans (and AI teachers) are much more consistent at saying "Image 1 is better than Image 2" than assigning absolute numbers.

### JSON Schema for Training/Testing:
```json
[
  {
    "task_id": "0001",
    "prompt": "A photo of [Subject A] holding [Subject B] in a park.",
    "stitched_image_path": "./data/images/stitch_0001.jpg",
    "preference_label": 1, 
    "classification_label": 2,
    "metadata": {
      "subject_a": "a man wearing glasses",
      "subject_b": "a red backpack",
      "model_1": "SDXL",
      "model_2": "Midjourney"
    }
  }
]
```

### Label Definitions:

**`preference_label` (Ranking Target):**
- `0`: Image 1 is strictly better (preserves identities better).
- `1`: Image 2 is strictly better.
- `2`: Tie (Both are equally good or equally bad).

**`classification_label` (Diagnostic Target):**
- `0`: Perfect (No errors).
- `1`: Attribute Bleeding (Features mix together, e.g., the man is wearing a red shirt).
- `2`: Identity Swapping (The backpack is wearing glasses).
- `3`: Homogenization (Both subjects look like the same person/object).
- `4`: Missing Subject (One or both subjects completely failed to generate).

## 4. Why this Format?
1. **Ranking Loss (Bradley-Terry)**: The `preference_label` allows us to train the `score_head` to output a continuous scalar value. We train it so that `Score(Img 1) > Score(Img 2)` if `preference_label == 0`.
2. **Cross-Entropy Loss**: The `classification_label` allows us to train the `classification_head` to diagnose *why* an image failed, giving the model interpretability (which CLIP/DINO lack).