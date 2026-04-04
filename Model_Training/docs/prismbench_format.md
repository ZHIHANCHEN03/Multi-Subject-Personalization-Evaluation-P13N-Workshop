# PrismBench & LENS: Data and Model Architecture

This document defines the formal data structures for the **PrismBench** dataset, designed to train the **LENS** (Localized Entanglement Navigation and Scoring) metric model for multi-subject personalization evaluation.

## 1. Data Composition
To train LENS as an industry-standard diagnostic metric model, PrismBench uses a "Quality over Quantity" strategy:

- **Silver Set (Training)**: ~20,000 image pairs.
  - Generative Models used: Strong models (Flux, SDXL + ControlNet) vs. Weak models (SD 1.5, Low CFG).
  - VLM pseudo-labeled by an advanced AI Teacher (e.g., `Qwen3.5-35B-A3B-FP8`).
  - Strict distribution limits: $N \le 4$ subjects (to prevent gradient death from 100% collapse).
- **Golden Set (Testing/Validation)**: ~1,500 image pairs.
  - Strictly annotated by human experts. 
  - Scales up to $N=8$ subjects to prove LENS's zero-shot robustness against extreme density failures (where CLIP/DINO fail).

## 2. Image Preprocessing (Stitching)
Instead of feeding multiple independent images into the VLM, we **stitch** them into a single grid.
- **Reference Images**: Solid white background (to isolate identity attributes).
- **Generated Images**: Complex backgrounds driven by the prompt (to test spatial attention entanglement).

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

## 3. JSON Label Format (Siamese Pairwise Learning)

### JSON Schema:
```json
[
  {
    "task_id": "0001",
    "prompt": "A photo of [Subject A] and [Subject B] walking in a cyberpunk city.",
    "stitched_image_path": "./data/images/stitch_0001.jpg",
    "preference_label": 1, 
    "classification_label": 2,
    "metadata": {
      "subject_a": "a man wearing glasses",
      "subject_b": "a red backpack"
    }
  }
]
```

### The Hierarchical Taxonomy (4 Classes):

This strict decision tree guarantees MECE (Mutually Exclusive, Collectively Exhaustive) classification, eliminating human annotator disagreement:

1. **Entity Collapse (Class 3)**: Are there exactly $N$ distinct subjects as requested? (If NO $\rightarrow$ Class 3). Includes Missing subjects, Cloning, or complete Homogenization.
2. **Semantic Swapping (Class 2)**: Are the core identities assigned to the wrong actions/roles? (If YES $\rightarrow$ Class 2). 
3. **Attribute Bleeding (Class 1)**: Are the core identities correct but local attributes (color, accessories, limb features) leaking across subjects? (If YES $\rightarrow$ Class 1).
4. **Perfect Alignment (Class 0)**: Is everything correct? (If YES $\rightarrow$ Class 0).

## 4. Multi-Task Learning (MTL) Architecture
LENS uses a Siamese Network to process Image A and Image B independently through a shared VLM backbone.

1. **Ranking Loss (Score Head)**: Uses the `preference_label` to pull the score of the preferred image higher.
2. **Diagnostic Loss (Classification Head)**: Uses the `classification_label` to force the backbone's cross-attention mechanisms to explicitly focus on the subject boundaries. **This acts as a powerful regularizer, preventing the model from taking "spurious background shortcuts" like CLIP does.**
