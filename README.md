# When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization

[![arXiv](https://img.shields.io/badge/arXiv-2603.26078-b31b1b.svg)](https://arxiv.org/abs/2603.26078)
[![CVPR 2026 P13N](https://img.shields.io/badge/CVPR%202026-P13N%20Workshop-blue.svg)](#)

**Zhihan Chen, Yuhuan Zhao, Yijie Zhu, Xinyu Yao**

![Teaser](Paper/latex_source/figures/teaser.png)

This repository contains the code, data, and evaluation scripts for our paper: **"When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization"** (Accepted by CVPR 2026 P13N Workshop).

## 📢 Overview

Subject-driven text-to-image diffusion models have achieved remarkable success in preserving single identities. However, their ability to compose multiple interacting subjects remains highly challenging. When faced with multiple identities, current models often suffer from **Catastrophic Identity Collapse**—features bleed across subjects, or the model generates multiple clones of a single dominant identity.

This repository provides a comprehensive **stress-test benchmark** and a novel evaluation metric (**Subject Collapse Rate - SCR**) to rigorously quantify the limits of state-of-the-art multi-subject models (MOSAIC, XVerse, PSR) as the number of interacting identities scales from 2 to 10.

### Key Contributions
1. **A Scalable Multi-Subject Benchmark**: A rigorous testing suite scaling from 2 to 10 subjects, categorized by interaction complexity (Neutral, Occlusion, Interaction).
2. **Subject Collapse Rate (SCR)**: A new DINOv2-based metric that explicitly quantifies the percentage of subjects that lose their identity in a generated scene, overcoming the "Semantic Shortcut" flaw of global CLIP metrics.
3. **Comprehensive Failure Analysis**: Quantitative and qualitative evidence revealing that while current models succeed at `N=2`, they suffer >95% identity collapse at `N=8`.

---

## 📊 Benchmark Design

Our benchmark is constructed by sampling from a unified subject pool and inserting them into carefully crafted prompts across five difficulty levels (2, 4, 6, 8, 10 subjects) and three scene types:

*   **Neutral (No Interaction)**: Subjects are spatially separated.
*   **Occlusion**: Subjects partially block one another, testing amodal completion.
*   **Interaction**: Subjects are physically engaged (e.g., hugging, shaking hands), testing severe attention entanglement.

[See Benchmark Design Diagram](Paper/latex_source/figures/benchmark_design.pdf)

---

## 📈 Evaluation Metrics & SCR

Standard CLIP-T metrics often present an *illusion of scalability*. As `N` increases, models default to generating a generic "group of people," which satisfies the global text prompt but completely destroys local identity fidelity. 

To address this, we propose **SCR (Subject Collapse Rate)**:

```math
\text{SCR}_{@\tau} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big[\cos(\text{DINOv2}(I_{gen}), \text{DINOv2}(I_{ref}^{(i)})) < \tau\big]
```

*See `scripts/` for the evaluation code used to compute DINOv2, CLIP-I, CLIP-T, and SCR.*

---

## 🚀 Repository Structure

```text
├── Paper/
│   └── latex_source/         # Full LaTeX source code and figures for the CVPR submission
├── eval_outputs/             # Raw JSON/CSV evaluation metrics for MOSAIC, XVerse, PSR
├── results/                  # Generated images from the benchmarked models
├── scripts/                  # Python scripts for data processing and chart generation
├── val_dataset/              # The benchmark dataset (prompts, subjects, config)
├── MOSAIC-main/              # Submodule/Fork of MOSAIC for evaluation
├── XVerse-main/              # Submodule/Fork of XVerse for evaluation
├── PSR-main/                 # Submodule/Fork of PSR for evaluation
└── start.sh                  # Main entry point for running the benchmark
```

---

## 🛠️ Installation & Quick Start

Our benchmark is designed to be fully reproducible. You can use the provided `start.sh` script to automate environment setup, model execution, and evaluation.

### Prerequisites
- Linux or macOS
- Python 3.9+
- NVIDIA GPU (Recommended) or Apple Silicon (Mac)

### 1. Run the Full Benchmark Pipeline
The `start.sh` script will automatically create virtual environments, download required weights (e.g., SAM2, Florence-2, DINOv2), and run the generation and evaluation loop.

```bash
# Run generation and evaluation for all three baseline models
bash start.sh all
```

### 2. Step-by-Step Execution
If you prefer to run generation and evaluation separately:

```bash
# Step 1: Generate images using the benchmark prompts (outputs to results/)
bash start.sh gen --models xverse,mosaic,psr

# Step 2: Evaluate the generated images to compute SCR, DINOv2, and CLIP metrics
bash start.sh eval --models xverse,mosaic,psr
```

*Note: The script automatically handles dependency isolation for different models to prevent pip conflicts.*

---

## 💻 Visualizations and Results

You can find all our generated analytical charts in `Paper/latex_source/figures/`:

1.  **Quantitative Collapse**: `fig1_metrics_vs_subject_count.pdf/png` demonstrates the sharp decline in DINOv2 and the rise of SCR as subject counts increase.
2.  **Scene Complexity**: `fig2_metrics_vs_scene_type.pdf/png` compares model performance across Neutral, Occlusion, and Interaction scenarios.
3.  **Case Analysis**: `fig_case_analysis.png` provides a detailed look at Identity Bleeding during physical interaction.

![Case Analysis](Paper/latex_source/figures/fig_case_analysis.png)

---

## 📝 Citation

If you find our benchmark or metrics useful, please consider citing our work:

```bibtex
@inproceedings{chen2026identities,
  title={When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization},
  author={Chen, Zhihan and Zhao, Yuhuan and Zhu, Yijie and Yao, Xinyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```
