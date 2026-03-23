## Experiments

In this section, we present the quantitative and qualitative results of our evaluation. We benchmark three state-of-the-art multi-subject personalization models: **MOSAIC**~[mosaic2024], **XVerse**~[xverse2024], and **PSR**~[psr2024], across five difficulty levels ranging from 2 to 10 interacting subjects.

### Quantitative Results: The Limits of Personalization

To rigorously assess model performance, we first examine the overall trend of identity preservation as the subject count increases. The quantitative results, averaged across all scene types, are summarized in **Table tab:main_results** and visualized in **Figure fig:trend_charts**.

 
[Table Content Here]

 
![**Performance Trends across Subject Counts.** (Left) DINOv2 identity similarity exhibits a precipitous drop for all models as scenes become denser. (Right) Subject Collapse Rate (SCR@0.4) skyrockets from $\sim$50\% at 2 subjects to nearly 100\% at 8-10 subjects, highlighting a fundamental scalability bottleneck.](images/trend_charts.png)
***Performance Trends across Subject Counts.** (Left) DINOv2 identity similarity exhibits a precipitous drop for all models as scenes become denser. (Right) Subject Collapse Rate (SCR@0.4) skyrockets from $\sim$50\% at 2 subjects to nearly 100\% at 8-10 subjects, highlighting a fundamental scalability bottleneck.*

**Catastrophic Identity Forgetting.** 
The most striking finding from our benchmark is the catastrophic collapse of identity fidelity when scaling beyond 4 subjects. As shown in the DINOv2 scores, all models experience a precipitous drop. For instance, MOSAIC~[mosaic2024], which performs best at the 2-subject level with a DINOv2 score of 0.425, plummets to 0.110 at 10 subjects. XVerse~[xverse2024] and PSR~[psr2024] follow a nearly identical trajectory, dropping from 0.355 and 0.325 to 0.104 and 0.101, respectively. 

This failure is further magnified when examining the Subject Collapse Rate (SCR). At the most lenient threshold ($\tau=0.4$), MOSAIC exhibits a 48.9\% collapse rate even with just 2 subjects. When pushed to 8 subjects, the SCR skyrockets to 94.7\% for MOSAIC, 94.4\% for XVerse, and 95.0\% for PSR. At 10 subjects, the SCR for all models exceeds 96\%, indicating that nearly every generated subject has lost its intended identity. This quantitative evidence strongly supports our hypothesis: current attention-routing mechanisms are fundamentally inadequate for resolving dense physical entanglement.

**Model Comparison** 
While all models fail at extreme subject counts, **MOSAIC** demonstrates a noticeably stronger baseline in low-complexity scenarios. At 2 subjects, MOSAIC achieves the highest DINOv2 score (0.425) and the lowest SCR (48.9\%), outperforming XVerse (0.355 / 58.9\%) and PSR (0.325 / 63.3\%). This multifaceted performance profile is further detailed in **Figure fig:radar_plot**, which compares the semantic, style, structure, and identity metrics simultaneously. This suggests that MOSAIC's representation-centric disentanglement provides a more robust defense against attention leakage when the scene is relatively sparse. However, this advantage quickly dissipates as the subject count reaches 6 and beyond, underscoring the universal scalability bottleneck in current DiT/FLUX architectures~[peebles2023scalable,flux2024].

![**Comprehensive Metric Radar (2 Subjects).** A multi-dimensional comparison of MOSAIC, XVerse, and PSR. While all models maintain high semantic alignment (CLIP-T), MOSAIC shows a distinct advantage in structural (DINO) and fine-grained identity (DINOv2) preservation.](images/fig_radar_metrics.png)
***Comprehensive Metric Radar (2 Subjects).** A multi-dimensional comparison of MOSAIC, XVerse, and PSR. While all models maintain high semantic alignment (CLIP-T), MOSAIC shows a distinct advantage in structural (DINO) and fine-grained identity (DINOv2) preservation.*

### The Trade-off: Semantic vs. Physical Disentanglement

A counter-intuitive phenomenon emerges when we analyze the Text-Image Alignment (CLIP-T)~[radford2021learning] scores alongside the identity metrics. While DINOv2~[oquab2023dinov2] scores plummet as the subject count increases, the CLIP-T scores for all models actually exhibit a slight upward trend. For example, PSR's CLIP-T score rises from 0.274 (2 subjects) to 0.309 (10 subjects), and MOSAIC rises from 0.261 to 0.300.

This exposes a critical trade-off mechanism inherent in current diffusion models. When faced with the overwhelming complexity of generating 8 or 10 distinct, interacting individuals, the models actively ``take a shortcut.'' Instead of attempting to faithfully reconstruct each specific identity---which would require precise, uncorrupted attention routing---the models revert to generating a generic ``group of people'' that satisfies the macro-level semantic constraints of the prompt. Consequently, the global CLIP-T score improves, but at the total expense of personalized identity.

### Qualitative Failure Analysis

To better understand *how* these models fail, we perform a qualitative visual analysis of the generated scenes.

 
![**Qualitative Failure Analysis.** Comparison between different models across various subject counts and scene types. In denser scenes (4+ subjects) or complex interactions, models exhibit severe (1) Identity Bleeding, where features merge; (2) Homogenization, generating clones of a single dominant identity; and (3) Amodal Collapse, resulting in malformed geometry under occlusion.](images/qualitative_failures.png)
***Qualitative Failure Analysis.** Comparison between different models across various subject counts and scene types. In denser scenes (4+ subjects) or complex interactions, models exhibit severe (1) Identity Bleeding, where features merge; (2) Homogenization, generating clones of a single dominant identity; and (3) Amodal Collapse, resulting in malformed geometry under occlusion.*

![**Detailed Case Analysis of Identity Bleeding.** (Left) Reference images of two distinct subjects. (Right) Generation results for a complex interaction prompt (``A black woman and a western woman shaking hands''). While the MOSAIC baseline successfully maintains identity boundaries (marked with green check), XVerse and PSR suffer from severe identity bleeding, where the features of Subject A contaminate the spatial region of Subject B (marked with red cross).](images/fig_case_analysis.png)
***Detailed Case Analysis of Identity Bleeding.** (Left) Reference images of two distinct subjects. (Right) Generation results for a complex interaction prompt (``A black woman and a western woman shaking hands''). While the MOSAIC baseline successfully maintains identity boundaries (marked with green check), XVerse and PSR suffer from severe identity bleeding, where the features of Subject A contaminate the spatial region of Subject B (marked with red cross).*

Visual inspection corroborates our quantitative findings and reveals three primary failure modes during multi-subject interaction:

    - **Identity Bleeding**: Features from one subject (e.g., clothing color, facial structure, glasses) seamlessly bleed into an adjacent subject, especially during physical contact (e.g., hugging).
    - **Homogenization**: In dense scenes (8-10 subjects), the model often generates multiple copies of the *same* dominant reference identity, completely ignoring the other requested subjects.
    - **Amodal Collapse**: When subjects are occluded, the model frequently fails to render the hidden geometry correctly, resulting in fused limbs or disembodied floating heads.

