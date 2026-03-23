## Evaluation Metrics

A central contribution of our benchmark is identifying the limitations of standard personalization metrics when applied to multi-subject composition. Existing protocols heavily rely on global CLIP embeddings~[radford2021learning]. However, global embeddings are highly tolerant of local structural distortions and identity bleeding, rendering them inadequate for measuring severe identity entanglement. To provide a rigorous and comprehensive assessment that aligns with the need for advanced evaluation metrics for personalization quality, we introduce a multi-tiered evaluation framework.

### Text-Image Alignment (CLIP-T)

To measure how well the generated image aligns with the semantic intent of the text prompt, we compute the cosine similarity between the CLIP~[radford2021learning] text embedding of the prompt and the CLIP image embedding of the generated output. 

While CLIP-T is a standard metric for semantic fidelity, we observe a critical caveat in multi-subject scenarios: models may achieve high CLIP-T scores by generating a generic group of people that matches the macro-semantics of the prompt (e.g., ``a group of people''), while completely failing to preserve the specific identities requested. Therefore, CLIP-T must be analyzed in conjunction with fine-grained identity metrics.

### Identity Preservation: From CLIP to DINOv2

Traditionally, identity preservation is measured by computing the cosine similarity between the CLIP image embedding of the generated subject and the reference image (denoted as **CLIP-I**). However, our experiments reveal that CLIP-I scores remain artificially high even when subjects undergo severe identity bleeding or facial distortion.

To capture these local structural failures, we shift our primary identity evaluation from CLIP to **DINOv2**~[oquab2023dinov2]. Unlike CLIP, which is trained primarily via contrastive language-image pre-training and tends to prioritize global semantic layout (e.g., ``a person with glasses''), DINOv2 is a self-supervised vision transformer trained via image-level and patch-level objectives. This training paradigm grants DINOv2 an exceptional sensitivity to fine-grained local features, part-level correspondence, and structural geometry, making it significantly more robust for evaluating identity preservation under complex physical interactions where semantic features might otherwise bleed. While earlier versions like DINOv1~[caron2021emerging] also capture structural priors, DINOv2 leverages a significantly larger and curated pre-training dataset along with an improved objective function, resulting in more discriminative and stable feature embeddings for complex multi-entity scenes.

While a rigorous subject-level evaluation would ideally require instance segmentation masks to isolate each generated subject, such masks are notoriously difficult to obtain accurately in highly entangled multi-subject scenes (e.g., severe occlusion or physical interaction). Therefore, as an established proxy for overall identity fidelity, we compute the DINOv2 image embedding of the *entire* generated image $I_{gen}$ and calculate its average cosine similarity against the embeddings of all $N$ individual reference images $\{I_{ref}^{(1)}, I_{ref}^{(2)}, \dots, I_{ref}^{(N)}\}$:

$$
\text{DINOv2 Score} = \frac{1}{N} \sum_{i=1}^{N} \cos(\text{DINOv2}(I_{gen}), \text{DINOv2}(I_{ref}^{(i)}))
$$

Although comparing a multi-subject scene embedding against single-subject reference embeddings introduces scene complexity into the score, this formulation serves as a highly effective penalty mechanism. It strictly demands that the structural identity of *every* requested subject be strongly represented in the global feature space.

### Subject Collapse Rate (SCR)

Average similarity scores can obscure catastrophic failures of individual subjects within a complex scene. For instance, in an 8-subject image, if 7 subjects are perfectly generated but 1 subject is completely missing or morphed into another identity, the mean DINOv2 score might still appear acceptable. 

 
![**Subject Collapse Rate (SCR).** Unlike average similarity scores which mask individual failures, SCR explicitly counts the proportion of subjects whose DINOv2 identity similarity falls below a strict threshold $\tau$. This provides a more realistic measure of multi-subject entanglement.](images/scr_illustration.png)
***Subject Collapse Rate (SCR).** Unlike average similarity scores which mask individual failures, SCR explicitly counts the proportion of subjects whose DINOv2 identity similarity falls below a strict threshold $\tau$. This provides a more realistic measure of multi-subject entanglement.*

To explicitly quantify these localized failures, we propose the **Subject Collapse Rate (SCR)**, conceptually illustrated in **Figure fig:scr_illustration**. While this metric still utilizes the scene-level generated embedding, it shifts the evaluation from a continuous average to a strict discrete thresholding. We define a subject as "collapsed" if its DINOv2 cosine similarity with the reference image falls below a predefined threshold $\tau$. The SCR for a given generated image is defined as the ratio of collapsed subjects to the total number of subjects:

$$
\text{SCR}_{@\tau} = \frac{1}{N} \sum_{i=1}^{N} \mathds{1}\Big[\cos(\text{DINOv2}(I_{gen}), \text{DINOv2}(I_{ref}^{(i)})) < \tau\Big]
$$

where $\mathds{1}[\cdot]$ is the indicator function. Because DINOv2 similarities typically occupy a lower and more discriminative numerical range than CLIP, we employ strict thresholds $\tau \in \{0.4, 0.5, 0.6\}$, which empirical visual inspection confirms align with severe human-perceivable identity loss. Importantly, this metric naturally penalizes the ``Homogenization'' failure mode: if a model generates multiple clones of a single dominant subject, only that subject's reference will yield a high similarity, while the remaining $N-1$ subjects will fall below the threshold, correctly driving the SCR towards 1.0. A lower SCR indicates better multi-subject preservation, while an SCR approaching 1.0 signifies a complete collapse of personalization.
