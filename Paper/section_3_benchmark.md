## Multi-Subject Benchmark

To systematically evaluate the limits of current subject-driven diffusion models in handling complex multi-entity compositions, we construct a rigorous, scalable stress-test benchmark. Unlike existing datasets that primarily focus on single or dual subjects in isolated settings, our benchmark is specifically designed to assess identity preservation and physical interaction reasoning as the number of subjects scales from 2 to 10. This directly addresses the critical need for dataset curation for benchmarking personalized generative models.

### Subject Pool Construction

We establish a diverse and challenging subject pool by sourcing reference images from two well-established personalization datasets: the XVerse dataset~[xverse2024] and the COSMISC dataset~[mosaic2024]. These datasets provide a wide array of human and animal identities with varying attributes, poses, and clothing. We curate a unified subject pool by extracting high-quality reference images and assigning a unique identifier (e.g., S001, S002) to each distinct identity. This unified pool ensures that the models are tested on a consistent set of challenging visual features.

### Benchmark Design and Scalability

The core of our benchmark is a set of carefully crafted evaluation prompts designed to test both scalability and spatial reasoning. We define five distinct difficulty levels based on the subject count: **2, 4, 6, 8, and 10 subjects**, as illustrated in **Figure fig:benchmark_design**. 

 
![**Multi-Subject Benchmark Construction.** Our pipeline samples identities from a unified subject pool (left) to populate prompts of increasing complexity (2 to 10 subjects). Prompts are systematically categorized into Neutral, Occlusion, and Interaction scenarios (right) to isolate and test specific failure modes in attention routing and geometric reasoning.](images/benchmark_design.png)
***Multi-Subject Benchmark Construction.** Our pipeline samples identities from a unified subject pool (left) to populate prompts of increasing complexity (2 to 10 subjects). Prompts are systematically categorized into Neutral, Occlusion, and Interaction scenarios (right) to isolate and test specific failure modes in attention routing and geometric reasoning.*

For each subject level, we design 15 unique prompts, resulting in a total of 75 base prompts. To thoroughly investigate the impact of spatial entanglement on attention leakage, we strictly control the prompt IDs (1-75) and categorize them into three distinct scene types, ensuring an even distribution of 5 prompts per type within each subject level:

    - **Neutral (No Interaction)**: Subjects are placed in the same scene without physical overlap (e.g., *``Subject A and Subject B standing next to each other''*). These prompts test the model's baseline ability to prevent identity bleeding across spatially separated regions.
    - **Occlusion**: One subject partially blocks the view of another (e.g., *``Subject A standing behind Subject B''*). These prompts test the model's amodal completion and depth ordering capabilities.
    - **Interaction**: Subjects are physically engaged with one another (e.g., *``Subject A hugging Subject B''* or *``Subject A hugging Subject B while Subject C stands behind Subject D''*). These are the most challenging prompts, testing the model's ability to maintain fine-grained contact geometry and identity under severe attention entanglement.

### Evaluation Protocol

We evaluate three recent state-of-the-art multi-subject personalization models: **XVerse**~[xverse2024], **MOSAIC**~[mosaic2024], and **PSR**~[psr2024]. These models were specifically selected because they represent the frontier of non-invasive, attention-modulation techniques designed to overcome the limitations of naive region-masking, making them the strongest candidates for complex physical interactions. For each of the 75 prompts, the models take the text prompt and the corresponding subject reference images as input. To account for the stochastic nature of diffusion models, we generate images using 3 random seeds per prompt. This yields a total of 225 generated images per model (675 images across all three models), providing a statistically robust foundation for our quantitative analysis. All generation metadata, including prompt IDs, subject assignments, and output paths, are strictly tracked to ensure reproducible evaluation.
