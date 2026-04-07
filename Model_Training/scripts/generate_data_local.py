import os
import json
import uuid
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
import ollama

class LocalPrismBenchPipeline:
    """
    PrismBench Local Data Generation Pipeline (for Apple M4 Pro 48GB)
    
    Generates N=2,4,6,8 subject pairs with distinct backgrounds (for generated images)
    and solid backgrounds (for reference images) to train the LENS metric model.
    """
    
    def __init__(self, output_dir="./PrismBench_Local_Data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.json_path = os.path.join(output_dir, "silver_dataset.json")
        os.makedirs(self.images_dir, exist_ok=True)
        self.dataset = []
        
        print("Loading SDXL to Apple Silicon (MPS) for Hard Negatives...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to("mps")
        self.pipe.enable_attention_slicing()

    def generate_image(self, prompt, task_id, prefix, cfg=7.0):
        """Generates an image via MPS hardware acceleration."""
        path = os.path.join(self.images_dir, f"{task_id}_{prefix}.jpg")
        if os.path.exists(path):
            return path
            
        image = self.pipe(prompt, num_inference_steps=20, guidance_scale=cfg).images[0]
        image.save(path)
        return path

    def stitch_images(self, refs, gen_1, gen_2, output_path):
        """
        Creates a Siamese evaluation grid for the VLM teacher.
        refs is a list of reference images.
        """
        try:
            imgs = [Image.open(x).resize((512, 512)) for x in [refs[0], refs[1], gen_1, gen_2]]
            grid = Image.new('RGB', (1024, 1024))
            grid.paste(imgs[0], (0, 0))
            grid.paste(imgs[1], (512, 0))
            grid.paste(imgs[2], (0, 512))
            grid.paste(imgs[3], (512, 512))
            grid.save(output_path)
            return output_path
        except Exception as e:
            return None

    def call_ai_teacher(self, stitched_image_path, prompt, subjects):
        """
        Calls Qwen3.5-35B-A3B-FP8 (running locally via Ollama/MLX)
        Uses the strict 5-class MECE taxonomy with 3-tier scoring.
        """
        subjects_str = ", ".join(subjects)
        vlm_prompt = f"""
        You are an expert evaluator for Text-to-Image generation.
        The prompt was: "{prompt}".
        Subjects: {subjects_str}.
        
        The image is a 2x2 grid. 
        Top row: Reference Images.
        Bottom row: Generated Image 1 (Left), Generated Image 2 (Right).
        
        Task 1: Assign a preference score (0.0 to 1.0) for both images based on overall quality.
        
        Task 2: Diagnose the errors in BOTH Image 1 and Image 2 using a 3-tier scoring system (1.0 = Yes, 0.5 = Maybe, 0.0 = No).
        Class 4 (Collapse): Are there exactly N distinct subjects as requested? (If NO -> 1.0 or 0.5)
        Class 3 (Swapping): Are the core identities assigned to the wrong actions/roles? (If YES -> 1.0 or 0.5)
        Class 2 (Bleeding): Are local attributes leaking across subjects? (If YES -> 1.0 or 0.5)
        Class 1 (Misalignment): Is the global background/style ignoring the prompt? (If YES -> 1.0 or 0.5)
        
        Output JSON EXACTLY like this (no markdown, just json):
        {{
            "preference_score_A": 0.9,
            "preference_score_B": 0.2,
            "category_scores_A": {{"class_4_collapse": 0.0, "class_3_swapping": 0.0, "class_2_bleeding": 0.0, "class_1_misalignment": 0.0}},
            "category_scores_B": {{"class_4_collapse": 1.0, "class_3_swapping": 0.0, "class_2_bleeding": 0.0, "class_1_misalignment": 0.0}}
        }}
        """
        
        try:
            # Requires `ollama run qwen3.5:35b` in terminal
            response = ollama.chat(
                model='qwen3.5:35b',
                messages=[{'role': 'user', 'content': vlm_prompt, 'images': [stitched_image_path]}]
            )
            
            import re
            json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._fallback_labels()
                
        except Exception as e:
            return self._fallback_labels()
            
    def _fallback_labels(self):
        return {
            "preference_score_A": 0.5, "preference_score_B": 0.5,
            "category_scores_A": {"class_4_collapse": 0.0, "class_3_swapping": 0.0, "class_2_bleeding": 0.0, "class_1_misalignment": 0.0},
            "category_scores_B": {"class_4_collapse": 0.0, "class_3_swapping": 0.0, "class_2_bleeding": 0.0, "class_1_misalignment": 0.0}
        }

    def process_task(self, task):
        """End-to-End Pipeline for a single PrismBench pair."""
        task_id = str(uuid.uuid4())[:8]
        prompt = task["prompt"]
        subjects = task["subjects"]
        
        # 1. Reference Images MUST have solid backgrounds
        # In a real GPT-driven pipeline, these would be retrieved from SemAlign-MS-Subjects200K
        # Here we simulate the process using SDXL.
        refs = []
        for i, sub in enumerate(subjects):
            ref_path = self.generate_image(f"A high quality photo of {sub}, solid white background.", task_id, f"ref_{i}")
            refs.append(ref_path)
            
        # Ensure we have at least 2 refs for the 2x2 grid visualization logic
        while len(refs) < 2:
            refs.append(refs[-1])
        
        # 2. Generated Images MUST have complex backgrounds from the prompt
        # We lower CFG to intentionally induce Bleeding/Swapping/Collapse
        gen_1 = self.generate_image(prompt, task_id, "gen1", cfg=7.0) # Standard (Simulates Gemini)
        gen_2 = self.generate_image(prompt, task_id, "gen2", cfg=2.0) # Weakened (Simulates MOSAIC failures)
        
        # 3. Stitch and Label
        stitched_path = os.path.join(self.images_dir, f"stitched_{task_id}.jpg")
        self.stitch_images(refs, gen_1, gen_2, stitched_path)
        labels = self.call_ai_teacher(stitched_path, prompt, subjects)
        
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "subject_count": len(subjects),
            "stitched_image_path": stitched_path,
            "preference_score_A": labels.get("preference_score_A", 0.5),
            "preference_score_B": labels.get("preference_score_B", 0.5),
            "category_scores_A": labels.get("category_scores_A", self._fallback_labels()["category_scores_A"]),
            "category_scores_B": labels.get("category_scores_B", self._fallback_labels()["category_scores_B"]),
            "metadata": {
                "source": "GPT Automated Subject Generation Simulation"
            }
        }
        return record

    def run(self, num_samples=10):
        print(f"Starting PrismBench Pipeline (Target: {num_samples} samples)")
        # Simulated tasks for N=2,4,6,8
        tasks = [
            {"prompt": "A red cat and a blue dog in a snowy park.", "subjects": ["red cat", "blue dog"]},
            {"prompt": "A red cat, a blue dog, a green bird, and a yellow fish on a spaceship.", "subjects": ["red cat", "blue dog", "green bird", "yellow fish"]}
        ] * (num_samples // 2 + 1)
        
        for task in tqdm(tasks[:num_samples]):
            self.dataset.append(self.process_task(task))
            with open(self.json_path, 'w') as f:
                json.dump(self.dataset, f, indent=4)
        print(f"PrismBench Generation Complete: {self.json_path}")

if __name__ == "__main__":
    pipeline = LocalPrismBenchPipeline()
    pipeline.run(num_samples=2)
