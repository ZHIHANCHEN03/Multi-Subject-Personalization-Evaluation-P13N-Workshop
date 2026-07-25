import os
import sys

# 1. Fix Python Path for 'lens' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    def __init__(self, output_dir="./PrismBench_Local_Data", teacher_models=None):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.json_path = os.path.join(output_dir, "silver_dataset.json")
        os.makedirs(self.images_dir, exist_ok=True)
        self.dataset = []
        # For a real Silver Set build, pass two different teacher model names here.
        # The duplicated default keeps the local pipeline runnable as a smoke test.
        self.teacher_models = teacher_models or ["qwen3.5:35b", "qwen3.5:35b"]
        
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
        except Exception:
            return None

    def call_ai_teacher(self, model_name, image_A_path, image_B_path, refs, prompt, subjects):
        """
        Calls a Teacher VLM (running locally via Ollama/MLX)
        Uses the strict 3-class MECE taxonomy with binary scoring.
        """
        subjects_str = ", ".join(subjects)
        vlm_prompt = f"""
        You are an expert evaluator for Text-to-Image generation.
        The prompt was: "{prompt}".
        Subjects: {subjects_str}.
        
        Task 1: Assign a binary preference choice ("A" or "B") based on overall quality.
        
        Task 2: Diagnose BOTH Image A and Image B using a binary image-level scoring system (1 = Pass, 0 = Fail).
        Existence: Does the generated image contain all requested reference subjects correctly? (1 or 0)
        Appearance: Does the generated image preserve the correct appearance of the requested reference subjects? (1 or 0)
        Interaction: Does the generated image correctly depict the requested interaction semantics? (1 or 0)
        
        Output JSON EXACTLY like this (no markdown, just json):
        {{
            "preference": "A",
            "category_scores_A": {{
                "existence": 1,
                "appearance": 0,
                "interaction": 0
            }},
            "category_scores_B": {{
                "existence": 0,
                "appearance": 0,
                "interaction": 0
            }}
        }}
        """
        
        try:
            # Requires the configured Ollama model to be available locally.
            # We pass subject refs and both candidates so the teacher can actually inspect the images.
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': vlm_prompt,
                    'images': refs + [image_A_path, image_B_path],
                }]
            )
            
            import re
            json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            print(f"Teacher {model_name} returned non-JSON output. Dropping sample.")
            return None
        except Exception as e:
            print(f"Teacher {model_name} failed to label sample: {e}")
            return None

    def _is_aligned(self, annotator_results):
        if len(annotator_results) < 2:
            return True

        first = annotator_results[0]
        for current in annotator_results[1:]:
            if current.get("preference") != first.get("preference"):
                return False
            if current.get("category_scores_A") != first.get("category_scores_A"):
                return False
            if current.get("category_scores_B") != first.get("category_scores_B"):
                return False

        return True

    def _is_valid_label_payload(self, labels):
        if not isinstance(labels, dict):
            return False
        if labels.get("preference") not in {"A", "B"}:
            return False
        for key in ("category_scores_A", "category_scores_B"):
            scores = labels.get(key)
            if not isinstance(scores, dict):
                return False
            if not {"existence", "appearance", "interaction"}.issubset(scores):
                return False
        return True
            
    def process_task(self, task):
        """End-to-End Pipeline for a single PrismBench pair."""
        task_id = str(uuid.uuid4())[:8]
        prompt = task["prompt"]
        subjects = task["subjects"]
        
        # 1. Reference Images MUST have solid backgrounds
        # In the full Multi-Agent pipeline, these are retrieved from GPT-4o / DALL-E 3
        # Here we simulate the process using SDXL.
        refs = []
        subject_refs = []
        for i, sub in enumerate(subjects):
            ref_path = self.generate_image(f"A high quality photo of {sub}, solid white background.", task_id, f"ref_{i}")
            refs.append(ref_path)
            subject_refs.append({"id": sub, "image_path": ref_path})
            
        # 2. Generated Images MUST have complex backgrounds from the prompt
        # We lower CFG to intentionally induce Bleeding/Swapping/Collapse
        gen_1 = self.generate_image(prompt, task_id, "gen1", cfg=7.0) # Standard (Simulates Gemini)
        gen_2 = self.generate_image(prompt, task_id, "gen2", cfg=2.0) # Weakened (Simulates MOSAIC failures)
        
        # 3. Label with multiple teacher VLMs
        annotator_results = []
        for idx, model_name in enumerate(self.teacher_models):
            labels = self.call_ai_teacher(model_name, gen_1, gen_2, refs, prompt, subjects)
            if labels is None or not self._is_valid_label_payload(labels):
                return None
            annotator_results.append(
                {
                    "annotator_id": f"teacher_vlm_{idx+1:02d}",
                    "model_name": model_name,
                    "preference": labels.get("preference", "A"),
                    "category_scores_A": labels.get("category_scores_A"),
                    "category_scores_B": labels.get("category_scores_B")
                }
            )

        if not self._is_aligned(annotator_results):
            return None
        
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "subject_count": len(subjects),
            "subject_refs": subject_refs,
            "image_A_path": gen_1,
            "image_B_path": gen_2,
            "annotator_results": annotator_results,
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
            record = self.process_task(task)
            if record is None:
                continue
            self.dataset.append(record)
            with open(self.json_path, 'w') as f:
                json.dump(self.dataset, f, indent=4)
        print(f"PrismBench Generation Complete: {self.json_path}")

if __name__ == "__main__":
    pipeline = LocalPrismBenchPipeline()
    pipeline.run(num_samples=2)
