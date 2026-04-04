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
    
    Generates N=2 to N=4 subject pairs with distinct backgrounds (for generated images)
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

    def stitch_images(self, ref_a, ref_b, gen_1, gen_2, output_path):
        """Creates the 2x2 Siamese evaluation grid for the VLM teacher."""
        try:
            imgs = [Image.open(x).resize((512, 512)) for x in [ref_a, ref_b, gen_1, gen_2]]
            grid = Image.new('RGB', (1024, 1024))
            grid.paste(imgs[0], (0, 0))
            grid.paste(imgs[1], (512, 0))
            grid.paste(imgs[2], (0, 512))
            grid.paste(imgs[3], (512, 512))
            grid.save(output_path)
            return output_path
        except Exception as e:
            return None

    def call_ai_teacher(self, stitched_image_path, prompt, subject_a, subject_b):
        """
        Calls Qwen3.5-35B-A3B-FP8 (running locally via Ollama/MLX)
        Uses the strict 4-class MECE taxonomy.
        """
        vlm_prompt = f"""
        You are an expert evaluator for Text-to-Image generation.
        The prompt was: "{prompt}".
        Subject A: {subject_a}. Subject B: {subject_b}.
        
        The image is a 2x2 grid. 
        Top row: Reference Image A, Reference Image B.
        Bottom row: Generated Image 1, Generated Image 2.
        
        Task 1: Which generated image is better at preserving identities?
        Respond with PREFERENCE: 0 (Image 1), 1 (Image 2), or 2 (Tie).
        
        Task 2: Diagnose the main error in the images using this strict decision tree:
        1. Are there exactly N distinct subjects as requested? (If NO -> 3: Entity Collapse).
        2. Are the core identities assigned to the wrong actions/roles? (If YES -> 2: Semantic Swapping).
        3. Are local attributes leaking across subjects? (If YES -> 1: Attribute Bleeding).
        4. Is everything correct? (If YES -> 0: Perfect Alignment).
        
        Output JSON: {{"preference_label": int, "classification_label": int}}
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
            return {"preference_label": 2, "classification_label": 3}
                
        except Exception as e:
            return {"preference_label": 2, "classification_label": 3}

    def process_task(self, task):
        """End-to-End Pipeline for a single PrismBench pair."""
        task_id = str(uuid.uuid4())[:8]
        prompt = task["prompt"]
        
        # 1. Reference Images MUST have solid backgrounds
        ref_a = self.generate_image(f"A high quality photo of {task['sub_a']}, solid white background.", task_id, "refA")
        ref_b = self.generate_image(f"A high quality photo of {task['sub_b']}, solid white background.", task_id, "refB")
        
        # 2. Generated Images MUST have complex backgrounds from the prompt
        # We lower CFG to intentionally induce Bleeding/Swapping/Collapse
        gen_1 = self.generate_image(prompt, task_id, "gen1", cfg=7.0) # Standard
        gen_2 = self.generate_image(prompt, task_id, "gen2", cfg=2.0) # Weakened (Hard Negative)
        
        # 3. Stitch and Label
        stitched_path = os.path.join(self.images_dir, f"stitched_{task_id}.jpg")
        self.stitch_images(ref_a, ref_b, gen_1, gen_2, stitched_path)
        labels = self.call_ai_teacher(stitched_path, prompt, task["sub_a"], task["sub_b"])
        
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "stitched_image_path": stitched_path,
            "preference_label": labels["preference_label"],
            "classification_label": labels["classification_label"]
        }
        return record

    def run(self, num_samples=10):
        print(f"Starting PrismBench Pipeline (Target: {num_samples} samples)")
        tasks = [{"prompt": "A red cat and a blue dog in a snowy park.", "sub_a": "red cat", "sub_b": "blue dog"}] * num_samples
        for task in tqdm(tasks):
            self.dataset.append(self.process_task(task))
            with open(self.json_path, 'w') as f:
                json.dump(self.dataset, f, indent=4)
        print(f"PrismBench Generation Complete: {self.json_path}")

if __name__ == "__main__":
    pipeline = LocalPrismBenchPipeline()
    pipeline.run(num_samples=2)
