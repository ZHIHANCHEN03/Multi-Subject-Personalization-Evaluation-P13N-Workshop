import os
import json
import uuid
import torch
from PIL import Image
from tqdm import tqdm
import base64

# 1. 导入本地生图库 (Diffusers) 和 本地大模型库 (Ollama)
from diffusers import StableDiffusionXLPipeline
import ollama

class LocalLeakBenchPipeline:
    """
    专为 Mac M4 Pro (48GB 统一内存) 打造的全本地数据生成与打标流水线。
    无需任何 API Key，无需云服务器。
    
    内存占用预估:
    - SDXL (MPS/FP16): ~8GB
    - Qwen-VL 7B/32B (Ollama 4-bit): ~6GB / ~20GB
    总计 < 30GB，在 48GB M4 Pro 上完全可以同时常驻内存！
    """
    
    def __init__(self, output_dir="./LeakBench_Local_Data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.json_path = os.path.join(output_dir, "silver_dataset.json")
        os.makedirs(self.images_dir, exist_ok=True)
        self.dataset = []
        
        print("🚀 正在加载本地图像生成模型 (SDXL) 到 Apple Silicon GPU (MPS)...")
        # 加载 SDXL (如果需要更快，可以换成 SD1.5 或 Flux-Schnell)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to("mps") # mps 是 Mac 的 Metal 硬件加速
        
        # 优化 Mac 上的显存和生成速度
        self.pipe.enable_attention_slicing()

    def generate_image_locally(self, prompt, task_id, prefix):
        """调用本地 MPS 硬件加速生成图像"""
        path = os.path.join(self.images_dir, f"{task_id}_{prefix}.jpg")
        # 如果文件已存在则跳过 (支持断点续传)
        if os.path.exists(path):
            return path
            
        # 仅用 20 步推理以加快速度
        image = self.pipe(prompt, num_inference_steps=20).images[0]
        image.save(path)
        return path

    def stitch_images(self, ref_a, ref_b, gen_1, gen_2, output_path):
        """将 4 张图片拼接成 2x2 网格"""
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
            print(f"Stitching failed: {e}")
            return None

    def call_local_vlm(self, stitched_image_path, prompt, subject_a, subject_b):
        """
        调用本地 Ollama 运行的视觉大模型进行打分和分类
        """
        vlm_prompt = f"""
        You are an expert evaluator for Text-to-Image generation.
        The prompt was: "{prompt}".
        Subject A: {subject_a}. Subject B: {subject_b}.
        
        The image is a 2x2 grid. 
        Top row: Reference Image A, Reference Image B.
        Bottom row: Generated Image 1, Generated Image 2.
        
        Task 1: Which generated image is better at preserving identities without mixing them?
        Respond with PREFERENCE: 0 (Image 1 is better), 1 (Image 2 is better), or 2 (Tie).
        
        Task 2: Diagnose the main error in the images.
        Respond with CLASSIFICATION: 0 (Perfect), 1 (Bleeding), 2 (Swapping), 3 (Homogenization), 4 (Missing).
        
        Provide your reasoning, then output the final JSON:
        {{"preference_label": int, "classification_label": int}}
        """
        
        try:
            # 调用本地 Ollama (确保你在终端运行了 ollama run qwen3.5:35b 等视觉模型)
            # 注意: 如果你使用的是 HuggingFace 上的 Qwen3.5-35B-A3B-FP8 (MoE架构)
            # 你可能需要使用 vLLM 或 mlx 框架来加载。这里为了统一，仍然使用 ollama 接口格式。
            response = ollama.chat(
                model='qwen3.5:35b', # 指定使用 35B 的模型
                messages=[{
                    'role': 'user',
                    'content': vlm_prompt,
                    'images': [stitched_image_path]
                }]
            )
            
            content = response['message']['content']
            # 这里简单模拟解析 JSON (实际应用中可以用正则表达式提取 JSON 块)
            # 为了保证代码不报错，我们做个 Fallback
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["gpt_reasoning"] = content
                return result
            else:
                return {"preference_label": 2, "classification_label": 0, "gpt_reasoning": content}
                
        except Exception as e:
            print(f"Local VLM Error: {e}")
            return {"preference_label": 2, "classification_label": 4, "gpt_reasoning": str(e)}

    def process_single_task(self, task_config):
        """执行端到端的本地任务"""
        task_id = str(uuid.uuid4())[:8]
        prompt = task_config["prompt"]
        
        # 1. 本地生成 Reference 图片 (单独的主体)
        ref_a = self.generate_image_locally(f"A high quality photo of {task_config['sub_a']}", task_id, "refA")
        ref_b = self.generate_image_locally(f"A high quality photo of {task_config['sub_b']}", task_id, "refB")
        
        # 2. 本地生成两张包含双主体的测试图片 (这里模拟两张，可以加不同的 negative prompt 或 seed)
        gen_1 = self.generate_image_locally(prompt + " (Version 1)", task_id, "gen1")
        gen_2 = self.generate_image_locally(prompt + " (Version 2)", task_id, "gen2")
        
        # 3. 拼接
        stitched_path = os.path.join(self.images_dir, f"stitched_{task_id}.jpg")
        self.stitch_images(ref_a, ref_b, gen_1, gen_2, stitched_path)
        
        # 4. 本地 VLM 打标
        labels = self.call_local_vlm(stitched_path, prompt, task_config["sub_a"], task_config["sub_b"])
        
        # 5. 保存
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "stitched_image_path": stitched_path,
            "preference_label": labels["preference_label"],
            "classification_label": labels["classification_label"],
            "gpt_reasoning": labels.get("gpt_reasoning", ""),
        }
        return record

    def run_pipeline(self, num_samples=10):
        print(f"🚀 开始全本地 Pipeline 生成 (目标: {num_samples} 条)")
        
        # 本地生成受限于 GPU，采用单线程串行或小并发
        tasks = []
        for i in range(num_samples):
            tasks.append({
                "prompt": "A photo of a corgi wearing a red hat and a siamese cat wearing blue glasses.",
                "sub_a": "corgi wearing a red hat",
                "sub_b": "siamese cat wearing blue glasses"
            })

        for i, task in enumerate(tqdm(tasks)):
            record = self.process_single_task(task)
            self.dataset.append(record)
            
            # 每跑完一张就保存一次，防止中断
            with open(self.json_path, 'w') as f:
                json.dump(self.dataset, f, indent=4)
                
        print(f"✅ 全本地流水线完成！已保存至 {self.json_path}")

if __name__ == "__main__":
    pipeline = LocalLeakBenchPipeline()
    # 先测试 2 条数据
    pipeline.run_pipeline(num_samples=2)
