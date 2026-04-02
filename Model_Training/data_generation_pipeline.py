import os
import json
import uuid
import openai
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from dotenv import load_load_dotenv

# 加载你的 API Key
# load_dotenv() 
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LeakBenchDataGenerator:
    """
    LeakBench 自动化数据生成与伪标签打标 Pipeline (针对 Mac M4 Pro 48GB 优化)
    
    流程：
    1. 从预设的 Subject 池中抽取组合，生成 Prompt。
    2. (模拟) 调用文生图 API (如 Midjourney, SDXL, Flux) 生成图片。
    3. 将 4 张图拼接成一张大网格图 (Stitching)。
    4. 调用 GPT-4o (Vision) 对这张大图进行 Pairwise 偏好打分和错误诊断分类。
    5. 保存为 JSON 格式，供 Qwen3.5-9B 训练。
    """
    
    def __init__(self, output_dir="./LeakBench_Data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.json_path = os.path.join(output_dir, "silver_dataset.json")
        
        os.makedirs(self.images_dir, exist_ok=True)
        self.dataset = []
        
        # self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def _simulate_image_generation(self, prompt, model_name, task_id, idx):
        """
        模拟调用外部生图 API (实际中你需要换成 Replicate / Baseten / 阿里百炼 等 API)
        """
        # 这里创建一个带颜色的纯色图像模拟生成的图片
        color = (255, 0, 0) if idx == 1 else (0, 0, 255)
        img = Image.new('RGB', (512, 512), color=color)
        path = os.path.join(self.images_dir, f"{task_id}_{model_name}_{idx}.jpg")
        img.save(path)
        return path

    def stitch_images(self, ref_a, ref_b, gen_1, gen_2, output_path):
        """
        将 4 张图片拼接成 2x2 网格，供 GPT-4o 视觉模型判断
        [Ref A] [Ref B]
        [Gen 1] [Gen 2]
        """
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

    def call_gpt4o_for_pseudo_label(self, stitched_image_path, prompt, subject_a, subject_b):
        """
        核心函数：调用 GPT-4o 视觉模型，自动生成 preference_label 和 classification_label
        """
        # 实际代码中，你需要将图片转成 base64 喂给 GPT-4o
        # 这里我们模拟 GPT-4o 的返回结果
        import random
        
        # 模拟偏好：0(图1好), 1(图2好), 2(平局)
        pref = random.choice([0, 1, 2])
        
        # 模拟分类：0(完美), 1(属性泄漏), 2(身份交换), 3(同质化), 4(丢失)
        cls = random.choice([0, 1, 2, 3, 4])
        
        return {
            "preference_label": pref,
            "classification_label": cls,
            "gpt_reasoning": "This is a simulated GPT-4o reasoning text."
        }

    def process_single_task(self, task_config):
        """处理单个生图与打标任务"""
        task_id = str(uuid.uuid4())[:8]
        prompt = task_config["prompt"]
        
        # 1. 准备 Reference 图片 (这里模拟两张参考图)
        ref_a = self._simulate_image_generation("ref", "real", task_id, "refA")
        ref_b = self._simulate_image_generation("ref", "real", task_id, "refB")
        
        # 2. 生成两张测试图片 (模拟不同模型生成的)
        gen_1 = self._simulate_image_generation(prompt, task_config["model_1"], task_id, 1)
        gen_2 = self._simulate_image_generation(prompt, task_config["model_2"], task_id, 2)
        
        # 3. 拼接图片
        stitched_path = os.path.join(self.images_dir, f"stitched_{task_id}.jpg")
        self.stitch_images(ref_a, ref_b, gen_1, gen_2, stitched_path)
        
        # 4. GPT-4o 自动打标
        labels = self.call_gpt4o_for_pseudo_label(
            stitched_path, prompt, task_config["sub_a"], task_config["sub_b"]
        )
        
        # 5. 构建 JSON 记录
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "stitched_image_path": stitched_path,
            "preference_label": labels["preference_label"],
            "classification_label": labels["classification_label"],
            "gpt_reasoning": labels["gpt_reasoning"],
            "metadata": {
                "subject_a": task_config["sub_a"],
                "subject_b": task_config["sub_b"],
                "model_1": task_config["model_1"],
                "model_2": task_config["model_2"]
            }
        }
        return record

    def run_pipeline(self, num_samples=100, max_workers=10):
        """
        利用 Mac M4 Pro 的强大多线程能力，并发调用 API
        """
        print(f"🚀 启动自动化 Pipeline (目标: {num_samples} 条数据, 并发: {max_workers})")
        
        # 构造 Dummy 任务队列
        tasks = []
        for i in range(num_samples):
            tasks.append({
                "prompt": "A photo of a corgi wearing a red hat and a siamese cat wearing blue glasses.",
                "sub_a": "corgi wearing a red hat",
                "sub_b": "siamese cat wearing blue glasses",
                "model_1": "SDXL",
                "model_2": "Flux.1-schnell"
            })

        # 并发执行 (非常适合 Mac M4 Pro 处理大量 I/O 密集型 API 请求)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self.process_single_task, tasks), total=num_samples))
            
        self.dataset.extend(results)
        
        # 保存到 JSON
        with open(self.json_path, 'w') as f:
            json.dump(self.dataset, f, indent=4)
            
        print(f"✅ 自动化流水线完成！共生成 {len(self.dataset)} 条数据，已保存至 {self.json_path}")

if __name__ == "__main__":
    generator = LeakBenchDataGenerator()
    # 在本地测试时，先生成 50 条看看流程是否跑通
    generator.run_pipeline(num_samples=50, max_workers=8)
