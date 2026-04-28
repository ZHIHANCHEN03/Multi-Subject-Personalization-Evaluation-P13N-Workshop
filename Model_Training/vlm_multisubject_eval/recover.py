import json
from pathlib import Path

# 直接导入你原脚本里的核心逻辑（假设你的原脚本名叫 run_vlm_eval.py）
from run_vlm_eval import (
    load_dataset,
    prepare_items,
    chunk_gemini_prepared_items
)

def main():
    # 这里的路径请根据你的实际情况修改，默认用了你 argparser 里的默认值
    dataset_path = Path("data/train_60k_v13_2.jsonl") 
    base_dir = Path("data")                           
    
    # 1. 2.5-flash 配置
    flash_2_5_chunks = 50
    flash_2_5_out = Path("results/gemini_2_5_flash_full_60000.jsonl")
    
    # 2. 3.1-lite 配置
    lite_3_1_chunks = 173
    lite_3_1_out = Path("results/gemini_3_1_flash_lit_preview_full_60000.jsonl")

    print(f"正在加载和预处理数据 (需要读取和计算图片 bytes 大小，可能需要几十秒)...")
    items = load_dataset(dataset_path, base_dir)
    prepared_items = prepare_items(items, base_dir)
    
    print(f"数据总数: {len(prepared_items)}。正在本地模拟分块计算...")
    # 因为分块逻辑只看图片大小，跟具体选什么模型无关，所以切一次就行
    chunks = chunk_gemini_prepared_items(prepared_items)
    print(f"模拟计算完成！总共切出了 {len(chunks)} 个 batch。")

    # ----- 恢复 2.5-flash 的 task_id -----
    print(f"\n正在提取前 {flash_2_5_chunks} 个 batch 的 task_id，写入 {flash_2_5_out}")
    flash_2_5_out.parent.mkdir(parents=True, exist_ok=True)
    with open(flash_2_5_out, "w", encoding="utf-8") as f:
        for chunk in chunks[:flash_2_5_chunks]:
            for prepared in chunk:
                # 写入占位符
                f.write(json.dumps({"task_id": str(prepared["task_id"]), "status": "recovering"}) + "\n")
                
    # ----- 恢复 3.1-lite 的 task_id -----
    print(f"正在提取前 {lite_3_1_chunks} 个 batch 的 task_id，写入 {lite_3_1_out}")
    lite_3_1_out.parent.mkdir(parents=True, exist_ok=True)
    with open(lite_3_1_out, "w", encoding="utf-8") as f:
        for chunk in chunks[:lite_3_1_chunks]:
            for prepared in chunk:
                f.write(json.dumps({"task_id": str(prepared["task_id"]), "status": "recovering"}) + "\n")

    print("\n✅ 本地模拟计算完美结束！你现在可以安全地继续运行主程序了。")

if __name__ == "__main__":
    main()