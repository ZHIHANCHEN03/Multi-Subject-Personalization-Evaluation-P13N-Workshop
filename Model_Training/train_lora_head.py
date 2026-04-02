import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

class LeakBenchDataset(Dataset):
    """
    数据加载器骨架：假设你已经有了数据，你需要在这里处理图像和文本。
    """
    def __init__(self, length=20):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模拟 VLM Processor 处理后的输出格式
        seq_len = 32
        return {
            "input_ids_A": torch.randint(0, 1000, (seq_len,)),
            "attention_mask_A": torch.ones(seq_len),
            
            "input_ids_B": torch.randint(0, 1000, (seq_len,)),
            "attention_mask_B": torch.ones(seq_len),
            
            "preference_label": torch.tensor(1.0 if torch.rand(1).item() > 0.5 else -1.0), 
            "classification_label": torch.randint(0, 5, (1,)).squeeze()
        }

class LeakGuardWithLoRA(nn.Module):
    """
    LeakGuard Architecture (Phase 2):
    - Backbone: Qwen (使用 LoRA 微调注意力层)
    - Head 1: 偏好打分头
    - Head 2: 错误分类诊断头
    """
    def __init__(self, model_name="Qwen/Qwen3.5-9B", num_classes=5):
        super(LeakGuardWithLoRA, self).__init__()
        
        print(f"Loading base model {model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 2. Apply PEFT / LoRA to the Backbone
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        print("Injecting LoRA adapters into backbone...")
        self.backbone = get_peft_model(self.base_model, peft_config)
        self.backbone.print_trainable_parameters()
        
        hidden_size = self.backbone.config.hidden_size
        
        # 3. Add custom heads
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(torch.bfloat16)
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes)
        ).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, pixel_values=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "output_hidden_states": True}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            
        outputs = self.backbone(**kwargs)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        
        score = self.score_head(last_hidden_state)
        logits = self.classification_head(last_hidden_state)
        
        return score, logits

def main():
    print("初始化 LeakGuard Model (LoRA + Head Joint Training)...")
    
    # 初始化模型
    model = LeakGuardWithLoRA(model_name="Qwen/Qwen3.5-9B")
    
    # 提取所有可训练的参数 (包括 LoRA 参数和自定义 Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=2e-5 # LoRA 联合微调时，学习率设置较低
    )
    
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0)
    classification_loss_fn = nn.CrossEntropyLoss()
    
    dataset = LeakBenchDataset(length=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    print("开始联合训练循环 (LoRA 适配器和 Heads 都在更新)...")
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        device = model.backbone.device
        
        score_A, logits_A = model(
            input_ids=batch["input_ids_A"].to(device),
            attention_mask=batch["attention_mask_A"].to(device)
        )
        
        score_B, logits_B = model(
            input_ids=batch["input_ids_B"].to(device),
            attention_mask=batch["attention_mask_B"].to(device)
        )
        
        labels = batch["preference_label"].to(device).to(torch.bfloat16)
        loss_rank = ranking_loss_fn(score_A.squeeze(), score_B.squeeze(), labels)
        
        cls_labels = batch["classification_label"].to(device)
        loss_cls = classification_loss_fn(logits_A, cls_labels)
        
        total_loss = loss_rank + loss_cls
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{len(dataloader)} | Rank Loss: {loss_rank.item():.4f} | Cls Loss: {loss_cls.item():.4f} | Total Loss: {total_loss.item():.4f}")

    print("LoRA+Head 联合训练跑通！")

if __name__ == "__main__":
    main()
