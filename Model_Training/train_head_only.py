import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
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
        # 实际使用时，这里应该使用 AutoProcessor(text=prompt, images=image) 生成
        seq_len = 32
        return {
            # 假设 Image A 的输入
            "input_ids_A": torch.randint(0, 1000, (seq_len,)),
            "attention_mask_A": torch.ones(seq_len),
            # "pixel_values_A": ... (根据具体 VLM 补充图像张量)
            
            # 假设 Image B 的输入
            "input_ids_B": torch.randint(0, 1000, (seq_len,)),
            "attention_mask_B": torch.ones(seq_len),
            
            # MarginRankingLoss 需要的 target 必须是 1 (A排前面) 或 -1 (B排前面)
            "preference_label": torch.tensor(1.0 if torch.rand(1).item() > 0.5 else -1.0), 
            
            # CrossEntropyLoss 需要的类别索引 (0-4 代表 5 种错误分类)
            "classification_label": torch.randint(0, 5, (1,)).squeeze()
        }

class LeakGuard(nn.Module):
    """
    LeakGuard Architecture:
    - Backbone: 冻结的视觉语言模型
    - Head 1: 偏好打分头
    - Head 2: 错误分类诊断头
    """
    def __init__(self, model_name="Qwen/Qwen3.5-9B", num_classes=5):
        super(LeakGuard, self).__init__()
        # 使用 Qwen3.5-9B (原生多模态大模型) 作为骨干网络
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 1. 冻结骨干网络 (仅训练头部)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size
        
        # 2. 初始化分类和打分头，并对齐精度到 bfloat16
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
        # 兼容纯文本或多模态的 Forward
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "output_hidden_states": True}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            
        outputs = self.backbone(**kwargs)
        
        # 获取最后一个 Token 的隐藏层状态作为全局表征
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        
        score = self.score_head(last_hidden_state)
        logits = self.classification_head(last_hidden_state)
        
        return score, logits

def main():
    print("初始化 LeakGuard Model (Head Only Training)...")
    
    # 初始化模型
    model = LeakGuard(model_name="Qwen/Qwen3.5-9B") 
    
    # 仅将 Head 的参数传入优化器
    optimizer = torch.optim.AdamW(
        list(model.score_head.parameters()) + list(model.classification_head.parameters()), 
        lr=1e-4
    )
    
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0)
    classification_loss_fn = nn.CrossEntropyLoss()
    
    # 初始化 Dataset 和 DataLoader
    dataset = LeakBenchDataset(length=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    print("开始训练循环 (Qwen Backbone 已冻结，仅更新 Score & Classification Heads)...")
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        device = model.backbone.device
        
        # 1. 前向传播 Image A
        score_A, logits_A = model(
            input_ids=batch["input_ids_A"].to(device),
            attention_mask=batch["attention_mask_A"].to(device)
        )
        
        # 2. 前向传播 Image B
        score_B, logits_B = model(
            input_ids=batch["input_ids_B"].to(device),
            attention_mask=batch["attention_mask_B"].to(device)
        )
        
        # 3. 计算 Ranking Loss (A 和 B 谁更好)
        labels = batch["preference_label"].to(device).to(torch.bfloat16)
        loss_rank = ranking_loss_fn(score_A.squeeze(), score_B.squeeze(), labels)
        
        # 4. 计算 Classification Loss (诊断 Image A 的错误类型)
        cls_labels = batch["classification_label"].to(device)
        loss_cls = classification_loss_fn(logits_A, cls_labels)
        
        # 5. 反向传播
        total_loss = loss_rank + loss_cls
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{len(dataloader)} | Rank Loss: {loss_rank.item():.4f} | Cls Loss: {loss_cls.item():.4f} | Total Loss: {total_loss.item():.4f}")

    print("Head-only 训练跑通！")

if __name__ == "__main__":
    main()
