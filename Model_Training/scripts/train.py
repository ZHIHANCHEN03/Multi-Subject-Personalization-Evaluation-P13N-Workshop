import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lens.model import LENS
from lens.dataset import PrismBenchDataset

def main(args):
    print(f"Initializing LENS Model Training. LoRA Enabled: {args.use_lora}")
    
    # 1. Load Model (Dual-Head VLM)
    # Using Qwen3.5-9B as the foundation
    model = LENS(model_name="Qwen/Qwen3.5-9B", num_classes=4, use_lora=args.use_lora)
    
    # 2. Extract Trainable Parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Adjust learning rate based on Phase (Head-only vs LoRA Joint)
    lr = 2e-5 if args.use_lora else 1e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 3. Initialize Multi-Task Loss Functions
    # The Margin Ranking Loss is for the Score Head (Preference)
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0)
    
    # The Cross Entropy Loss is for the Classification Head (Diagnostic Taxonomy)
    # 0: Perfect, 1: Bleeding, 2: Swapping, 3: Collapse
    classification_loss_fn = nn.CrossEntropyLoss()
    
    # 4. Load PrismBench Data
    dataset = PrismBenchDataset(length=20) # Dummy for now, replace with JSON loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model.train()
    print("Starting Diagnostic-Aware Joint Training Loop...")
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        device = model.backbone.device
        
        # A. Siamese Forward Pass for Image A
        score_A, logits_A = model(
            input_ids=batch["input_ids_A"].to(device),
            attention_mask=batch["attention_mask_A"].to(device)
        )
        
        # B. Siamese Forward Pass for Image B
        score_B, logits_B = model(
            input_ids=batch["input_ids_B"].to(device),
            attention_mask=batch["attention_mask_B"].to(device)
        )
        
        # C. Calculate Ranking Loss (Who won?)
        labels = batch["preference_label"].to(device).to(torch.bfloat16)
        loss_rank = ranking_loss_fn(score_A.squeeze(), score_B.squeeze(), labels)
        
        # D. Calculate Diagnostic Classification Loss (Why did Image A/B fail?)
        # Assuming logits_A maps to the flawed image in this batch for demonstration
        cls_labels = batch["classification_label"].to(device)
        loss_cls = classification_loss_fn(logits_A, cls_labels)
        
        # E. Joint Backward Pass (Multi-Task Learning Regularization)
        # This forces the Backbone to extract fine-grained subject features
        # instead of relying on spurious background shortcuts.
        total_loss = loss_rank + loss_cls
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{len(dataloader)} | Rank Loss: {loss_rank.item():.4f} | "
              f"Cls Loss: {loss_cls.item():.4f} | Total Loss: {total_loss.item():.4f}")

    print("Training Step Completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LENS Metric Model Training")
    parser.add_argument("--use_lora", action="store_true", help="Enable Phase 2 LoRA joint training on the backbone")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()
    main(args)
