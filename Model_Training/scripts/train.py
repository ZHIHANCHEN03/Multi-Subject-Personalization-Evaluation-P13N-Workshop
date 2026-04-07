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
    # Using Qwen3.5-9B as the foundation. 4 error classes for multi-label prediction.
    model = LENS(model_name="Qwen/Qwen3.5-9B", num_error_classes=4, use_lora=args.use_lora)
    
    # 2. Extract Trainable Parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Adjust learning rate based on Phase (Head-only vs LoRA Joint)
    lr = 2e-5 if args.use_lora else 1e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 3. Initialize Multi-Task Loss Functions
    # The Margin Ranking Loss is for the Score Head (Preference)
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0)
    
    # The BCEWithLogitsLoss is for the Classification Head (Diagnostic Taxonomy)
    # Allows for multi-label and 3-tier continuous soft-labels (0.0, 0.5, 1.0)
    classification_loss_fn = nn.BCEWithLogitsLoss()
    
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
        # Using preference_score_A and preference_score_B to derive ranking target
        pref_A = batch["preference_score_A"].to(device)
        pref_B = batch["preference_score_B"].to(device)
        # target = 1 if A > B, else -1 (simplification for MarginRankingLoss)
        labels = torch.where(pref_A > pref_B, torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device)).to(torch.bfloat16)
        loss_rank = ranking_loss_fn(score_A.squeeze(-1), score_B.squeeze(-1), labels)
        
        # D. Calculate Diagnostic Classification Loss (Why did Image A/B fail?)
        # Apply BCE loss on both branches to penalize diagnostic errors on both images
        targets_A = batch["category_scores_A"].to(device).to(torch.bfloat16)
        targets_B = batch["category_scores_B"].to(device).to(torch.bfloat16)
        
        loss_cls_A = classification_loss_fn(logits_A, targets_A)
        loss_cls_B = classification_loss_fn(logits_B, targets_B)
        loss_cls = (loss_cls_A + loss_cls_B) / 2.0
        
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
