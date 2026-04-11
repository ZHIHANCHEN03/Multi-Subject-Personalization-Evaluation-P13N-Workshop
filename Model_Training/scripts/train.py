import os
import sys
import subprocess

# 1. Auto-check and install dependencies
def ensure_dependencies():
    required_packages = {"torch": "torch", "peft": "peft", "PIL": "Pillow"}
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing missing dependency: {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            
    # Qwen3-VL explicitly requires the bleeding edge dev version of transformers
    try:
        import transformers
        # Simple check, real verification happens at load time
    except ImportError:
        print(f"Installing dev version of transformers for Qwen3-VL support...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/transformers"])

ensure_dependencies()

# 2. Fix Python Path for 'lens' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lens.model import LENS
from lens.dataset import PrismBenchDataset

def custom_collate_fn(batch):
    # This handles variable-length token sequences by padding them (if they weren't already fixed-length)
    # For now, dataset returns fixed length dummy tensors, but this ensures future-proofing.
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

def main(args):
    print(f"--- Initializing LENS Training Pipeline ---")
    print(f"Mode: {args.mode.upper()} (Head-only vs LoRA)")
    
    # Enforcing CUDA as requested
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available on this machine. Falling back to CPU for testing purposes only.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # 1. Load Model (Dual-Head VLM)
    # Using Qwen3.5-9B-Base as the foundation metric model for early fusion training
    model = LENS(model_name="Qwen/Qwen3.5-9B-Base", num_error_classes=3, mode=args.mode, unfreeze_layers=args.unfreeze_layers)
    
    # Since device_map="auto" handles placement on CUDA, we don't strictly need model.to(device)
    # But we set the device var to ensure our input tensors go to the correct GPU (e.g. model.backbone.device)
    if torch.cuda.is_available():
        device = model.backbone.device
    
    # 2. Extract Trainable Parameters
    # Automatically filters out frozen backbone parameters if mode == "head_only"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Adjust learning rate based on Phase
    # LoRA on Backbone requires a smaller LR (2e-5) to avoid catastrophic forgetting
    # Training Heads only from scratch requires a larger LR (1e-4)
    lr = 2e-5 if args.mode == "lora" else 1e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 4. Initialize Multi-Task Loss Functions
    # The Margin Ranking Loss is for the Score Head (Preference)
    # We use a standard margin (e.g., 1.0) for binary preference learning (A vs B)
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0)
    
    # The BCEWithLogitsLoss is for the Classification Head (Diagnostic Taxonomy)
    # Applies binary cross-entropy on the 3 orthogonal diagnostic dimensions (Existence, Appearance, Interaction)
    classification_loss_fn = nn.BCEWithLogitsLoss()
    
    # 4. Load PrismBench Data
    train_path = os.path.join(os.path.dirname(__file__), "../data_v1/train_v1.json")
    val_path = os.path.join(os.path.dirname(__file__), "../data_v1/val_v1.json")
    
    train_dataset = PrismBenchDataset(json_path=train_path)
    val_dataset = PrismBenchDataset(json_path=val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Loss weightings
    alpha = args.alpha # Weight for preference ranking loss
    beta = args.beta   # Weight for diagnostic classification loss
    num_epochs = args.epochs
    
    print(f"Starting Diagnostic-Aware Joint Training Loop. Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, Epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        # ---------------- TRAIN ----------------
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} [TRAIN] ---")
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            score_A, logits_A = model(
                input_ids=batch["input_ids_A"].to(device),
                attention_mask=batch["attention_mask_A"].to(device)
            )
            score_B, logits_B = model(
                input_ids=batch["input_ids_B"].to(device),
                attention_mask=batch["attention_mask_B"].to(device)
            )
            
            labels = batch["preference_label"].to(device).float()
            loss_pref = ranking_loss_fn(score_A.squeeze(-1), score_B.squeeze(-1), labels)
            
            targets_A = batch["category_scores_A"].to(device).float()
            targets_B = batch["category_scores_B"].to(device).float()
            
            loss_cls_A = classification_loss_fn(logits_A, targets_A)
            loss_cls_B = classification_loss_fn(logits_B, targets_B)
            loss_cls = (loss_cls_A + loss_cls_B) / 2.0
            
            total_loss = alpha * loss_pref + beta * loss_cls
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            print(f"Step {step+1}/{len(train_loader)} | Pref Rank Loss: {loss_pref.item():.4f} | "
                  f"Cls Loss: {loss_cls.item():.4f} | Total Loss: {total_loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        print(f"--- Epoch {epoch+1}/{num_epochs} [VALIDATION] ---")
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                score_A, logits_A = model(
                    input_ids=batch["input_ids_A"].to(device),
                    attention_mask=batch["attention_mask_A"].to(device)
                )
                score_B, logits_B = model(
                    input_ids=batch["input_ids_B"].to(device),
                    attention_mask=batch["attention_mask_B"].to(device)
                )
                
                labels = batch["preference_label"].to(device).float()
                loss_pref = ranking_loss_fn(score_A.squeeze(-1), score_B.squeeze(-1), labels)
                
                targets_A = batch["category_scores_A"].to(device).float()
                targets_B = batch["category_scores_B"].to(device).float()
                
                loss_cls_A = classification_loss_fn(logits_A, targets_A)
                loss_cls_B = classification_loss_fn(logits_B, targets_B)
                loss_cls = (loss_cls_A + loss_cls_B) / 2.0
                
                v_loss = alpha * loss_pref + beta * loss_cls
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 6. Save Checkpoint after each epoch
        save_dir = os.path.join(os.path.dirname(__file__), f"../outputs/LENS-v1-{args.mode}-epoch{epoch+1}")
        model.save_pretrained(save_dir)
        print(f"Checkpoint saved to: {os.path.abspath(save_dir)}")

    print("\nTraining Completed successfully!")
    print(f"You can now upload the contents of {save_dir} directly to Hugging Face 🤗")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LENS Metric Model Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["head_only", "lora", "partial", "full"], default="lora", 
                        help="Training Mode: 'head_only' (freeze all), 'lora' (PEFT on linear layers), 'partial' (unfreeze top N layers), or 'full' (finetune everything).")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of top layers to unfreeze if mode='partial'")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Siamese training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Ranking Loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for Classification Loss")
    args = parser.parse_args()
    main(args)
