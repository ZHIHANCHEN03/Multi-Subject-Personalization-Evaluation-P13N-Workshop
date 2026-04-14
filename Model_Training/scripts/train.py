import os
import sys

# Safe default Hugging Face cache settings for server-side training.
# This prevents quota issues from /workspace-backed Xet storage and
# keeps manual `python scripts/train.py ...` runs consistent.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/huggingface_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 2. Fix Python Path for 'lens' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unsloth
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from lens.model import LENS
from lens.dataset import PrismBenchDataset

def custom_collate_fn(batch):
    # This handles variable-length token sequences by padding them
    from torch.nn.utils.rnn import pad_sequence
    
    # We must custom collate because image sizes and text sequences might differ slightly
    out = {}
    
    out["task_id"] = [b["task_id"] for b in batch]
    out["preference_label"] = torch.stack([b["preference_label"] for b in batch])
    out["category_scores_A"] = torch.stack([b["category_scores_A"] for b in batch])
    out["category_scores_B"] = torch.stack([b["category_scores_B"] for b in batch])
    
    # Pad sequences
    out["input_ids_A"] = pad_sequence([b["input_ids_A"] for b in batch], batch_first=True, padding_value=0)
    out["attention_mask_A"] = pad_sequence([b["attention_mask_A"] for b in batch], batch_first=True, padding_value=0)
    out["input_ids_B"] = pad_sequence([b["input_ids_B"] for b in batch], batch_first=True, padding_value=0)
    out["attention_mask_B"] = pad_sequence([b["attention_mask_B"] for b in batch], batch_first=True, padding_value=0)
    if "mm_token_type_ids_A" in batch[0]:
        out["mm_token_type_ids_A"] = pad_sequence([b["mm_token_type_ids_A"] for b in batch], batch_first=True, padding_value=0)
        out["mm_token_type_ids_B"] = pad_sequence([b["mm_token_type_ids_B"] for b in batch], batch_first=True, padding_value=0)
    
    # Concat pixel values (typically batched directly if images are resized identically, else list)
    # Qwen-VL processor typically outputs flat pixel_values, so we can concatenate them.
    # Let's concatenate them since that's what the model expects.
    out["pixel_values_A"] = torch.cat([b["pixel_values_A"] for b in batch], dim=0)
    out["pixel_values_B"] = torch.cat([b["pixel_values_B"] for b in batch], dim=0)
    
    if "image_grid_thw_A" in batch[0]:
        out["image_grid_thw_A"] = torch.cat([b["image_grid_thw_A"] for b in batch], dim=0)
        out["image_grid_thw_B"] = torch.cat([b["image_grid_thw_B"] for b in batch], dim=0)
        
    return out

def main(args):
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    print(f"--- Initializing LENS Training Pipeline ---")
    print(f"Mode: {args.mode.upper()} (Head-only vs LoRA)")
    
    # PyTorch performance optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Enforcing CUDA as requested
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available on this machine. Falling back to CPU for testing purposes only.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # 1. Load Model (Dual-Head VLM)
    # Use a real multimodal VL backbone so image differences actually affect the scores
    model = LENS(model_name=args.model_name, num_error_classes=3, mode=args.mode, unfreeze_layers=args.unfreeze_layers)
    
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
    lr = 2e-5 if args.mode in {"lora", "lora_layer"} else 5e-5
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 4. Initialize Multi-Task Loss Functions
    # The Margin Ranking Loss is for the Score Head (Preference)
    # We use a standard margin (e.g., 0.1) for binary preference learning (A vs B)
    # A smaller margin helps the model push the scores apart more easily when using bfloat16
    ranking_loss_fn = nn.MarginRankingLoss(margin=0.1)
    
    # The BCEWithLogitsLoss is for the Classification Head (Diagnostic Taxonomy)
    # Applies binary cross-entropy on the 3 orthogonal diagnostic dimensions (Existence, Appearance, Interaction)
    classification_loss_fn = nn.BCEWithLogitsLoss()
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 4. Load PrismBench Data
    train_path = os.path.join(os.path.dirname(__file__), "../data_v1/train_v1.json")
    val_path = os.path.join(os.path.dirname(__file__), "../data_v1/val_v1.json")
    
    train_dataset = PrismBenchDataset(json_path=train_path, processor=processor, image_size=args.image_size)
    val_dataset = PrismBenchDataset(json_path=val_path, processor=processor, image_size=args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    # 5. Auto-Scale Logic (Optional)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    if args.auto_scale:
        # Determine model size category based on name
        is_large_model = "9b" in args.model_name.lower() or "4b" in args.model_name.lower()
        is_medium_model = "2b" in args.model_name.lower()
        
        # Base target effective batch size on dataset size
        if train_size < 2000:
            target_ebs = 8
        elif train_size < 10000:
            target_ebs = 16
        else:
            target_ebs = 32
            
        # Adjust target EBS for larger models (they often need larger batches to smooth gradients and smaller learning rates)
        # Also prevents excessive accumulation steps which can cause precision issues in bf16 over many steps
        if is_large_model:
            target_ebs = min(64, target_ebs * 2) # e.g. 4B/9B on 40k data will aim for EBS 64
        
        args.grad_accum_steps = max(1, target_ebs // args.batch_size)
        actual_ebs = args.batch_size * args.grad_accum_steps
        
        # Determine target update steps based on model size and mode
        # Larger models / LoRA need more steps to converge than small models / head-only
        base_target_steps = 3000
        if is_large_model:
            base_target_steps = 5000
        elif is_medium_model:
            base_target_steps = 4000
            
        if args.mode in {"lora", "lora_layer"}:
            base_target_steps = int(base_target_steps * 1.5) # LoRA needs more iterations
            
        steps_per_epoch = max(1, train_size // actual_ebs)
        
        # Calculate epochs, bounding between reasonable limits
        calculated_epochs = base_target_steps // steps_per_epoch
        args.epochs = max(3, min(20, calculated_epochs))
        
        print("\n" + "="*50)
        print(f"📊 AUTO-SCALING TRIGGERED")
        print(f"   - Model               : {args.model_name} (Large: {is_large_model}, Medium: {is_medium_model})")
        print(f"   - Dataset Size        : {train_size} train, {val_size} val")
        print(f"   - Physical Batch Size : {args.batch_size} (VRAM constraint)")
        print(f"   - Target Effective BS : {target_ebs}")
        print(f"   - Grad Accum Steps    : {args.grad_accum_steps}")
        print(f"   - Actual Effective BS : {actual_ebs}")
        print(f"   - Target Updates      : ~{base_target_steps}")
        print(f"   - Epochs Calculated   : {args.epochs} (Total Updates: {args.epochs * steps_per_epoch})")
        print("="*50 + "\n")
    
    # Loss weightings
    alpha = args.alpha # Weight for preference ranking loss
    beta = args.beta   # Weight for diagnostic classification loss
    num_epochs = args.epochs
    
    print(f"Starting Diagnostic-Aware Joint Training Loop. Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, Epochs: {num_epochs}")
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # ---------------- TRAIN ----------------
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} [TRAIN] ---")
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            
            kwargs_A = {
                "input_ids": batch["input_ids_A"].to(device),
                "attention_mask": batch["attention_mask_A"].to(device),
                "pixel_values": batch["pixel_values_A"].to(device).to(torch.bfloat16)
            }
            if "image_grid_thw_A" in batch:
                kwargs_A["image_grid_thw"] = batch["image_grid_thw_A"].to(device)
            if "mm_token_type_ids_A" in batch:
                kwargs_A["mm_token_type_ids"] = batch["mm_token_type_ids_A"].to(device)
            
            kwargs_B = {
                "input_ids": batch["input_ids_B"].to(device),
                "attention_mask": batch["attention_mask_B"].to(device),
                "pixel_values": batch["pixel_values_B"].to(device).to(torch.bfloat16)
            }
            if "image_grid_thw_B" in batch:
                kwargs_B["image_grid_thw"] = batch["image_grid_thw_B"].to(device)
            if "mm_token_type_ids_B" in batch:
                kwargs_B["mm_token_type_ids"] = batch["mm_token_type_ids_B"].to(device)
            
            # Forward Pass A (Pass kwargs directly to model)
            score_A, logits_A = model(**kwargs_A)
            
            # Forward Pass B (Pass kwargs directly to model)
            score_B, logits_B = model(**kwargs_B)

            labels = batch["preference_label"].to(device).float()
            loss_pref = ranking_loss_fn(score_A.squeeze(-1), score_B.squeeze(-1), labels)
            
            targets_A = batch["category_scores_A"].to(device).float()
            targets_B = batch["category_scores_B"].to(device).float()
            
            loss_cls_A = classification_loss_fn(logits_A, targets_A)
            loss_cls_B = classification_loss_fn(logits_B, targets_B)
            loss_cls = (loss_cls_A + loss_cls_B) / 2.0
            
            total_loss = alpha * loss_pref + beta * loss_cls
            scaled_loss = total_loss / args.grad_accum_steps
            
            loss_value = total_loss.item()
            pref_val = loss_pref.item()
            cls_val = loss_cls.item()
            score_A_mean = score_A.mean().item()
            score_B_mean = score_B.mean().item()
            score_gap_mean = (score_A.squeeze(-1) - score_B.squeeze(-1)).abs().mean().item()
            
            scaled_loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Explicitly clear batch tensors to free up VRAM during loop
            del kwargs_A, kwargs_B, score_A, score_B, logits_A, logits_B, loss_pref, loss_cls_A, loss_cls_B, loss_cls, total_loss, scaled_loss
            
            epoch_loss += loss_value
            
            print(f"Step {step+1}/{len(train_loader)} | Pref Rank Loss: {pref_val:.4f} | "
                  f"Cls Loss: {cls_val:.4f} | Total Loss: {loss_value:.4f} | "
                  f"ScoreA: {score_A_mean:.4f} | ScoreB: {score_B_mean:.4f} | Gap: {score_gap_mean:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        print(f"--- Epoch {epoch+1}/{num_epochs} [VALIDATION] ---")
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                kwargs_A = {
                    "input_ids": batch["input_ids_A"].to(device),
                    "attention_mask": batch["attention_mask_A"].to(device),
                    "pixel_values": batch["pixel_values_A"].to(device).to(torch.bfloat16)
                }
                if "image_grid_thw_A" in batch:
                    kwargs_A["image_grid_thw"] = batch["image_grid_thw_A"].to(device)
                if "mm_token_type_ids_A" in batch:
                    kwargs_A["mm_token_type_ids"] = batch["mm_token_type_ids_A"].to(device)
                
                kwargs_B = {
                    "input_ids": batch["input_ids_B"].to(device),
                    "attention_mask": batch["attention_mask_B"].to(device),
                    "pixel_values": batch["pixel_values_B"].to(device).to(torch.bfloat16)
                }
                if "image_grid_thw_B" in batch:
                    kwargs_B["image_grid_thw"] = batch["image_grid_thw_B"].to(device)
                if "mm_token_type_ids_B" in batch:
                    kwargs_B["mm_token_type_ids"] = batch["mm_token_type_ids_B"].to(device)
    
                # Forward Pass A & B
                score_A, logits_A = model(**kwargs_A)
                score_B, logits_B = model(**kwargs_B)
                
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
        
        # Track Best Checkpoint
        if avg_val_loss < best_val_loss:
            print(f"🌟 New Best Validation Loss: {avg_val_loss:.4f} (Previous: {best_val_loss:.4f})")
            best_val_loss = avg_val_loss
            best_dir = os.path.join(os.path.dirname(__file__), f"../outputs/LENS-v1-{args.mode}-best")
            model.save_pretrained(best_dir)
            print(f"🌟 Best checkpoint updated at: {os.path.abspath(best_dir)}")

    print("\nTraining Completed successfully!")
    print(f"You can now upload the contents of {best_dir} directly to Hugging Face 🤗")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LENS Metric Model Training Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-0.8B",
                        help="Backbone model name. Use the intended multimodal backbone for image-conditioned scoring.")
    parser.add_argument("--mode", type=str, choices=["head_only", "lora", "partial", "layer_only", "lora_layer", "full"], default="lora", 
                        help="Training Mode: 'layer_only' (only unfreeze the top N layers), 'lora_layer' (LoRA plus top N layers), 'lora', 'head_only', 'partial' (alias of layer_only), or 'full'.")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of top layers to unfreeze for layer_only / lora_layer / partial")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Siamese training")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--auto_scale", action="store_true", help="Dynamically scale grad_accum_steps and epochs based on dataset size")
    parser.add_argument("--image_size", type=int, default=512, help="Square image size used for resize-and-pad before processor encoding")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Ranking Loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for Classification Loss")
    args = parser.parse_args()
    main(args)
