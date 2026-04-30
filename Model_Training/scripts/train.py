import os
import sys
import gc
import math
import argparse
import unsloth
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence

# Safe default Hugging Face cache settings for server-side training.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/huggingface_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Fix Python Path for 'lens' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lens.model import LENS
from lens.dataset import PrismBenchDataset

_ = unsloth  # Keep Unsloth imported before transformers/peft patch sites.

def create_collate_fn(pad_token_id=0):
    def custom_collate_fn(batch):
        """Handles variable-length token sequences by padding them."""
        out = {}
        
        out["task_id"] = [b["task_id"] for b in batch]
        out["preference_label"] = torch.stack([b["preference_label"] for b in batch])
        out["category_scores_A"] = torch.stack([b["category_scores_A"] for b in batch])
        out["category_scores_B"] = torch.stack([b["category_scores_B"] for b in batch])
        
        # Pad sequences using the correct tokenizer pad_token_id
        out["input_ids_A"] = pad_sequence([b["input_ids_A"] for b in batch], batch_first=True, padding_value=pad_token_id)
        out["attention_mask_A"] = pad_sequence([b["attention_mask_A"] for b in batch], batch_first=True, padding_value=0)
        out["input_ids_B"] = pad_sequence([b["input_ids_B"] for b in batch], batch_first=True, padding_value=pad_token_id)
        out["attention_mask_B"] = pad_sequence([b["attention_mask_B"] for b in batch], batch_first=True, padding_value=0)
        
        if "mm_token_type_ids_A" in batch[0]:
            out["mm_token_type_ids_A"] = pad_sequence([b["mm_token_type_ids_A"] for b in batch], batch_first=True, padding_value=0)
            out["mm_token_type_ids_B"] = pad_sequence([b["mm_token_type_ids_B"] for b in batch], batch_first=True, padding_value=0)
        
        # Concat pixel values
        out["pixel_values_A"] = torch.cat([b["pixel_values_A"] for b in batch], dim=0)
        out["pixel_values_B"] = torch.cat([b["pixel_values_B"] for b in batch], dim=0)
        
        if "image_grid_thw_A" in batch[0]:
            out["image_grid_thw_A"] = torch.cat([b["image_grid_thw_A"] for b in batch], dim=0)
            out["image_grid_thw_B"] = torch.cat([b["image_grid_thw_B"] for b in batch], dim=0)
            
        return out
    return custom_collate_fn

def calculate_auto_scaling(args, train_size, val_size):
    """Calculates and updates grad_accum_steps and epochs dynamically."""
    is_large_model = "9b" in args.model_name.lower() or "4b" in args.model_name.lower()
    is_medium_model = "2b" in args.model_name.lower()
    
    target_ebs = 8 if train_size < 2000 else (16 if train_size < 10000 else 32)
    
    if is_large_model:
        target_ebs = min(64, target_ebs * 2)
    
    args.grad_accum_steps = max(1, target_ebs // args.batch_size)
    actual_ebs = args.batch_size * args.grad_accum_steps
    
    base_target_steps = 5000 if is_large_model else (4000 if is_medium_model else 3000)
    
    if args.mode in {"lora", "lora_layer"}:
        base_target_steps = int(base_target_steps * 1.5)
        
    steps_per_epoch = max(1, train_size // actual_ebs)
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

def prepare_model_inputs(batch, device, suffix):
    """Extracts and formats inputs for a specific forward pass branch (A or B)."""
    kwargs = {
        "input_ids": batch[f"input_ids_{suffix}"].to(device),
        "attention_mask": batch[f"attention_mask_{suffix}"].to(device),
        "pixel_values": batch[f"pixel_values_{suffix}"].to(device).to(torch.bfloat16)
    }
    if f"image_grid_thw_{suffix}" in batch:
        kwargs["image_grid_thw"] = batch[f"image_grid_thw_{suffix}"].to(device)
    if f"mm_token_type_ids_{suffix}" in batch:
        kwargs["mm_token_type_ids"] = batch[f"mm_token_type_ids_{suffix}"].to(device)
    return kwargs

def get_epoch_loss_weights(args, epoch):
    """Optionally anneal the classification weight as training progresses."""
    alpha = args.alpha
    beta = args.beta
    if args.adaptive_beta and args.epochs > 1:
        progress = epoch / max(1, args.epochs - 1)
        beta = args.beta + (args.beta_final - args.beta) * progress
    return alpha, beta

def compute_loss(model, batch, device, ranking_loss_fn, classification_loss_fn, alpha, beta):
    """Performs forward passes and calculates the multi-task loss."""
    kwargs_A = prepare_model_inputs(batch, device, "A")
    kwargs_B = prepare_model_inputs(batch, device, "B")
    
    score_A, logits_A = model(**kwargs_A)
    score_B, logits_B = model(**kwargs_B)

    labels = batch["preference_label"].to(device).float()
    # Force float32 for numerical stability in margin ranking loss
    loss_pref = ranking_loss_fn(score_A.squeeze(-1).float(), score_B.squeeze(-1).float(), labels)
    
    targets_A = batch["category_scores_A"].to(device).float()
    targets_B = batch["category_scores_B"].to(device).float()
    
    # Force float32 for BCEWithLogitsLoss to prevent underflow/overflow during Sigmoid
    loss_cls_A = classification_loss_fn(logits_A.float(), targets_A)
    loss_cls_B = classification_loss_fn(logits_B.float(), targets_B)
    loss_cls = (loss_cls_A + loss_cls_B) / 2.0
    
    total_loss = alpha * loss_pref + beta * loss_cls
    
    metrics = {
        "pref_loss": loss_pref.item(),
        "cls_loss": loss_cls.item(),
        "total_loss": total_loss.item(),
        "score_A_mean": score_A.mean().item(),
        "score_B_mean": score_B.mean().item(),
        "score_gap_mean": (score_A.squeeze(-1) - score_B.squeeze(-1)).abs().mean().item()
    }
    
    return total_loss, metrics

def train_epoch(model, train_loader, optimizer, scheduler, device, ranking_loss_fn, classification_loss_fn, args, epoch):
    """Runs a single training epoch."""
    model.train()
    alpha, beta = get_epoch_loss_weights(args, epoch)
    print(f"\n--- Epoch {epoch+1}/{args.epochs} [TRAIN] ---")
    print(f"Loss Weights | alpha={alpha:.4f} beta={beta:.4f}")
    epoch_loss = 0.0
    epoch_pref_loss = 0.0
    epoch_cls_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        total_loss, metrics = compute_loss(model, batch, device, ranking_loss_fn, classification_loss_fn, alpha, beta)
        
        # Calculate actual accumulation steps for the tail batch
        actual_accum_steps = min(args.grad_accum_steps, len(train_loader) - step + (step % args.grad_accum_steps))
        scaled_loss = total_loss / actual_accum_steps
        scaled_loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        epoch_loss += metrics["total_loss"]
        epoch_pref_loss += metrics["pref_loss"]
        epoch_cls_loss += metrics["cls_loss"]
        
        print(f"Step {step+1}/{len(train_loader)} | Pref Rank Loss: {metrics['pref_loss']:.4f} | "
              f"Cls Loss: {metrics['cls_loss']:.4f} | Total Loss: {metrics['total_loss']:.4f} | "
              f"ScoreA: {metrics['score_A_mean']:.4f} | ScoreB: {metrics['score_B_mean']:.4f} | Gap: {metrics['score_gap_mean']:.4f}")

        # Local variables are naturally GC'd at loop end, but we explicitly clear tensors to guarantee VRAM hygiene
        del batch, total_loss, scaled_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "total_loss": epoch_loss / len(train_loader),
        "pref_loss": epoch_pref_loss / len(train_loader),
        "cls_loss": epoch_cls_loss / len(train_loader),
        "alpha": alpha,
        "beta": beta,
    }

def validate_epoch(model, val_loader, device, ranking_loss_fn, classification_loss_fn, args, epoch):
    """Runs validation for a single epoch."""
    model.eval()
    print(f"--- Epoch {epoch+1}/{args.epochs} [VALIDATION] ---")
    alpha, beta = get_epoch_loss_weights(args, epoch)
    val_loss = 0.0
    val_pref_loss = 0.0
    val_cls_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            total_loss, metrics = compute_loss(model, batch, device, ranking_loss_fn, classification_loss_fn, alpha, beta)
            val_loss += total_loss.item()
            val_pref_loss += metrics["pref_loss"]
            val_cls_loss += metrics["cls_loss"]
            
            del batch, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_total_loss = val_loss / len(val_loader)
    avg_pref_loss = val_pref_loss / len(val_loader)
    avg_cls_loss = val_cls_loss / len(val_loader)
    monitor_loss = avg_pref_loss + args.monitor_cls_weight * avg_cls_loss
    return {
        "total_loss": avg_total_loss,
        "pref_loss": avg_pref_loss,
        "cls_loss": avg_cls_loss,
        "monitor_loss": monitor_loss,
        "alpha": alpha,
        "beta": beta,
    }

def main(args):
    torch.cuda.empty_cache()
    gc.collect()

    print(f"--- Initializing LENS Training Pipeline ---")
    print(f"Mode: {args.mode.upper()} (Head-only vs LoRA)")
    
    # PyTorch performance optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is not available on this machine. Falling back to CPU for testing purposes only.")
    print(f"Using device: {device}")
    
    # 1. Load Model (Dual-Head VLM)
    model = LENS(model_name=args.model_name, num_error_classes=3, mode=args.mode, unfreeze_layers=args.unfreeze_layers)
    if torch.cuda.is_available():
        device = model.backbone.device
    
    # 2. Extract Trainable Parameters & Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = 2e-5 if args.mode in {"lora", "lora_layer"} else 5e-5
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 3. Loss Functions
    ranking_loss_fn = nn.MarginRankingLoss(margin=0.1)
    classification_loss_fn = nn.BCEWithLogitsLoss()
    
    # 4. Data Setup
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    pad_token_id = processor.tokenizer.pad_token_id
    
    train_dataset = PrismBenchDataset(json_path=args.train_path, processor=processor, image_size=args.image_size)
    val_dataset = PrismBenchDataset(json_path=args.val_path, processor=processor, image_size=args.image_size)
    
    collate_fn = create_collate_fn(pad_token_id=pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    
    # 5. Auto-Scale
    if args.auto_scale:
        calculate_auto_scaling(args, len(train_dataset), len(val_dataset))
        
    # 5.5 Setup LR Scheduler
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.05), # 5% warmup
        num_training_steps=total_steps
    )
    
    print(f"Starting Diagnostic-Aware Joint Training Loop. Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, Epochs: {args.epochs}")
    
    # 6. Training Loop
    best_monitor_loss = float("inf")
    epochs_without_improvement = 0
    safe_model_name = args.model_name.replace("/", "_")
    outputs_dir = os.path.abspath(args.outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)
    best_dir = os.path.join(outputs_dir, f"{safe_model_name}-{args.mode}-best")
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, ranking_loss_fn, classification_loss_fn, args, epoch)
        val_metrics = validate_epoch(model, val_loader, device, ranking_loss_fn, classification_loss_fn, args, epoch)
        
        print(
            f"Epoch {epoch+1} Summary | "
            f"Train Total: {train_metrics['total_loss']:.4f} | Train Pref: {train_metrics['pref_loss']:.4f} | Train Cls: {train_metrics['cls_loss']:.4f} | "
            f"Val Total: {val_metrics['total_loss']:.4f} | Val Pref: {val_metrics['pref_loss']:.4f} | Val Cls: {val_metrics['cls_loss']:.4f} | "
            f"Monitor: {val_metrics['monitor_loss']:.4f}"
        )
        
        save_dir = os.path.join(outputs_dir, f"{safe_model_name}-{args.mode}-epoch{epoch+1}")
        model.save_pretrained(save_dir)
        print(f"Checkpoint saved to: {os.path.abspath(save_dir)}")
        
        if val_metrics["monitor_loss"] < best_monitor_loss - args.early_stopping_min_delta:
            print(f"🌟 New Best Monitor Loss: {val_metrics['monitor_loss']:.4f} (Previous: {best_monitor_loss:.4f})")
            best_monitor_loss = val_metrics["monitor_loss"]
            epochs_without_improvement = 0
            model.save_pretrained(best_dir)
            print(f"🌟 Best checkpoint updated at: {os.path.abspath(best_dir)}")
        else:
            epochs_without_improvement += 1
            print(f"⏳ No monitor improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= args.early_stopping_patience:
                print("🛑 Early stopping triggered.")
                break

    print("\nTraining Completed successfully!")
    if best_monitor_loss != float("inf"):
        print(f"You can now upload the contents of {best_dir} directly to Hugging Face 🤗")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LENS Metric Model Training Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-0.8B", help="Backbone model name")
    parser.add_argument("--mode", type=str, choices=["head_only", "lora", "partial", "layer_only", "lora_layer", "full"], default="lora", help="Training Mode")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of top layers to unfreeze")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--auto_scale", action="store_true", help="Dynamically scale grad_accum_steps and epochs based on dataset size")
    parser.add_argument("--image_size", type=int, default=512, help="Square image size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Ranking Loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for Classification Loss")
    parser.add_argument("--adaptive_beta", action=argparse.BooleanOptionalAction, default=True, help="Linearly anneal beta over epochs")
    parser.add_argument("--beta_final", type=float, default=0.25, help="Final beta value when adaptive beta is enabled")
    parser.add_argument("--monitor_cls_weight", type=float, default=0.5, help="Fixed cls weight used for best-checkpoint monitoring")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm; set <=0 to disable")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Stop if monitor loss does not improve for N epochs")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-3, help="Minimum monitor improvement to reset patience")
    parser.add_argument("--train_path", type=str, default=os.path.join(os.path.dirname(__file__), "../data_v1/train_v1.json"), help="Path to training data")
    parser.add_argument("--val_path", type=str, default=os.path.join(os.path.dirname(__file__), "../data_v1/val_v1.json"), help="Path to validation data")
    parser.add_argument("--outputs_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../outputs"), help="Directory where checkpoints will be written")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    args = parser.parse_args()
    main(args)
