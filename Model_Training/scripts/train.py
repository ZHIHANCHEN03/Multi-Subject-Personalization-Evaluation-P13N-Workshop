import os
import sys
import subprocess

# Safe default Hugging Face cache settings for server-side training.
# This prevents quota issues from /workspace-backed Xet storage and
# keeps manual `python scripts/train.py ...` runs consistent.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/huggingface_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 1. Auto-check and install dependencies
def ensure_dependencies():
    required_packages = {"torch": "torch", "peft": "peft", "PIL": "Pillow", "unsloth": "unsloth"}
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing missing dependency: {pip_name}...")
            if pip_name == "unsloth":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth", "unsloth_zoo"])
            else:
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
from transformers import AutoProcessor
from lens.model import LENS
from lens.dataset import PrismBenchDataset


#region debug-point helper: fail-open debug reporting
def _try_debug_event(hypothesis_id, location, msg, data):
    try:
        import json
        import time
        import urllib.request

        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".dbg",
            "ranking-gap-zero.env",
        )
        url = "http://127.0.0.1:7777/event"
        session_id = "ranking-gap-zero"

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                content = f.read().splitlines()
            for line in content:
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1]
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1]
        except OSError:
            pass

        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": int(time.time() * 1000),
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1).read()
    except Exception:
        # Debug reporting must never block or crash training.
        pass
#endregion

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
    lr = 2e-5 if args.mode == "lora" else 1e-4
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
    
    train_dataset = PrismBenchDataset(json_path=train_path, processor=processor)
    val_dataset = PrismBenchDataset(json_path=val_path, processor=processor)
    
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
            
            kwargs_A = {
                "input_ids": batch["input_ids_A"].to(device),
                "attention_mask": batch["attention_mask_A"].to(device),
                "pixel_values": batch["pixel_values_A"].to(device).to(torch.bfloat16)
            }
            if "image_grid_thw_A" in batch:
                kwargs_A["image_grid_thw"] = batch["image_grid_thw_A"].to(device)
            
            kwargs_B = {
                "input_ids": batch["input_ids_B"].to(device),
                "attention_mask": batch["attention_mask_B"].to(device),
                "pixel_values": batch["pixel_values_B"].to(device).to(torch.bfloat16)
            }
            if "image_grid_thw_B" in batch:
                kwargs_B["image_grid_thw"] = batch["image_grid_thw_B"].to(device)
            
            # Forward Pass A (Pass kwargs directly to model)
            score_A, logits_A = model(**kwargs_A)
            
            # Forward Pass B (Pass kwargs directly to model)
            score_B, logits_B = model(**kwargs_B)

            # #region debug-point B:post-forward-output-compare
            if step < 3:
                _try_debug_event(
                    "B",
                    "scripts/train.py:post-forward",
                    "[DEBUG] Post-forward A/B output comparison",
                    {
                        "step": step,
                        "score_a_mean": float(score_A.mean().item()),
                        "score_b_mean": float(score_B.mean().item()),
                        "score_gap_mean": float((score_A.squeeze(-1) - score_B.squeeze(-1)).abs().mean().item()),
                        "logit_gap_mean": float((logits_A - logits_B).abs().mean().item()),
                        "logit_gap_max": float((logits_A - logits_B).abs().max().item()),
                    },
                )
            # #endregion
            
            labels = batch["preference_label"].to(device).float()
            loss_pref = ranking_loss_fn(score_A.squeeze(-1), score_B.squeeze(-1), labels)
            
            targets_A = batch["category_scores_A"].to(device).float()
            targets_B = batch["category_scores_B"].to(device).float()
            
            loss_cls_A = classification_loss_fn(logits_A, targets_A)
            loss_cls_B = classification_loss_fn(logits_B, targets_B)
            loss_cls = (loss_cls_A + loss_cls_B) / 2.0
            
            total_loss = alpha * loss_pref + beta * loss_cls
            loss_value = total_loss.item()
            pref_val = loss_pref.item()
            cls_val = loss_cls.item()
            score_A_mean = score_A.mean().item()
            score_B_mean = score_B.mean().item()
            score_gap_mean = (score_A.squeeze(-1) - score_B.squeeze(-1)).abs().mean().item()
            
            total_loss.backward()
            optimizer.step()
            
            # Explicitly clear batch tensors to free up VRAM during loop
            del kwargs_A, kwargs_B, score_A, score_B, logits_A, logits_B, loss_pref, loss_cls_A, loss_cls_B, loss_cls, total_loss
            
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
                
                kwargs_B = {
                    "input_ids": batch["input_ids_B"].to(device),
                    "attention_mask": batch["attention_mask_B"].to(device),
                    "pixel_values": batch["pixel_values_B"].to(device).to(torch.bfloat16)
                }
                if "image_grid_thw_B" in batch:
                    kwargs_B["image_grid_thw"] = batch["image_grid_thw_B"].to(device)
    
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

    print("\nTraining Completed successfully!")
    print(f"You can now upload the contents of {save_dir} directly to Hugging Face 🤗")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LENS Metric Model Training Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-0.8B",
                        help="Backbone model name. Use the intended multimodal backbone for image-conditioned scoring.")
    parser.add_argument("--mode", type=str, choices=["head_only", "lora", "partial", "full"], default="lora", 
                        help="Training Mode: 'head_only' (freeze all), 'lora' (PEFT on linear layers), 'partial' (unfreeze top N layers), or 'full' (finetune everything).")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of top layers to unfreeze if mode='partial'")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Siamese training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Ranking Loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for Classification Loss")
    args = parser.parse_args()
    main(args)
