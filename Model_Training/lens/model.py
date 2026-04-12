import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
from peft import LoraConfig, get_peft_model, TaskType

class LENS(nn.Module):
    """
    LENS (Localized Entanglement Navigation and Scoring) Architecture
    - Backbone: Qwen3.5-9B-Base (Frozen, LoRA, Partial Layers, or Full)
    - Score Head: 1D scalar for preference ranking (Margin Ranking Loss)
    - Classification Head: 3D logits for diagnostic taxonomy (BCEWithLogitsLoss)
      * Index 0: Existence (0=pass, 1=fail)
      * Index 1: Appearance (0=pass, 1=fail)
      * Index 2: Interaction (0=pass, 1=fail)
    """
    def __init__(self, model_name="Qwen/Qwen3.5-9B-Base", num_error_classes=3, mode="lora", unfreeze_layers=4):
        super(LENS, self).__init__()
        
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        
        print(f"Loading VLM Backbone: {model_name} in [{mode.upper()}] mode on CUDA...")
        
        # Determine the correct Auto class based on the model name
        if "VL" in model_name.upper():
            auto_model_cls = AutoModelForImageTextToText
        else:
            auto_model_cls = AutoModelForCausalLM
            print("WARNING: Using AutoModelForCausalLM. If this model has a vision encoder, ensure pixel_values are not ignored.")
            
        self.base_model = auto_model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Disable KV cache during training to save VRAM (must be set on config, not init)
        self.base_model.config.use_cache = False
        
        self.mode = mode
        if mode == "lora":
            # Enable gradient checkpointing to drastically reduce VRAM usage
            self.base_model.gradient_checkpointing_enable()
            print("Injecting LoRA adapters into Qwen3.5 (Optimal for VLM)...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16, 
                lora_alpha=32,
                lora_dropout=0.05,
                # Reduce targeted modules slightly to save VRAM on 9B model
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.backbone = get_peft_model(self.base_model, peft_config)
            self.backbone.print_trainable_parameters()
            
        elif mode == "partial":
            # Enable gradient checkpointing
            self.base_model.gradient_checkpointing_enable()
            print(f"Freezing backbone but UNFREEZING the last {unfreeze_layers} Transformer layers...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Attempt to unfreeze the last N layers of the Qwen decoder
            if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
                layers = self.backbone.model.layers
                for layer in layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Successfully unfroze the last {unfreeze_layers} layers. Total layers: {len(layers)}")
            else:
                print("Warning: Could not automatically detect 'model.layers' for partial unfreezing.")
                
        elif mode == "head_only":
            print("Freezing entirely Backbone for Head-only training...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif mode == "full":
            print("WARNING: Full Parameter Fine-Tuning (Requires massive VRAM, e.g., multiple A100s).")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = True
                
        else:
            raise ValueError(f"Unknown training mode: {mode}")
                
        # Robust hidden_size extraction for Early-Fusion Multimodal / New Architectures
        config = self.backbone.config
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "hidden_dim"):
            hidden_size = config.hidden_dim
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        else:
            # Fallback for Qwen3.5-9B-Base according to official specs
            hidden_size = 4096
            
        print(f"Detected backbone hidden dimension: {hidden_size}")
        
        # Dual-Head Architecture (Always Trainable)
        print("Initializing Score Head and Classification Head...")
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(torch.bfloat16).to(self.backbone.device)
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_error_classes)
        ).to(torch.bfloat16).to(self.backbone.device)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Extracts global contextual feature from the VLM and passes it to both heads.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Extract the hidden state of the LAST VALID token (aggregates the multimodal context)
        # We must use attention_mask to find the true last token, avoiding PAD tokens
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1).long() - 1
        last_hidden_state = outputs.hidden_states[-1][torch.arange(batch_size, device=input_ids.device), sequence_lengths, :]
        
        # Pass through the two heads
        # Force float32 for loss calculation to prevent underflow in bfloat16 causing identical scores
        score = self.score_head(last_hidden_state).to(torch.float32)
        logits = self.classification_head(last_hidden_state).to(torch.float32)
        
        return score, logits

    def save_pretrained(self, save_directory):
        """
        Saves the model weights so it can be loaded later or uploaded to Hugging Face Hub.
        - If 'head_only': Saves only the MLP heads (score_head and classification_head).
        - If 'lora': Saves the LoRA adapter weights AND the MLP heads.
        The base Qwen model is NEVER saved (it will be downloaded on-the-fly by users).
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"Saving LENS model to {save_directory}...")
        
        # 1. Save the Custom Heads (Always)
        heads_state_dict = {
            "score_head": self.score_head.state_dict(),
            "classification_head": self.classification_head.state_dict()
        }
        torch.save(heads_state_dict, os.path.join(save_directory, "lens_heads.pt"))
        print("- Saved Custom MLP Heads (lens_heads.pt)")
        
        # 2. Save the LoRA Weights (If applicable)
        if self.mode == "lora":
            # PEFT has a built-in method to save only the LoRA weights, not the 9B base model
            self.backbone.save_pretrained(os.path.join(save_directory, "lora_adapter"))
            print("- Saved LoRA Adapters (lora_adapter/)")
            
        # 3. Save Config for Inference
        config = {
            "base_model_name": self.base_model.name_or_path,
            "mode": self.mode,
            "num_error_classes": self.classification_head[-1].out_features
        }
        import json
        with open(os.path.join(save_directory, "lens_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print("- Saved Model Config (lens_config.json)")
        print("Model saved successfully! Ready for Hugging Face Hub upload.")
