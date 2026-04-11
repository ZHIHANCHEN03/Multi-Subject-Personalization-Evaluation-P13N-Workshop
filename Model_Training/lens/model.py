import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

class LENS(nn.Module):
    """
    LENS (Localized Entanglement Navigation and Scoring) Architecture
    - Backbone: Qwen3.5-9B (Frozen or LoRA)
    - Score Head: 1D scalar for preference ranking (Margin Ranking Loss)
    - Classification Head: 3D logits for diagnostic taxonomy (BCEWithLogitsLoss)
      * Index 0: Existence (0=pass, 1=fail)
      * Index 1: Appearance (0=pass, 1=fail)
      * Index 2: Interaction (0=pass, 1=fail)
    """
    def __init__(self, model_name="Qwen/Qwen3.5-9B", num_error_classes=3, mode="head_only"):
        super(LENS, self).__init__()
        
        print(f"Loading VLM Backbone: {model_name} in [{mode.upper()}] mode...")
        # device_map="auto" works for CUDA but can cause issues on MPS. 
        # For cross-platform compatibility, we let the external train.py script handle the .to(device) mapping.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None
        )
        
        self.mode = mode
        if mode == "lora":
            print("Injecting LoRA adapters into Qwen3.5...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16, 
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.backbone = get_peft_model(self.base_model, peft_config)
            self.backbone.print_trainable_parameters()
        elif mode == "head_only":
            print("Freezing Backbone for Head-only training...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown training mode: {mode}")
                
        hidden_size = self.backbone.config.hidden_size
        
        # Dual-Head Architecture (Always Trainable)
        print("Initializing Score Head and Classification Head...")
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(torch.bfloat16)
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_error_classes)
        ).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, pixel_values=None):
        """
        Extracts global contextual feature from the VLM and passes it to both heads.
        """
        kwargs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "output_hidden_states": True
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            
        outputs = self.backbone(**kwargs)
        
        # Extract the hidden state of the LAST token (aggregates the multimodal context)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        
        # Pass through the two heads
        score = self.score_head(last_hidden_state)
        logits = self.classification_head(last_hidden_state)
        
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
