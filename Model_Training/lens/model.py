import torch
import torch.nn as nn

class LENS(nn.Module):
    """
    LENS (Localized Entanglement Navigation and Scoring) Architecture
    - Backbone: Qwen3.5-0.8B (via Unsloth for correct Vision-Language forward pass)
    - Score Head: 1D scalar for preference ranking (Margin Ranking Loss)
    - Classification Head: 3D logits for diagnostic taxonomy (BCEWithLogitsLoss)
    """
    def __init__(self, model_name="unsloth/Qwen3.5-0.8B", num_error_classes=3, mode="lora", unfreeze_layers=4):
        super(LENS, self).__init__()
        
        print(f"Loading VLM Backbone: {model_name} in [{mode.upper()}] mode using Unsloth on CUDA...")
        
        # We must use FastVisionModel for Qwen3.5 multimodal capabilities in Unsloth
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError("Unsloth is not installed. Please run: pip install unsloth unsloth_zoo")

        # Load base model via Unsloth (handles multi-modal correctly and saves 50% VRAM)
        self.base_model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=False,      # Don't use 4bit for Qwen3.5 per Unsloth docs
            load_in_16bit=True,      # Use bf16/16bit
            use_gradient_checkpointing="unsloth", # Use unsloth's optimized checkpointing
        )

        def locate_transformer_layers(module_root):
            candidate_paths = [
                "model.layers",
                "model.model.layers",
                "language_model.layers",
                "language_model.model.layers",
                "layers",
            ]

            def resolve_attr_path(obj, path):
                cur = obj
                for part in path.split("."):
                    if not hasattr(cur, part):
                        return None
                    cur = getattr(cur, part)
                return cur

            for path in candidate_paths:
                layers = resolve_attr_path(module_root, path)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    return layers, path

            module_lists = []
            for name, submodule in module_root.named_modules():
                if isinstance(submodule, nn.ModuleList) and len(submodule) > 0:
                    module_lists.append((name, submodule))
            if module_lists:
                module_lists.sort(key=lambda item: len(item[1]), reverse=True)
                return module_lists[0][1], module_lists[0][0]

            return None, None

        def unfreeze_last_layers(module_root, num_layers):
            layers, layer_path = locate_transformer_layers(module_root)
            if layers is None:
                print("Warning: Could not automatically detect transformer layers for selective unfreezing.")
                return False

            actual_num_layers = min(num_layers, len(layers))
            for layer in layers[-actual_num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            print(f"Successfully unfroze the last {actual_num_layers} layers via `{layer_path}`.")
            return True
        
        self.mode = mode
        if mode == "lora":
            print("Injecting LoRA adapters via Unsloth (Optimized for VLM)...")
            self.backbone = FastVisionModel.get_peft_model(
                self.base_model,
                finetune_vision=True,     # Essential: Allow vision encoder to train
                finetune_language=True,   # Essential: Allow language model to train
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
            )
            self.backbone.print_trainable_parameters()
            
        elif mode in {"partial", "layer_only"}:
            self.base_model.gradient_checkpointing_enable()
            print(f"Freezing backbone and unfreezing the last {unfreeze_layers} Transformer layers only...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False

            unfreeze_last_layers(self.backbone, unfreeze_layers)

        elif mode == "lora_layer":
            print(f"Injecting LoRA adapters and unfreezing the last {unfreeze_layers} Transformer layers...")
            self.backbone = FastVisionModel.get_peft_model(
                self.base_model,
                finetune_vision=True,
                finetune_language=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
            )
            unfreeze_last_layers(self.backbone, unfreeze_layers)
            self.backbone.print_trainable_parameters()
                
        elif mode == "head_only":
            print("Freezing entirely Backbone for Head-only training...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif mode == "full":
            print("WARNING: Full Parameter Fine-Tuning.")
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
        if self.mode in {"lora", "lora_layer"}:
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
