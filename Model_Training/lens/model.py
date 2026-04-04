import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

class LENS(nn.Module):
    """
    LENS (Localized Entanglement Navigation and Scoring) Architecture
    - Backbone: Qwen3.5-9B (Frozen or LoRA)
    - Score Head: 1D scalar for preference ranking (Margin Ranking Loss)
    - Classification Head: 4D logits for diagnostic taxonomy (Cross Entropy Loss)
      * Class 0: Perfect Alignment
      * Class 1: Attribute Bleeding
      * Class 2: Semantic Swapping
      * Class 3: Entity Collapse (Homogenization / Missing)
    """
    def __init__(self, model_name="Qwen/Qwen3.5-9B", num_classes=4, use_lora=False):
        super(LENS, self).__init__()
        
        print(f"Loading VLM Backbone: {model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.use_lora = use_lora
        if use_lora:
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
        else:
            print("Freezing Backbone for Head-only training...")
            self.backbone = self.base_model
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        hidden_size = self.backbone.config.hidden_size
        
        # Dual-Head Architecture
        print("Initializing Score Head and Classification Head...")
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
