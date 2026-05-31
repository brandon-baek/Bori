import json
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from rich.console import Console

console = Console()

def initialize_model_from_config(config_path: str, device: str = "cpu", vocab_size: int = None):
    """
    Initializes a new model randomly from the configuration for pre-training.
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    config = Qwen2Config.from_dict(config_dict)
    if vocab_size is not None:
        console.print(f"[bold yellow]Aligning model vocab_size with tokenizer: {vocab_size} (was {config.vocab_size})[/bold yellow]")
        config.vocab_size = vocab_size
    
    # Enable Flash Attention 2 if on supported hardware (Ampere+). T4 doesn't support it,
    # so we will rely on sdpa (Scaled Dot Product Attention) which PyTorch 2.0+ uses by default.
    # config._attn_implementation = "sdpa"
    
    console.print("[bold yellow]Initializing model weights...[/bold yellow]")
    model = Qwen2ForCausalLM(config)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print(f"[bold green]Model initialized successfully![/bold green]")
    console.print(f"Total Parameters: {total_params / 1e6:.2f}M")
    console.print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
    
    return model

def load_model_from_checkpoint(checkpoint_path: str, device_map="auto"):
    """
    Loads an already pre-trained model for Fine-Tuning or Inference.
    """
    model = Qwen2ForCausalLM.from_pretrained(
        checkpoint_path, 
        device_map=device_map,
        torch_dtype=torch.float16 # Use float16 for T4 compatibility
    )
    return model
