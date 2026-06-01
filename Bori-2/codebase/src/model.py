import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

console = Console()

def load_pretrained_for_cpt(model_name: str, merged_tokenizer_path: str, original_tokenizer_path: str = None):
    console.print(f"[bold cyan]Loading pretrained model: {model_name}[/bold cyan]")
    
    orig_tokenizer_path = original_tokenizer_path or model_name
    original_tokenizer = AutoTokenizer.from_pretrained(orig_tokenizer_path)
    original_vocab_size = len(original_tokenizer)
    
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_tokenizer_path)
    new_vocab_size = len(merged_tokenizer)
    num_new_tokens = new_vocab_size - original_vocab_size
    
    console.print(f"Original vocab size: {original_vocab_size}")
    console.print(f"Merged vocab size:   {new_vocab_size}")
    console.print(f"New tokens added:    {num_new_tokens}")
    
    # Always load in fp32 — the Trainer's AMP (fp16=True) handles mixed-precision
    # casting during training.  Loading in fp16 + AMP causes:
    #   "ValueError: Attempting to unscale FP16 gradients."
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    original_params = sum(p.numel() for p in model.parameters())
    
    if num_new_tokens > 0:
        console.print(f"\n[bold yellow]Resizing embeddings: {original_vocab_size} → {new_vocab_size}[/bold yellow]")
        model.resize_token_embeddings(new_vocab_size)
        _initialize_new_embeddings_eeve(
            model, original_tokenizer, merged_tokenizer, 
            original_vocab_size, num_new_tokens
        )
    else:
        console.print("[green]No new tokens to add — vocab sizes match.[/green]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print(f"\n[bold green]Model loaded successfully![/bold green]")
    console.print(f"Original Parameters: {original_params / 1e6:.2f}M")
    console.print(f"Total Parameters:    {total_params / 1e6:.2f}M")
    console.print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
    console.print(f"Parameter increase from vocab expansion: {(total_params - original_params) / 1e6:.2f}M")
    
    return model, merged_tokenizer


def _initialize_new_embeddings_eeve(model, original_tokenizer, merged_tokenizer, 
                                      original_vocab_size, num_new_tokens):
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    
    with torch.no_grad():
        existing_input_mean = input_embeddings.weight[:original_vocab_size].mean(dim=0)
        existing_output_mean = output_embeddings.weight[:original_vocab_size].mean(dim=0)
    
    subword_init_count = 0
    fallback_count = 0
    
    with torch.no_grad():
        for i in range(num_new_tokens):
            new_token_id = original_vocab_size + i
            new_token_str = merged_tokenizer.convert_ids_to_tokens(new_token_id)
            
            if new_token_str is None:
                input_embeddings.weight[new_token_id] = existing_input_mean.clone()
                output_embeddings.weight[new_token_id] = existing_output_mean.clone()
                fallback_count += 1
                continue
            
            try:
                decoded_str = merged_tokenizer.decode([new_token_id])
                subword_ids = original_tokenizer.encode(decoded_str, add_special_tokens=False)
            except Exception:
                subword_ids = []
            
            if len(subword_ids) > 0:
                subword_embeds = input_embeddings.weight[subword_ids]
                input_embeddings.weight[new_token_id] = subword_embeds.mean(dim=0)
                
                subword_out_embeds = output_embeddings.weight[subword_ids]
                output_embeddings.weight[new_token_id] = subword_out_embeds.mean(dim=0)
                
                subword_init_count += 1
            else:
                input_embeddings.weight[new_token_id] = existing_input_mean.clone()
                output_embeddings.weight[new_token_id] = existing_output_mean.clone()
                fallback_count += 1
    
    console.print(f"[green]Embedding initialization complete:[/green]")
    console.print(f"  Subword-based (EEVE): {subword_init_count} tokens")
    console.print(f"  Fallback (global mean): {fallback_count} tokens")


def load_model_from_checkpoint(checkpoint_path: str, tokenizer_path: str = None, device_map="auto"):
    console.print(f"[bold cyan]Loading model from checkpoint: {checkpoint_path}[/bold cyan]")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    
    tokenizer = None
    tok_path = tokenizer_path or checkpoint_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        console.print(f"[green]Tokenizer loaded from {tok_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tokenizer from {tok_path}: {e}[/yellow]")
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold green]Checkpoint loaded: {total_params / 1e6:.2f}M parameters[/bold green]")
    
    return model, tokenizer


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    console.print(f"[bold yellow]Backbone frozen for embedding warm-up[/bold yellow]")
    console.print(f"  Trainable: {trainable / 1e6:.2f}M parameters (embeddings + LM head)")
    console.print(f"  Frozen:    {frozen / 1e6:.2f}M parameters (transformer backbone)")
    
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold green]All parameters unfrozen: {trainable / 1e6:.2f}M trainable[/bold green]")
    
    return model
