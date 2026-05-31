import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def discover_tokenizer():
    """Attempts to auto-detect the merged tokenizer path to save the user from typing it."""
    possible_paths = [
        "./test_output/merged_tokenizer",
        "../test_output/merged_tokenizer",
        "./checkpoints/phase_1b",
        "./sft_checkpoints"
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # bori_v3 root
    possible_paths.append(os.path.join(project_root, "test_output", "merged_tokenizer"))
    
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and any(f.endswith(".json") for f in os.listdir(abs_p)):
            return abs_p
    return None

def main():
    parser = argparse.ArgumentParser(description="Unified W&B Downloader & Interactive CLI Chat for Bori")
    parser.add_argument("--artifact_path", type=str, default=None, help="W&B artifact path 'entity/project/artifact_name'")
    parser.add_argument("--alias", type=str, default="latest", help="Artifact version alias ('latest', 'step-300')")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to merged tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./downloaded_sft_checkpoint", help="Destination download directory")
    args = parser.parse_args()

    print("======================================================================")
    print("🌾 Bori (보리): Unified Downloader & Interactive CLI Chat")
    print("======================================================================\n")

    # 1. Interactive input fallbacks for maximum ease of use
    artifact_path = args.artifact_path
    if not artifact_path:
        default_path = "brandon-baek/bori-sft/bori-2-135m-sft"
        user_input = input(f"💬 Enter W&B artifact path [Default: {default_path}]: ").strip()
        artifact_path = user_input if user_input else default_path

    alias = args.alias
    if not args.artifact_path: # only prompt if we are in interactive mode
        user_input = input(f"💬 Enter version alias [Default: {alias}]: ").strip()
        alias = user_input if user_input else alias

    tokenizer_path = args.tokenizer_path
    if not tokenizer_path:
        discovered = discover_tokenizer()
        if discovered:
            print(f"🔍 Auto-detected merged tokenizer at: {discovered}")
            tokenizer_path = discovered
        else:
            tokenizer_path = input("💬 Path to merged tokenizer directory: ").strip()

    # 2. Download from W&B
    print(f"\n🚀 Initializing W&B API...")
    try:
        import wandb
    except ImportError:
        print("❌ Error: 'wandb' package is not installed. Please run: pip install wandb")
        return

    api = wandb.Api()
    full_path = f"{artifact_path}:{alias}"
    print(f"📦 Fetching W&B Artifact: {full_path} ...")
    
    try:
        artifact = api.artifact(full_path)
        print(f"📥 Selective Downloading to: {args.output_dir} ...")
        
        # Selectively download only what is needed for inference (skips 1.1GB optimizer states!)
        skipped_count = 0
        for file in artifact.files():
            if file.name.endswith((".pt", ".pth", "trainer_state.json", "training_args.bin")):
                skipped_count += 1
                continue
            print(f"   Downloading {file.name} ({file.size / 1024 / 1024:.1f} MB)...")
            file.download(root=args.output_dir)
            
        if skipped_count > 0:
            print(f"   ⚡ Skipped {skipped_count} optimizer/trainer state files to save ~1.1 GB of bandwidth!")
        print("✅ Checkpoint successfully downloaded!")
    except Exception as e:
        print(f"❌ Failed to download artifact: {e}")
        print("💡 Tip: Double check your W&B credentials and path. Fallback: Loading local model...")
        if not os.path.exists(args.output_dir):
            return

    # 3. Load model and tokenizer
    print("\n🔮 Loading Bori model into memory...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            local_files_only=True
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model or tokenizer: {e}")
        return

    # 4. Interactive chat loop
    print("\n" + "="*70)
    print("💬 Bori SFT Interactive Chat Session Initialized!")
    print("   Type your prompt below. Type 'exit' or 'quit' to end.")
    print("="*70 + "\n")

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{{ '<|im_start|>assistant\n' }}"
    )

    history = []
    
    while True:
        try:
            user_msg = input("\n👤 User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        history.append({"role": "user", "content": user_msg})
        
        # Apply chat template
        # We manually render the chat template using jinja2 pattern
        prompt = ""
        for msg in history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("🤖 Bori: ", end="", flush=True)
        
        # Stream response token by token
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Strip trailing tags if present
        completion = completion.replace("<|im_end|>", "").strip()
        
        # Print response
        print(completion)
        history.append({"role": "assistant", "content": completion})

if __name__ == "__main__":
    main()
