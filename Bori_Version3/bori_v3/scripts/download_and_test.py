import argparse
import os
import sys
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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

def setup_wandb_api_key():
    """Loads a locally saved W&B API key from .env, or prompts and saves it."""
    # Find .env at project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bori_root = os.path.dirname(os.path.dirname(script_dir)) # /Users/brandon.baek/Development/Bori
    env_path = os.path.join(bori_root, ".env")
    
    # 1. Try to load from existing .env
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["WANDB_API_KEY"] = key
                    print("🔑 Loaded W&B API Key from local .env config.")
                    return True

    # 2. If not authenticated already, prompt and save
    try:
        import wandb
    except ImportError:
        print("❌ Error: 'wandb' package is not installed. Please run: pip install wandb")
        sys.exit(1)

    # Simple check if already logged in via ~/.netrc
    try:
        # If logged in, this won't throw an error
        api = wandb.Api()
        # Try a dummy query to verify login
        api.viewer
        return True
    except Exception:
        # Not logged in
        pass

    if not os.environ.get("WANDB_API_KEY"):
        print("\n🔑 W&B Authentication Required.")
        user_key = input("👉 Please paste your W&B API Key (it will be saved locally in .env): ").strip()
        if user_key:
            os.environ["WANDB_API_KEY"] = user_key
            # Save to .env (git-ignored)
            with open(env_path, "w") as f:
                f.write(f"WANDB_API_KEY={user_key}\n")
            print(f"💾 Saved API key to {env_path} (Git-ignored) for future runs.")
            return True
        else:
            print("⚠️ Warning: No W&B API Key entered. Proceeding without explicit credentials...")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Unified W&B Downloader & Streaming CLI for Bori")
    parser.add_argument("--artifact_path", type=str, default=None, help="W&B artifact path 'entity/project/artifact_name'")
    parser.add_argument("--alias", type=str, default="v1", help="Artifact version alias ('v1', 'latest', 'step-400')")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to merged tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./downloaded_sft_checkpoint", help="Destination download directory")
    args = parser.parse_args()

    print("======================================================================")
    print("🌾 Bori (보리): Unified Downloader & Live Autocomplete/Chat CLI")
    print("======================================================================\n")

    # W&B API Key setup
    setup_wandb_api_key()

    # 1. Interactive input fallbacks for maximum ease of use
    artifact_path = args.artifact_path
    if not artifact_path:
        default_path = "brandon_baek/Bori-V2/bori-2-135m-sft"
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
            print(f"🔍 Auto-detected local merged tokenizer at: {discovered}")
            tokenizer_path = discovered
        else:
            # SFT checkpoint already contains tokenizer.json and tokenizer_config.json!
            print(f"🔍 No standalone tokenizer directory found. Using the tokenizer bundled in the downloaded checkpoint.")
            tokenizer_path = args.output_dir

    # 2. Download from W&B
    import wandb
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

    # 4. Mode Selection Menu
    print("\n" + "="*50)
    print("💡 Select Generation Mode:")
    print("  [1] Interactive Chat Mode (Dialogue with SFT templates)")
    print("  [2] Raw Autocomplete Mode (Predict next-tokens directly)")
    print("="*50)
    
    mode = "1"
    mode_input = input("👉 Select mode [Default: 1]: ").strip()
    if mode_input in ["1", "2"]:
        mode = mode_input

    if mode == "1":
        run_chat_mode(model, tokenizer)
    else:
        run_autocomplete_mode(model, tokenizer)

def run_chat_mode(model, tokenizer):
    print("\n" + "="*70)
    print("💬 Bori SFT Interactive Chat Session Initialized (Live Streaming)!")
    print("   Type your prompt below. Type 'exit' or 'quit' to end.")
    print("="*70 + "\n")

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
        prompt = ""
        for msg in history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("🤖 Bori: ", end="", flush=True)
        
        # Live streaming setup using TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
            streamer=streamer
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        completion = ""
        for new_text in streamer:
            # Render tokens immediately
            print(new_text, end="", flush=True)
            completion += new_text
            
        print() # Newline at the end
        
        # Strip trailing tags if present
        completion = completion.replace("<|im_end|>", "").strip()
        history.append({"role": "assistant", "content": completion})

def run_autocomplete_mode(model, tokenizer):
    print("\n" + "="*70)
    print("📝 Bori Raw Autocomplete Session Initialized (Live Streaming)!")
    print("   Type any text prefix and Bori will complete it.")
    print("   Type 'exit' or 'quit' to end.")
    print("="*70 + "\n")
    
    while True:
        try:
            prefix = input("\n📝 Prompt Prefix: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not prefix:
            continue
        if prefix.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        
        print("✍️ Completion: ", end="", flush=True)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            print(new_text, end="", flush=True)
            
        print() # Newline at the end

if __name__ == "__main__":
    main()
