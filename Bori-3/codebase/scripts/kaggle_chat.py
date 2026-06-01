import os
import sys

def run_bori_chat_in_kaggle(model_path, tokenizer_path):
    """
    Self-contained function to run Bori Interactive Chat & Autocomplete
    directly inside a Kaggle Notebook cell under GPU acceleration.
    """
    try:
        import torch
    except ImportError:
        print("📥 Installing torch...")
        os.system("pip install torch -q")
        import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    except ImportError:
        print("📥 Installing transformers...")
        os.system("pip install transformers -q")
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    from threading import Thread

    # 1. Load Model & Tokenizer
    print("\n🔮 Loading tokenizer and model into GPU memory...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Determine best available device (automatically maps to Kaggle GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"   Using device: {device.upper()} ({dtype})")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )
        print("✅ Bori successfully loaded!")
    except Exception as e:
        print(f"❌ Failed to load model or tokenizer: {e}")
        return

    # 2. Stateful unified interactive session
    current_mode = "1" # "1" = Chat, "2" = Autocomplete
    history = []
    
    print("\n" + "="*70)
    print("🌾 Bori (보리): Stateful Kaggle Interactive Session (GPU Accelerated)!")
    print("   Type '/mode' to switch between SFT Chat and Autocomplete.")
    print("   Type '/clear' to wipe chat history, or 'exit'/'quit' to end.")
    print("="*70)
    
    while True:
        if current_mode == "1":
            prompt_header = "\n💬 [Chat] User: "
        else:
            prompt_header = "\n📝 [Autocomplete] Prompt Prefix: "
            
        try:
            user_msg = input(prompt_header).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_msg:
            continue
            
        # Parse commands
        if user_msg.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("👋 Goodbye!")
            break
            
        if user_msg.lower() == "/help":
            print("\n📋 Bori Kaggle CLI Help Commands:")
            print("  /mode   - Swaps instantly between SFT Chat and Raw Autocomplete")
            print("  /clear  - Clears chat dialogue history")
            print("  /exit   - Terminate session and exit")
            continue
            
        if user_msg.lower() == "/clear":
            history = []
            print("🧹 Chat history cleared!")
            continue

        if user_msg.lower() == "/mode":
            if current_mode == "1":
                current_mode = "2"
                print("\n🔄 Switched to Raw Autocomplete Mode! (No SFT templates)")
            else:
                current_mode = "1"
                print("\n🔄 Switched to SFT Interactive Chat Mode! (Dialog template active)")
            continue

        # ----------------------------------------------------
        # Mode 1: SFT Chat Generation (Bilingual Chatbot)
        # ----------------------------------------------------
        if current_mode == "1":
            history.append({"role": "user", "content": user_msg})
            
            prompt = ""
            for msg in history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            print("🤖 Bori: ", end="", flush=True)
            
            # Live Token-by-Token Streaming inside Kaggle cell output
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
                print(new_text, end="", flush=True)
                completion += new_text
            print()
            
            completion = completion.replace("<|im_end|>", "").strip()
            history.append({"role": "assistant", "content": completion})

        # ----------------------------------------------------
        # Mode 2: Raw Autocomplete Generation (Text Completion)
        # ----------------------------------------------------
        else:
            inputs = tokenizer(user_msg, return_tensors="pt").to(model.device)
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
            print()

# ----------------------------------------------------
# 💡 How to execute in Kaggle:
# ----------------------------------------------------
# Replace these with your actual local/Kaggle dataset folder paths
# e.g., model_path = "./sft_checkpoints/checkpoint-400"
#       tokenizer_path = "./test_output/merged_tokenizer"
#
# run_bori_chat_in_kaggle(model_path, tokenizer_path)
