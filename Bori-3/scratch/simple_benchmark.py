import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_test(model, tokenizer, prompt, use_template=True):
    if use_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt
        
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    eos_token_id = tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_id if use_template else tokenizer.eos_token_id
        )
    
    input_len = inputs["input_ids"].shape[1]
    completion_ids = outputs[0][input_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion.strip()

def main():
    model_path = "./downloaded_sft_checkpoint"
    tokenizer_path = "./downloaded_sft_checkpoint"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
    )

    tests = [
        ("Greeting (Korean)", "안녕하세요! 만나서 반가워요."),
        ("Factual (Korean)", "대한민국의 수도는 어디인가요?"),
        ("Translation (EN->KO)", "Translate 'Where is the nearest subway station?' to Korean."),
        ("Konglish (Bilingual)", "Explain what '더치페이' (Dutch pay) means in Konglish, and how people use it."),
        ("Simple Logic (English)", "A farmer has 5 apples. He eats 2. How many apples does he have left? Explain in one sentence.")
    ]

    print("\n" + "="*80)
    print("🌾 BORI-2 SFT CHECKPOINT BENCHMARK RESULTS 🌾")
    print("="*80)

    for name, prompt in tests:
        print(f"\n📌 TEST: {name}")
        print(f"   Prompt: \"{prompt}\"")
        
        # 1. Raw Autocomplete Output
        raw_res = run_test(model, tokenizer, prompt, use_template=False)
        # 2. Template Chat Output
        chat_res = run_test(model, tokenizer, prompt, use_template=True)
        
        print(f"   [Raw Autocomplete Response]:\n   {repr(raw_res)}")
        print(f"   [SFT Chat Template Response]:\n   {chat_res}")
        print("-"*80)

if __name__ == "__main__":
    main()
