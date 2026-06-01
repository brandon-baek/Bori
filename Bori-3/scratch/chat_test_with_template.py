import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    prompts = [
        "안녕하세요, 오늘 날씨가 어떤가요?",
        "Explain how continuous pre-training works in simple terms.",
        "Konglish is a mixture of Korean and English. For example, 'I am going to have a meeting in the morning' can be said in Konglish as '오늘 아침에 미팅이 있어요.'"
    ]

    for p in prompts:
        print(f"\n=====================================")
        print(f"Raw Prompt: {p}")
        
        # Apply chat template
        messages = [
            {"role": "user", "content": p}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Formatted Prompt:\n{formatted_prompt}")
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate with SFT template settings (eos_token should be <|im_end|>)
        eos_token_id = tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_token_id
            )
            
        # Decode skipping the prompt
        input_len = inputs["input_ids"].shape[1]
        completion_ids = outputs[0][input_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"Response:\n{completion}")

if __name__ == "__main__":
    main()
