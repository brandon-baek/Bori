import os
from transformers import AutoTokenizer

def main():
    tokenizer_path = "./downloaded_sft_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "안녕하세요! 오늘 기분이 어떠신가요?"},
        {"role": "assistant", "content": "안녕하세요! 저는 기분이 아주 좋습니다. 감사합니다!"}
    ]

    # Full text formatted using apply_chat_template
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    print(f"Full formatted text:\n{repr(full_text)}\n")
    print(f"Full tokens length: {len(full_tokens)}")

    # Incremental tokenization and masking
    input_ids = []
    labels = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            t_text = f"<|im_start|>system\n{content}<|im_end|>\n"
            tokens = tokenizer.encode(t_text, add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))
        elif role == "user":
            t_text = f"<|im_start|>user\n{content}<|im_end|>\n"
            tokens = tokenizer.encode(t_text, add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))
        elif role == "assistant":
            prefix = "<|im_start|>assistant\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_ids.extend(prefix_tokens)
            labels.extend([-100] * len(prefix_tokens))
            
            response = f"{content}<|im_end|>\n"
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            input_ids.extend(response_tokens)
            labels.extend(response_tokens)

    print(f"Incremental tokens length: {len(input_ids)}")
    print(f"Match?: {input_ids == full_tokens}")

    if input_ids != full_tokens:
        print("\nMismatch details:")
        print(f"Full:        {full_tokens}")
        print(f"Incremental: {input_ids}")
    else:
        print("\nTokens match perfectly! Checking labels:")
        for idx, (tok_id, label) in enumerate(zip(input_ids, labels)):
            tok_str = tokenizer.decode([tok_id])
            print(f"Token {idx:2d}: {repr(tok_str):15s} -> ID: {tok_id:5d} -> Label: {label:5d}")

if __name__ == "__main__":
    main()
