import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    import os
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    is_local = os.path.isdir(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=is_local,
    )

    prompts = [
        "안녕하세요, 오늘 날씨가 어떤가요?",
        "Explain how continuous pre-training works in simple terms.",
        "Konglish is a mixture of Korean and English. For example, 'I am going to have a meeting in the morning' can be said in Konglish as '오늘 아침에 미팅이 있어요.'"
    ]

    for p in prompts:
        print(f"\nPrompt: {p}")
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {completion}")

if __name__ == "__main__":
    main()
