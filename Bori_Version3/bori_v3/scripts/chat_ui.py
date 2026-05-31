import gradio as gr
import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    def respond(message, history):
        formatted_history = []
        for h in history:
            formatted_history.append({"role": "user", "content": h[0]})
            formatted_history.append({"role": "assistant", "content": h[1]})
        formatted_history.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(formatted_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

    demo = gr.ChatInterface(
        respond,
        title="Bori V2 — Korean/English Chatbot",
        description="A SmolLM-135M model adapted for Korean via Continuous Pre-training and vocabulary expansion.",
        examples=[
            "안녕하세요! 반가워요.",
            "Explain how space exploration helps humanity.",
            "Konglish is cool! 오늘 저녁에 스케줄이 어떻게 되세요?"
        ]
    )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
