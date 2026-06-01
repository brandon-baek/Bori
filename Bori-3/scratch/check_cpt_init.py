import os
from transformers import AutoTokenizer

def main():
    merged_tok_path = "./downloaded_sft_checkpoint"
    base_model = "HuggingFaceTB/SmolLM2-135M"

    print("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model)
    orig_vocab_size = len(base_tokenizer)

    print("Loading merged tokenizer...")
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_tok_path)
    new_vocab_size = len(merged_tokenizer)

    num_new_tokens = new_vocab_size - orig_vocab_size
    print(f"\nOriginal vocab size: {orig_vocab_size}")
    print(f"Merged vocab size:   {new_vocab_size}")
    print(f"New tokens added:    {num_new_tokens}")

    if num_new_tokens <= 0:
        print("❌ No new tokens detected!")
        return

    subword_init_count = 0
    fallback_count = 0
    sample_eeve_tokens = []
    sample_fallback_tokens = []

    for i in range(num_new_tokens):
        new_token_id = orig_vocab_size + i
        new_token_str = merged_tokenizer.convert_ids_to_tokens(new_token_id)
        
        if new_token_str is None:
            fallback_count += 1
            continue
            
        try:
            decoded_str = merged_tokenizer.decode([new_token_id])
            subword_ids = base_tokenizer.encode(decoded_str, add_special_tokens=False)
        except Exception:
            subword_ids = []
            
        if len(subword_ids) > 0:
            subword_init_count += 1
            if len(sample_eeve_tokens) < 10:
                sample_eeve_tokens.append((new_token_str, decoded_str, [base_tokenizer.decode([sid]) for sid in subword_ids]))
        else:
            fallback_count += 1
            if len(sample_fallback_tokens) < 10:
                sample_fallback_tokens.append(new_token_str)

    print("\n" + "="*50)
    print("📊 EEVE INITIALIZATION DIAGNOSTICS 📊")
    print("="*50)
    print(f"Subword-based (EEVE) tokens:  {subword_init_count} ({subword_init_count/num_new_tokens*100:.2f}%)")
    print(f"Fallback (global mean) tokens: {fallback_count} ({fallback_count/num_new_tokens*100:.2f}%)")
    
    print("\n💡 Sample EEVE Initialized Tokens (New Token -> Decoded -> Constituent English Subwords):")
    for t_str, dec, subwords in sample_eeve_tokens:
        print(f"   - {repr(t_str)} -> {repr(dec)} -> {subwords}")
        
    print("\n💡 Sample Fallback (Global Mean) Tokens:")
    print(f"   - {sample_fallback_tokens}")
    print("="*50)

if __name__ == "__main__":
    main()
