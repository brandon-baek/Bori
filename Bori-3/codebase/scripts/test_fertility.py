#!/usr/bin/env python3
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Setup paths so we can import from src/
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.tokenizer_utils import compare_tokenizers

def find_tokenizer_path(user_path: str) -> Path:
    # Try multiple logical search paths
    search_paths = [
        Path(user_path),
        project_root / user_path,
        project_root.parent / user_path,
        Path("/kaggle/working") / user_path,
        project_root / "data/merged_tokenizer",
    ]
    for p in search_paths:
        if p.exists() and (p / "tokenizer.json").exists():
            return p
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick Tokenizer Fertility & Merge Rules Verification")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--merged_path", type=str, default="data/merged_tokenizer")
    args = parser.parse_args()

    print("=" * 60)
    print("      TOKENIZER FERTILITY & MERGE RULES VERIFICATION")
    print("=" * 60)

    # 1. Locate the merged tokenizer
    merged_dir = find_tokenizer_path(args.merged_path)
    if not merged_dir:
        print(f"❌ Error: Could not find a valid merged tokenizer at '{args.merged_path}' or fallback paths.")
        print("Please ensure you have built the tokenizer first or placed it in 'data/merged_tokenizer/'.")
        sys.exit(1)

    print(f"Base Model:       {args.base_model}")
    print(f"Merged Tokenizer: {merged_dir.resolve()}")
    print("-" * 60)

    # 2. Load tokenizers
    try:
        base_tok = AutoTokenizer.from_pretrained(args.base_model)
        merged_tok = AutoTokenizer.from_pretrained(str(merged_dir))
    except Exception as e:
        print(f"❌ Error loading tokenizers: {e}")
        sys.exit(1)

    # 3. Check vocabulary stats
    base_vocab_size = len(base_tok)
    merged_vocab_size = len(merged_tok)
    added_tokens_count = merged_vocab_size - base_vocab_size
    print(f"Base Vocab Size:   {base_vocab_size:,}")
    print(f"Merged Vocab Size: {merged_vocab_size:,} (+{added_tokens_count:,} tokens)")
    print("-" * 60)

    # 4. Perform a critical check to verify if the fixed merge rules are active
    # We use a standard Korean word "안녕하세요" which is highly tokenized by byte-level fallback
    test_word = "안녕하세요"
    base_tokens = base_tok.tokenize(test_word)
    merged_tokens = merged_tok.tokenize(test_word)

    base_len = len(base_tokens)
    merged_len = len(merged_tokens)

    print(f"Tokenization Check for '{test_word}':")
    print(f"  - Base Tokenizer:   {base_len} tokens -> {base_tokens}")
    print(f"  - Merged Tokenizer: {merged_len} tokens -> {merged_tokens}")
    print()

    if base_len == 14 and merged_len == 14:
        print("🚨 [bold red]CRITICAL WARNING: BROKEN MERGE DETECTED![/bold red]")
        print("   The merged tokenizer is producing 14 tokens, which is EXACTLY identical")
        print("   to the base tokenizer. The BPE merge rules were NOT injected!")
        print("   Action required: Delete the 'data/merged_tokenizer' directory and")
        print("   re-run the tokenizer training & merge step.")
        print("\n❌ Halting execution to save your GPU quota!")
        sys.exit(1)
    elif merged_len <= 7:
        print("🎉 [bold green]VERIFICATION SUCCESS: NEW BPE MERGE RULES ARE ACTIVE![/bold green]")
        print(f"   The word '{test_word}' is compressed down to {merged_len} tokens!")
        print("   The Korean subwords are merging properly. This tokenizer is production-ready.")
    else:
        print("⚠️ [bold yellow]PARTIAL MERGE WARNING:[/bold yellow]")
        print(f"   The word is tokenized to {merged_len} tokens. Check BPE rule integrity.")
    print("-" * 60)

    # 5. Run full fertility comparison table
    sample_texts = [
        "안녕하세요, 오늘 날씨가 어떤가요?",
        "인공지능 기술은 빠르게 발전하고 있습니다.",
        "대한민국의 수도는 서울입니다.",
        "오늘 저녁에 뭐 먹을까요?",
        "프로그래밍을 배우는 것은 재미있습니다.",
        "SmolLM2 model is great, but adding Korean vocabulary makes it even better!",
        "오늘 아침에 미팅이 있어서 일찍 일어났어요."
    ]

    print("Running detailed side-by-side fertility comparison...")
    compare_tokenizers(base_tok, merged_tok, sample_texts)
    print("=" * 60)

if __name__ == "__main__":
    main()
