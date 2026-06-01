import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

console = Console()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--korean_tokenizer_path", type=str, default="../data/korean_tokenizer")
    parser.add_argument("--output_dir", type=str, default="../data/merged_tokenizer")
    return parser.parse_args()

def fertility_comparison(base_tokenizer, merged_tokenizer, sentences: list[str]):
    table = Table(
        title="Fertility Comparison: Base vs Merged Tokenizer",
        show_lines=True,
    )
    table.add_column("Sentence", style="white", max_width=40, overflow="ellipsis")
    table.add_column("Base Tokens", style="red", justify="right")
    table.add_column("Merged Tokens", style="green", justify="right")
    table.add_column("Improvement", style="cyan", justify="right")

    for sentence in sentences:
        base_ids = base_tokenizer.encode(sentence, add_special_tokens=False)
        merged_ids = merged_tokenizer.encode(sentence, add_special_tokens=False)

        base_count = len(base_ids)
        merged_count = len(merged_ids)

        improvement = base_count / merged_count if merged_count > 0 else float("inf")
        table.add_row(sentence, str(base_count), str(merged_count), f"{improvement:.2f}x")

    console.print(table)

def main():
    args = parse_args()

    console.rule("[bold blue]Tokenizer Merging (BPE Vocab & Merge Rules)[/bold blue]")
    console.print(f"  Base model:        {args.base_model}")
    console.print(f"  Korean tokenizer:  {args.korean_tokenizer_path}")
    console.print(f"  Output dir:        {args.output_dir}")
    console.print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.log("[bold]Loading base tokenizer...[/bold]")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Save the base tokenizer temporarily to output directory to get its raw config files
    console.log("[bold]Saving base files to output directory...[/bold]")
    base_tokenizer.save_pretrained(str(output_path))
    
    # Path to the base tokenizer.json
    base_json_path = output_path / "tokenizer.json"
    if not base_json_path.exists():
        raise FileNotFoundError(f"Could not find tokenizer.json in base save path: {base_json_path}")

    # Load base JSON
    with open(base_json_path, "r", encoding="utf-8") as f:
        base_json = json.load(f)

    # Path to the Korean tokenizer.json
    kor_json_path = Path(args.korean_tokenizer_path) / "tokenizer.json"
    if not kor_json_path.exists():
        raise FileNotFoundError(f"Could not find tokenizer.json in Korean tokenizer path: {kor_json_path}")

    # Load Korean JSON
    with open(kor_json_path, "r", encoding="utf-8") as f:
        kor_json = json.load(f)

    console.log("[bold]Merging vocabulary and merge rules...[/bold]")
    base_model_sec = base_json["model"]
    kor_model_sec = kor_json["model"]

    base_vocab = base_model_sec["vocab"]
    kor_vocab = kor_model_sec["vocab"]

    base_merges = base_model_sec.get("merges", [])
    kor_merges = kor_model_sec.get("merges", [])

    original_vocab_size = len(base_vocab)

    # Identify new Korean tokens (excluding standard special tokens)
    special_tokens_set = {"<|endoftext|>", "<s>", "</s>", "<unk>", "<pad>"}
    new_tokens = [tok for tok in kor_vocab.keys() if tok not in base_vocab and tok not in special_tokens_set]
    new_tokens.sort()

    # Add new tokens to base vocab with sequential IDs
    next_id = max(base_vocab.values()) + 1
    num_added = 0
    for token in new_tokens:
        base_vocab[token] = next_id
        next_id += 1
        num_added += 1

    # Merge BPE rules
    added_merges_count = 0
    for merge in kor_merges:
        if merge not in base_merges:
            base_merges.append(merge)
            added_merges_count += 1

    # Save the updated tokenizer.json (fast tokenizer backend)
    with open(base_json_path, "w", encoding="utf-8") as f:
        json.dump(base_json, f, ensure_ascii=False, indent=2)

    # Also update vocab.json and merges.txt (slow tokenizer backend) for full compatibility
    vocab_json_path = output_path / "vocab.json"
    merges_txt_path = output_path / "merges.txt"

    if vocab_json_path.exists():
        console.log("[bold]Updating vocab.json (slow tokenizer)...[/bold]")
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            slow_vocab = json.load(f)
        for token in new_tokens:
            slow_vocab[token] = base_vocab[token]
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(slow_vocab, f, ensure_ascii=False)

    if merges_txt_path.exists():
        console.log("[bold]Updating merges.txt (slow tokenizer)...[/bold]")
        with open(merges_txt_path, "r", encoding="utf-8") as f:
            existing_merges_text = f.read()
        with open(merges_txt_path, "a", encoding="utf-8") as f:
            for merge in kor_merges:
                if merge not in existing_merges_text:
                    f.write(merge + "\n")

    console.log(f"[green]Successfully merged {num_added:,} tokens and {added_merges_count:,} BPE merge rules![/green]")

    # Reload the merged tokenizer from the output path
    console.log("[bold]Reloading merged tokenizer...[/bold]")
    merged_tokenizer = AutoTokenizer.from_pretrained(str(output_path))
    final_vocab_size = len(merged_tokenizer)

    console.print()
    stats_table = Table(title="Vocabulary & Merge Statistics", show_lines=True)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan", justify="right")
    stats_table.add_row("Original vocab size", f"{original_vocab_size:,}")
    stats_table.add_row("New tokens added", f"{num_added:,}")
    stats_table.add_row("BPE merges added", f"{added_merges_count:,}")
    stats_table.add_row("Final vocab size", f"{final_vocab_size:,}")
    console.print(stats_table)
    console.print()

    console.log("[bold]Running fertility comparison...[/bold]")
    test_sentences = [
        "안녕하세요, 오늘 날씨가 어떤가요?",
        "인공지능 기술은 빠르게 발전하고 있습니다.",
        "대한민국의 수도는 서울입니다.",
        "오늘 저녁에 뭐 먹을까요?",
        "프로그래밍을 배우는 것은 재미있습니다.",
    ]

    # Use original base tokenizer for comparison
    original_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    fertility_comparison(original_tokenizer, merged_tokenizer, test_sentences)
    console.print()

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Merged tokenizer saved to {output_path.resolve()}")

if __name__ == "__main__":
    main()
