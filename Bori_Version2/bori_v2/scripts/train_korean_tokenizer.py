import argparse
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

console = Console()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Korean BPE tokenizer on FineWeb2 Korean data."
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="../data/korean_tokenizer")
    parser.add_argument("--num_samples", type=int, default=500000)
    parser.add_argument("--text_column", type=str, default="text")
    return parser.parse_args()

def batch_iterator(dataset, text_column: str, num_samples: int, batch_size: int = 1000):
    batch = []
    count = 0

    for example in dataset:
        text = example.get(text_column, "")
        if not text:
            continue

        batch.append(text)
        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []
            if count % 50000 == 0:
                console.log(f"[cyan]Processed {count:,} / {num_samples:,} documents...[/cyan]")

        if count >= num_samples:
            break

    if batch:
        yield batch

    console.log(f"[green]Finished streaming {count:,} documents.[/green]")

def main():
    args = parse_args()

    console.rule("[bold blue]Korean BPE Tokenizer Training[/bold blue]")
    console.print(f"  Vocab size:    [yellow]{args.vocab_size:,}[/yellow]")
    console.print(f"  Num samples:   [yellow]{args.num_samples:,}[/yellow]")
    console.print(f"  Output dir:    {args.output_dir}")
    console.print(f"  Text column:   {args.text_column}")
    console.print()

    console.log("[bold]Loading FineWeb2 Korean data (streaming)...[/bold]")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="kor_Hang",
        split="train",
        streaming=True,
    )

    console.log("[bold]Initializing ByteLevel BPE tokenizer...[/bold]")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<|endoftext|>", "<s>", "</s>", "<unk>", "<pad>"]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    console.log("[bold]Training tokenizer...[/bold]")
    tokenizer.train_from_iterator(
        batch_iterator(dataset, args.text_column, args.num_samples),
        trainer=trainer,
    )

    console.log(f"[green]Training complete! Vocab size: {tokenizer.get_vocab_size():,}[/green]")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    wrapped_tokenizer.save_pretrained(str(output_path))

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Tokenizer saved to {output_path.resolve()}")
    console.print(f"Final vocab size: {len(wrapped_tokenizer):,}")

if __name__ == "__main__":
    main()
