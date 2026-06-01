import torch
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedModel, PreTrainedTokenizerBase

console = Console()

def expand_model_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    original_tokenizer: PreTrainedTokenizerBase,
    num_new_tokens: int,
) -> PreTrainedModel:
    original_vocab_size = len(tokenizer) - num_new_tokens

    console.log(
        f"[bold]Resizing embeddings: {original_vocab_size:,} → {len(tokenizer):,} "
        f"(+{num_new_tokens:,} tokens)[/bold]"
    )

    model.resize_token_embeddings(len(tokenizer))

    subword_initialized = 0
    fallback_initialized = 0

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        original_input_mean = input_embeddings.weight[:original_vocab_size].mean(dim=0)
        if output_embeddings is not None:
            original_output_mean = output_embeddings.weight[:original_vocab_size].mean(dim=0)

        for idx in range(original_vocab_size, len(tokenizer)):
            token_str = tokenizer.convert_ids_to_tokens(idx)
            sub_ids = original_tokenizer.encode(token_str, add_special_tokens=False)

            if sub_ids and len(sub_ids) > 0:
                valid_ids = [sid for sid in sub_ids if sid < original_vocab_size]

                if valid_ids:
                    sub_embeds = input_embeddings.weight[valid_ids]
                    input_embeddings.weight[idx] = sub_embeds.mean(dim=0)

                    if output_embeddings is not None:
                        out_sub_embeds = output_embeddings.weight[valid_ids]
                        output_embeddings.weight[idx] = out_sub_embeds.mean(dim=0)

                    subword_initialized += 1
                    continue

            input_embeddings.weight[idx] = original_input_mean
            if output_embeddings is not None:
                output_embeddings.weight[idx] = original_output_mean

            fallback_initialized += 1

    console.log(f"[green]Subword-initialized: {subword_initialized:,} tokens[/green]")
    console.log(f"[yellow]Fallback (global mean): {fallback_initialized:,} tokens[/yellow]")

    return model


def compute_fertility(tokenizer: PreTrainedTokenizerBase, texts: list[str]) -> dict:
    total_tokens = 0
    total_chars = 0
    total_words = 0

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)
        total_chars += len(text)
        total_words += len(text.split())

    return {
        "avg_tokens_per_char": total_tokens / total_chars if total_chars > 0 else 0.0,
        "avg_tokens_per_word": total_tokens / total_words if total_words > 0 else 0.0,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


def compare_tokenizers(
    base_tokenizer: PreTrainedTokenizerBase,
    expanded_tokenizer: PreTrainedTokenizerBase,
    sample_texts: list[str],
) -> dict:
    table = Table(
        title="Tokenizer Comparison: Base vs Expanded",
        show_lines=True,
    )
    table.add_column("Text", style="white", max_width=40, overflow="ellipsis")
    table.add_column("Base Tokens", style="red", justify="right")
    table.add_column("Expanded Tokens", style="green", justify="right")
    table.add_column("Improvement", style="cyan", justify="right")

    ratios = []

    for text in sample_texts:
        base_ids = base_tokenizer.encode(text, add_special_tokens=False)
        expanded_ids = expanded_tokenizer.encode(text, add_special_tokens=False)

        base_count = len(base_ids)
        expanded_count = len(expanded_ids)

        ratio = base_count / expanded_count if expanded_count > 0 else float("inf")
        ratios.append(ratio)

        table.add_row(
            text,
            str(base_count),
            str(expanded_count),
            f"{ratio:.2f}x",
        )

    console.print(table)

    base_fertility = compute_fertility(base_tokenizer, sample_texts)
    expanded_fertility = compute_fertility(expanded_tokenizer, sample_texts)

    avg_improvement = sum(ratios) / len(ratios) if ratios else 0.0

    console.print()
    summary_table = Table(title="Fertility Summary", show_lines=True)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Base", style="red", justify="right")
    summary_table.add_column("Expanded", style="green", justify="right")

    summary_table.add_row(
        "Avg tokens/char",
        f"{base_fertility['avg_tokens_per_char']:.3f}",
        f"{expanded_fertility['avg_tokens_per_char']:.3f}",
    )
    summary_table.add_row(
        "Avg tokens/word",
        f"{base_fertility['avg_tokens_per_word']:.3f}",
        f"{expanded_fertility['avg_tokens_per_word']:.3f}",
    )
    summary_table.add_row(
        "Total tokens",
        str(base_fertility["total_tokens"]),
        str(expanded_fertility["total_tokens"]),
    )
    summary_table.add_row(
        "Avg improvement ratio",
        "—",
        f"{avg_improvement:.2f}x",
    )

    console.print(summary_table)

    return {
        "base_fertility": base_fertility,
        "expanded_fertility": expanded_fertility,
        "avg_improvement_ratio": avg_improvement,
    }
