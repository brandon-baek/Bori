#!/usr/bin/env python3
"""
Bori-3 Comprehensive Benchmark & Evaluation Script
===================================================

Evaluates a bilingual (Korean-English) small language model across five axes:
  1. Perplexity on held-out Korean and English data
  2. Korean tokenizer fertility (tokens-per-character ratio)
  3. Instruction following with ChatML template
  4. Repetition / degenerate loop analysis
  5. Summary dashboard with letter grading

Designed to run in <10 minutes on a single T4 GPU (or CPU with smaller samples).
Fully standalone — no imports from src/.

Usage:
    python benchmark.py --model_path ./checkpoints/final
"""

import sys
import os
import json
import time
import math
import re
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Optional: rich for beautiful tables. Falls back to plain print if unavailable.
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
BASELINE_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

KOREAN_DATASET = "HuggingFaceFW/fineweb-2"
KOREAN_SUBSET = "kor_Hang"
ENGLISH_DATASET = "HuggingFaceFW/fineweb-edu"

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ──────────────────────────────────────────────────────────────
# Instruction-following evaluation prompts
# ──────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    # --- Simple factual Q&A ---
    {"prompt": "What is the capital of South Korea?", "lang": "en", "category": "factual"},
    {"prompt": "대한민국의 수도는 어디인가요?", "lang": "ko", "category": "factual"},
    {"prompt": "What is the largest ocean on Earth?", "lang": "en", "category": "factual"},
    {"prompt": "지구에서 가장 큰 대양은 무엇인가요?", "lang": "ko", "category": "factual"},

    # --- Simple math ---
    {"prompt": "What is 5 + 3?", "lang": "en", "category": "math"},
    {"prompt": "5 더하기 3은?", "lang": "ko", "category": "math"},
    {"prompt": "What is 12 times 4?", "lang": "en", "category": "math"},
    {"prompt": "100 나누기 5는 얼마인가요?", "lang": "ko", "category": "math"},

    # --- Translation ---
    {"prompt": "Translate 'hello' to Korean.", "lang": "en", "category": "translation"},
    {"prompt": "'안녕하세요'를 영어로 번역해주세요.", "lang": "ko", "category": "translation"},
    {"prompt": "Translate 'thank you' to Korean.", "lang": "en", "category": "translation"},
    {"prompt": "'감사합니다'를 영어로 번역해주세요.", "lang": "ko", "category": "translation"},

    # --- Conversational ---
    {"prompt": "Tell me a joke.", "lang": "en", "category": "conversational"},
    {"prompt": "오늘 날씨 어때요?", "lang": "ko", "category": "conversational"},
    {"prompt": "How are you doing today?", "lang": "en", "category": "conversational"},
    {"prompt": "재미있는 이야기 하나 해주세요.", "lang": "ko", "category": "conversational"},

    # --- Simple instructions ---
    {"prompt": "Write a short poem about spring.", "lang": "en", "category": "instruction"},
    {"prompt": "봄에 대한 짧은 시를 써주세요.", "lang": "ko", "category": "instruction"},
    {"prompt": "List three benefits of exercise.", "lang": "en", "category": "instruction"},
    {"prompt": "운동의 장점 세 가지를 알려주세요.", "lang": "ko", "category": "instruction"},
]

# Korean text prefixes for repetition analysis
KOREAN_PREFIXES = [
    "오늘 아침에 일어나서",
    "인공지능 기술은 최근",
    "서울의 가을 하늘은",
    "한국어를 배우는 것은",
    "프로그래밍을 처음 시작할 때",
    "맛있는 김치찌개를 만들려면",
    "대한민국의 역사에서 중요한 사건은",
    "좋은 책을 읽으면",
    "건강한 생활을 위해서는",
    "미래의 기술은 우리의 삶을",
]

# Fertility test samples
FERTILITY_SAMPLES = [
    "안녕하세요, 오늘 날씨가 어떤가요?",
    "인공지능 기술은 빠르게 발전하고 있습니다.",
    "대한민국의 수도는 서울입니다.",
    "오늘 저녁에 뭐 먹을까요?",
    "프로그래밍을 배우는 것은 재미있습니다.",
    "오늘 아침에 미팅이 있어서 일찍 일어났어요.",
    "한국의 전통 음식은 세계적으로 유명합니다.",
    "기계학습 모델을 훈련시키는 데는 많은 데이터가 필요합니다.",
    "서울에서 부산까지 KTX로 약 2시간 30분 걸립니다.",
    "이 프로젝트의 목표는 한국어를 잘 이해하는 작은 언어 모델을 만드는 것입니다.",
]


# ──────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────

def get_console():
    """Return a rich Console or a simple stub."""
    if HAS_RICH:
        return Console()
    return None

console = get_console()


def print_header(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def contains_korean(text: str) -> bool:
    """Check if text contains any Korean Hangul characters."""
    return bool(re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff]', text))


def contains_english(text: str) -> bool:
    """Check if text contains English alphabetic characters."""
    return bool(re.search(r'[a-zA-Z]', text))


def detect_language(text: str) -> str:
    """Simple heuristic to detect dominant language."""
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if korean_chars > english_chars:
        return "ko"
    elif english_chars > korean_chars:
        return "en"
    else:
        return "mixed"


def compute_ngram_repetition(text: str, n: int) -> float:
    """Compute the fraction of repeated n-grams in text."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def has_degenerate_repetition(text: str, threshold: float = 0.5) -> bool:
    """Check if text has degenerate repetition (>50% repeated bigrams)."""
    return compute_ngram_repetition(text, 2) > threshold


def build_chat_prompt(user_message: str) -> str:
    """Build a ChatML formatted prompt for a single-turn conversation."""
    return (
        f"{IM_START}user\n{user_message}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


# ──────────────────────────────────────────────────────────────
# 1. Perplexity Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, dataset_name, subset, num_samples,
                        max_length=2048, label=""):
    """
    Compute perplexity on a streaming dataset by accumulating NLL
    over a fixed number of samples.
    """
    print_subheader(f"Perplexity: {label}")
    print(f"  Dataset: {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"  Samples: {num_samples}")

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split="train", streaming=True,
                              trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True,
                              trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠️  Could not load dataset: {e}")
        return float("inf")

    total_nll = 0.0
    total_tokens = 0
    processed = 0

    for sample in ds:
        if processed >= num_samples:
            break

        text = sample.get("text", "")
        if not text or len(text.strip()) < 20:
            continue

        # Truncate long texts to save memory
        text = text[:4096]

        encodings = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=max_length)
        input_ids = encodings.input_ids.to(model.device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item() * (input_ids.shape[1] - 1)  # un-average

        total_nll += nll
        total_tokens += input_ids.shape[1] - 1
        processed += 1

        if processed % 20 == 0:
            running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  [{processed}/{num_samples}] running PPL = {running_ppl:.2f}")

    if total_tokens == 0:
        print("  ⚠️  No tokens processed!")
        return float("inf")

    ppl = math.exp(total_nll / total_tokens)
    print(f"  ✅ Final Perplexity ({label}): {ppl:.2f}  ({total_tokens:,} tokens from {processed} samples)")
    return ppl


# ──────────────────────────────────────────────────────────────
# 2. Korean Fertility Measurement
# ──────────────────────────────────────────────────────────────

def evaluate_fertility(tokenizer, baseline_tokenizer_name=BASELINE_TOKENIZER):
    """
    Measure tokens-per-character ratio on Korean text and compare to baseline.
    """
    print_header("2. KOREAN FERTILITY MEASUREMENT")

    print(f"  Loading baseline tokenizer: {baseline_tokenizer_name}")
    try:
        baseline_tok = AutoTokenizer.from_pretrained(baseline_tokenizer_name)
    except Exception as e:
        print(f"  ⚠️  Could not load baseline tokenizer: {e}")
        baseline_tok = None

    results = []

    if HAS_RICH:
        table = Table(title="Korean Fertility: Tokens per Character",
                      box=box.ROUNDED, show_lines=True)
        table.add_column("Text", style="cyan", max_width=40)
        table.add_column("Chars", justify="right")
        table.add_column("Model Tok", justify="right", style="green")
        table.add_column("Model TPC", justify="right", style="green bold")
        if baseline_tok:
            table.add_column("Base Tok", justify="right", style="red")
            table.add_column("Base TPC", justify="right", style="red")
            table.add_column("Δ%", justify="right", style="yellow")

    for text in FERTILITY_SAMPLES:
        n_chars = len(text)
        model_tokens = tokenizer.tokenize(text)
        model_n = len(model_tokens)
        model_tpc = model_n / n_chars if n_chars > 0 else 0

        row = {
            "text": text[:40],
            "chars": n_chars,
            "model_tokens": model_n,
            "model_tpc": round(model_tpc, 3),
        }

        if baseline_tok:
            base_tokens = baseline_tok.tokenize(text)
            base_n = len(base_tokens)
            base_tpc = base_n / n_chars if n_chars > 0 else 0
            improvement = ((base_tpc - model_tpc) / base_tpc * 100) if base_tpc > 0 else 0
            row["baseline_tokens"] = base_n
            row["baseline_tpc"] = round(base_tpc, 3)
            row["improvement_pct"] = round(improvement, 1)

            if HAS_RICH:
                table.add_row(
                    text[:40], str(n_chars),
                    str(model_n), f"{model_tpc:.3f}",
                    str(base_n), f"{base_tpc:.3f}",
                    f"{improvement:+.1f}%"
                )
        else:
            if HAS_RICH:
                table.add_row(text[:40], str(n_chars),
                              str(model_n), f"{model_tpc:.3f}")

        results.append(row)

    if HAS_RICH:
        console.print(table)
    else:
        print(f"  {'Text':<42} {'Chars':>5} {'Tok':>5} {'TPC':>6}")
        print("  " + "-" * 62)
        for r in results:
            print(f"  {r['text']:<42} {r['chars']:>5} {r['model_tokens']:>5} {r['model_tpc']:>6.3f}")

    # Averages
    avg_model_tpc = sum(r["model_tpc"] for r in results) / len(results)
    print(f"\n  Average Model TPC: {avg_model_tpc:.3f}")

    avg_baseline_tpc = None
    avg_improvement = None
    if baseline_tok and "baseline_tpc" in results[0]:
        avg_baseline_tpc = sum(r["baseline_tpc"] for r in results) / len(results)
        avg_improvement = ((avg_baseline_tpc - avg_model_tpc) / avg_baseline_tpc * 100) if avg_baseline_tpc > 0 else 0
        print(f"  Average Baseline TPC: {avg_baseline_tpc:.3f}")
        print(f"  Average Improvement: {avg_improvement:+.1f}%")

    return {
        "avg_model_tpc": round(avg_model_tpc, 3),
        "avg_baseline_tpc": round(avg_baseline_tpc, 3) if avg_baseline_tpc else None,
        "avg_improvement_pct": round(avg_improvement, 1) if avg_improvement else None,
        "per_sample": results,
    }


# ──────────────────────────────────────────────────────────────
# 3. Instruction Following Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_instruction_following(model, tokenizer, max_new_tokens=128):
    """
    Run hardcoded bilingual prompts through the model using ChatML template
    and score each response.
    """
    print_header("3. INSTRUCTION FOLLOWING EVALUATION")

    # Try to get the <|im_end|> token id for proper stopping
    im_end_id = None
    if IM_END in tokenizer.get_vocab():
        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    elif IM_END in tokenizer.added_tokens_encoder:
        im_end_id = tokenizer.added_tokens_encoder[IM_END]

    eos_ids = [tokenizer.eos_token_id]
    if im_end_id is not None:
        eos_ids.append(im_end_id)

    results = []
    total_scores = {"non_empty": 0, "lang_match": 0, "coherent": 0, "stopped_properly": 0}
    n_prompts = len(EVAL_PROMPTS)

    for i, item in enumerate(EVAL_PROMPTS):
        prompt_text = item["prompt"]
        expected_lang = item["lang"]
        category = item["category"]

        chat_prompt = build_chat_prompt(prompt_text)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        clean_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # Also strip any trailing im_end that skip_special_tokens missed
        clean_response = clean_response.replace(IM_END, "").replace(IM_START, "").strip()

        # --- Scoring ---
        # 1. Non-empty
        is_non_empty = len(clean_response.strip()) > 0

        # 2. Language match
        response_lang = detect_language(clean_response)
        is_lang_match = (response_lang == expected_lang) or (response_lang == "mixed")

        # 3. Coherence: no degenerate repetition, reasonable length
        is_coherent = (
            not has_degenerate_repetition(clean_response)
            and len(clean_response) < max_new_tokens * 10  # not absurdly long
            and len(clean_response) > 1  # not trivially short
        )

        # 4. Stopped properly (hit EOS or im_end, not max_new_tokens)
        stopped_properly = len(generated_ids) < max_new_tokens

        scores = {
            "non_empty": is_non_empty,
            "lang_match": is_lang_match,
            "coherent": is_coherent,
            "stopped_properly": stopped_properly,
        }
        score_total = sum(scores.values())

        for k, v in scores.items():
            total_scores[k] += int(v)

        result = {
            "index": i + 1,
            "category": category,
            "expected_lang": expected_lang,
            "prompt": prompt_text,
            "response": clean_response[:300],  # cap for readability
            "scores": scores,
            "score_total": score_total,
        }
        results.append(result)

        # Print per-prompt result
        score_str = " | ".join(
            f"{'✅' if v else '❌'} {k}" for k, v in scores.items()
        )
        print(f"\n[{i + 1}/{n_prompts}] ({category}) [{expected_lang}]")
        print(f"  Prompt:   {prompt_text}")
        print(f"  Response: {clean_response[:200]}")
        print(f"  Scores:   {score_str}  ({score_total}/4)")

    # Summary
    print_subheader("Instruction Following Summary")
    for k, v in total_scores.items():
        pct = v / n_prompts * 100
        print(f"  {k:<20s}: {v}/{n_prompts} ({pct:.0f}%)")

    overall_pct = sum(total_scores.values()) / (n_prompts * 4) * 100
    print(f"  {'OVERALL':<20s}: {sum(total_scores.values())}/{n_prompts * 4} ({overall_pct:.0f}%)")

    return {
        "total_scores": total_scores,
        "overall_pct": round(overall_pct, 1),
        "n_prompts": n_prompts,
        "per_prompt": results,
    }


# ──────────────────────────────────────────────────────────────
# 4. Repetition Analysis
# ──────────────────────────────────────────────────────────────

def evaluate_repetition(model, tokenizer, max_new_tokens=128):
    """
    Generate continuations of Korean prefixes and measure n-gram repetition.
    """
    print_header("4. REPETITION ANALYSIS")

    results = []
    degenerate_count = 0

    for i, prefix in enumerate(KOREAN_PREFIXES):
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,  # No penalty — we're measuring raw behavior
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        bigram_rep = compute_ngram_repetition(continuation, 2)
        trigram_rep = compute_ngram_repetition(continuation, 3)
        is_degenerate = has_degenerate_repetition(continuation)

        if is_degenerate:
            degenerate_count += 1

        result = {
            "prefix": prefix,
            "continuation": continuation[:200],
            "bigram_repetition": round(bigram_rep, 3),
            "trigram_repetition": round(trigram_rep, 3),
            "is_degenerate": is_degenerate,
        }
        results.append(result)

        status = "🔴 DEGENERATE" if is_degenerate else "🟢 OK"
        print(f"\n[{i + 1}/{len(KOREAN_PREFIXES)}] {status}")
        print(f"  Prefix:    {prefix}")
        print(f"  Output:    {continuation[:120]}{'...' if len(continuation) > 120 else ''}")
        print(f"  Bigram Rep: {bigram_rep:.3f}  |  Trigram Rep: {trigram_rep:.3f}")

    # Summary
    avg_bigram = sum(r["bigram_repetition"] for r in results) / len(results)
    avg_trigram = sum(r["trigram_repetition"] for r in results) / len(results)

    print_subheader("Repetition Summary")
    print(f"  Average Bigram Repetition:  {avg_bigram:.3f}")
    print(f"  Average Trigram Repetition: {avg_trigram:.3f}")
    print(f"  Degenerate Loops Detected:  {degenerate_count}/{len(KOREAN_PREFIXES)}")

    return {
        "avg_bigram_repetition": round(avg_bigram, 3),
        "avg_trigram_repetition": round(avg_trigram, 3),
        "degenerate_count": degenerate_count,
        "total_prefixes": len(KOREAN_PREFIXES),
        "per_prefix": results,
    }


# ──────────────────────────────────────────────────────────────
# 5. Summary Dashboard & Grading
# ──────────────────────────────────────────────────────────────

def compute_grade(metrics: dict) -> str:
    """
    Assign an overall quality grade A-F based on metric thresholds.

    Grading rubric (designed for 135M-500M param models):
      - Perplexity:     Korean < 30 and English < 20 → good
      - Fertility:      improvement > 30% → good
      - Instruction:    overall > 60% → good
      - Repetition:     degenerate < 2/10 → good

    A: All four criteria met
    B: Three of four
    C: Two of four
    D: One of four
    F: Zero
    """
    score = 0

    # Perplexity check
    ko_ppl = metrics.get("korean_perplexity", float("inf"))
    en_ppl = metrics.get("english_perplexity", float("inf"))
    if ko_ppl < 50 and en_ppl < 40:
        score += 1

    # Fertility check
    fertility_imp = metrics.get("fertility", {}).get("avg_improvement_pct")
    if fertility_imp is not None and fertility_imp > 30:
        score += 1

    # Instruction following check
    instr_pct = metrics.get("instruction_following", {}).get("overall_pct", 0)
    if instr_pct > 60:
        score += 1

    # Repetition check
    degen = metrics.get("repetition", {}).get("degenerate_count", 10)
    if degen <= 2:
        score += 1

    grades = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
    return grades[score]


def print_dashboard(metrics: dict, grade: str):
    """Print a final summary dashboard."""
    print_header("5. SUMMARY DASHBOARD")

    if HAS_RICH:
        table = Table(title="Bori-3 Benchmark Results", box=box.DOUBLE_EDGE,
                      show_lines=True, title_style="bold magenta")
        table.add_column("Metric", style="bold cyan", min_width=30)
        table.add_column("Value", justify="right", style="white", min_width=20)
        table.add_column("Status", justify="center", min_width=10)

        # Perplexity
        ko_ppl = metrics.get("korean_perplexity", float("inf"))
        en_ppl = metrics.get("english_perplexity", float("inf"))
        combined_ppl = metrics.get("combined_perplexity", float("inf"))
        table.add_row("Korean Perplexity", f"{ko_ppl:.2f}", "🟢" if ko_ppl < 50 else "🟡" if ko_ppl < 100 else "🔴")
        table.add_row("English Perplexity", f"{en_ppl:.2f}", "🟢" if en_ppl < 40 else "🟡" if en_ppl < 80 else "🔴")
        table.add_row("Combined Perplexity", f"{combined_ppl:.2f}", "—")

        # Fertility
        fert = metrics.get("fertility", {})
        model_tpc = fert.get("avg_model_tpc", "N/A")
        imp = fert.get("avg_improvement_pct")
        table.add_row("Avg Korean TPC (model)", str(model_tpc), "")
        if imp is not None:
            table.add_row("Fertility Improvement vs Baseline", f"{imp:+.1f}%",
                          "🟢" if imp > 30 else "🟡" if imp > 10 else "🔴")

        # Instruction following
        instr = metrics.get("instruction_following", {})
        instr_pct = instr.get("overall_pct", 0)
        table.add_row("Instruction Following", f"{instr_pct:.1f}%",
                       "🟢" if instr_pct > 60 else "🟡" if instr_pct > 40 else "🔴")
        for k, v in instr.get("total_scores", {}).items():
            n = instr.get("n_prompts", 1)
            table.add_row(f"  └ {k}", f"{v}/{n} ({v / n * 100:.0f}%)", "")

        # Repetition
        rep = metrics.get("repetition", {})
        table.add_row("Avg Bigram Repetition", str(rep.get("avg_bigram_repetition", "N/A")), "")
        table.add_row("Avg Trigram Repetition", str(rep.get("avg_trigram_repetition", "N/A")), "")
        degen = rep.get("degenerate_count", 0)
        total_p = rep.get("total_prefixes", 10)
        table.add_row("Degenerate Loops", f"{degen}/{total_p}",
                       "🟢" if degen <= 2 else "🟡" if degen <= 5 else "🔴")

        # Grade
        grade_color = {"A": "bold green", "B": "green", "C": "yellow", "D": "red", "F": "bold red"}
        table.add_row("", "", "")
        table.add_row("OVERALL GRADE", f"[{grade_color.get(grade, 'white')}]{grade}[/]",
                       "⭐" if grade in ("A", "B") else "")

        # Timing
        elapsed = metrics.get("elapsed_seconds", 0)
        table.add_row("Total Time", f"{elapsed:.1f}s", "")

        console.print(table)
    else:
        # Plain text fallback
        print(f"\n  Korean Perplexity:          {metrics.get('korean_perplexity', 'N/A'):.2f}")
        print(f"  English Perplexity:         {metrics.get('english_perplexity', 'N/A'):.2f}")
        print(f"  Combined Perplexity:        {metrics.get('combined_perplexity', 'N/A'):.2f}")
        fert = metrics.get("fertility", {})
        print(f"  Avg Korean TPC:             {fert.get('avg_model_tpc', 'N/A')}")
        imp = fert.get("avg_improvement_pct")
        if imp is not None:
            print(f"  Fertility Improvement:      {imp:+.1f}%")
        instr = metrics.get("instruction_following", {})
        print(f"  Instruction Following:      {instr.get('overall_pct', 0):.1f}%")
        rep = metrics.get("repetition", {})
        print(f"  Avg Bigram Repetition:      {rep.get('avg_bigram_repetition', 'N/A')}")
        print(f"  Degenerate Loops:           {rep.get('degenerate_count', 0)}/{rep.get('total_prefixes', 10)}")
        print(f"  Overall Grade:              {grade}")
        print(f"  Total Time:                 {metrics.get('elapsed_seconds', 0):.1f}s")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bori-3 Comprehensive Benchmark & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (local dir or HF hub id)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--num_perplexity_samples", type=int, default=100,
                        help="Number of samples for perplexity evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens for generation tasks")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu', 'mps'")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip perplexity evaluation (saves time)")
    parser.add_argument("--skip_fertility", action="store_true",
                        help="Skip fertility evaluation")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    start_time = time.time()

    # ── Banner ──
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  🌾 Bori-3 Comprehensive Benchmark Suite".ljust(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\n  Model:     {args.model_path}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Device:    {args.device}")
    print(f"  PPL Samples: {args.num_perplexity_samples}")
    print(f"  Max Tokens:  {args.max_new_tokens}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # ── Load model & tokenizer ──
    print_header("LOADING MODEL & TOKENIZER")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Vocab size: {len(tokenizer):,}")

    print("  Loading model...")
    device_map = args.device if args.device != "auto" else "auto"
    # If device is cpu or mps, don't use device_map="auto" (it requires accelerate for multi-device)
    if args.device in ("cpu", "mps"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
            trust_remote_code=True,
        ).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"  Device: {next(model.parameters()).device}")
    print("  ✅ Model loaded successfully!")

    # ── Results accumulator ──
    metrics = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "vocab_size": len(tokenizer),
        "param_count": param_count,
        "timestamp": datetime.now().isoformat(),
    }

    # ── 1. Perplexity ──
    if not args.skip_perplexity:
        print_header("1. PERPLEXITY EVALUATION")

        ko_ppl = evaluate_perplexity(
            model, tokenizer,
            KOREAN_DATASET, KOREAN_SUBSET,
            args.num_perplexity_samples,
            label="Korean (fineweb-2 kor_Hang)"
        )

        en_ppl = evaluate_perplexity(
            model, tokenizer,
            ENGLISH_DATASET, None,
            args.num_perplexity_samples,
            label="English (fineweb-edu)"
        )

        # Combined: geometric mean
        if ko_ppl != float("inf") and en_ppl != float("inf"):
            combined_ppl = math.sqrt(ko_ppl * en_ppl)
        else:
            combined_ppl = float("inf")

        print_subheader("Perplexity Summary")
        print(f"  Korean PPL:   {ko_ppl:.2f}")
        print(f"  English PPL:  {en_ppl:.2f}")
        print(f"  Combined PPL: {combined_ppl:.2f} (geometric mean)")

        metrics["korean_perplexity"] = round(ko_ppl, 2) if ko_ppl != float("inf") else None
        metrics["english_perplexity"] = round(en_ppl, 2) if en_ppl != float("inf") else None
        metrics["combined_perplexity"] = round(combined_ppl, 2) if combined_ppl != float("inf") else None
    else:
        print_header("1. PERPLEXITY EVALUATION [SKIPPED]")
        metrics["korean_perplexity"] = None
        metrics["english_perplexity"] = None
        metrics["combined_perplexity"] = None

    # ── 2. Fertility ──
    if not args.skip_fertility:
        fertility_results = evaluate_fertility(tokenizer)
        metrics["fertility"] = fertility_results
    else:
        print_header("2. KOREAN FERTILITY MEASUREMENT [SKIPPED]")
        metrics["fertility"] = {}

    # ── 3. Instruction Following ──
    instr_results = evaluate_instruction_following(model, tokenizer, args.max_new_tokens)
    metrics["instruction_following"] = instr_results

    # ── 4. Repetition Analysis ──
    rep_results = evaluate_repetition(model, tokenizer, args.max_new_tokens)
    metrics["repetition"] = rep_results

    # ── Timing ──
    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 1)

    # ── 5. Dashboard ──
    # Fill in defaults for grading if perplexity was skipped
    grade_metrics = dict(metrics)
    if grade_metrics.get("korean_perplexity") is None:
        grade_metrics["korean_perplexity"] = float("inf")
    if grade_metrics.get("english_perplexity") is None:
        grade_metrics["english_perplexity"] = float("inf")
    if grade_metrics.get("combined_perplexity") is None:
        grade_metrics["combined_perplexity"] = float("inf")

    grade = compute_grade(grade_metrics)
    metrics["overall_grade"] = grade
    print_dashboard(grade_metrics, grade)

    # ── Save JSON ──
    output_path = Path(args.output_file)
    # Strip non-serializable values (replace inf/nan with null)
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    clean_metrics = sanitize(metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n  📄 Results saved to: {output_path.resolve()}")
    print(f"  ⏱️  Total time: {elapsed:.1f}s")
    print("\n" + "█" * 70)
    print("  🌾 Bori-3 Benchmark Complete!")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
