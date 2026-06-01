#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Bori-3 Benchmark Runner
# ──────────────────────────────────────────────────────────────
# Example usage for running the comprehensive benchmark suite.
#
# Prerequisites:
#   pip install torch transformers datasets rich accelerate
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Default values (override via environment or CLI) ──
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUTPUT_FILE="${OUTPUT_FILE:-benchmark_results.json}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-128}"
DEVICE="${DEVICE:-auto}"

# ── Parse arguments ──
usage() {
    echo ""
    echo "Usage: $0 --model_path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --model_path PATH       Path to model checkpoint or HF hub id"
    echo ""
    echo "Optional:"
    echo "  --tokenizer_path PATH   Path to tokenizer (defaults to model_path)"
    echo "  --output_file PATH      Output JSON file (default: benchmark_results.json)"
    echo "  --num_samples N         Number of perplexity samples (default: 100)"
    echo "  --max_tokens N          Max new tokens for generation (default: 128)"
    echo "  --device DEVICE         Device: auto, cuda, cpu, mps (default: auto)"
    echo "  --skip_perplexity       Skip perplexity evaluation"
    echo "  --skip_fertility        Skip fertility evaluation"
    echo "  --quick                 Quick mode: 20 samples, skip perplexity"
    echo ""
    echo "Examples:"
    echo "  # Full benchmark on a local checkpoint"
    echo "  $0 --model_path ./checkpoints/sft-final"
    echo ""
    echo "  # Quick eval (no perplexity, for fast iteration)"
    echo "  $0 --model_path ./checkpoints/sft-final --quick"
    echo ""
    echo "  # Custom sample count on CPU"
    echo "  $0 --model_path ./checkpoints/cpt-10k --num_samples 50 --device cpu"
    echo ""
    echo "  # Evaluate a HuggingFace Hub model"
    echo "  $0 --model_path brandonbaek/bori-3-135m --output_file hub_benchmark.json"
    echo ""
    exit 1
}

EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)     MODEL_PATH="$2"; shift 2 ;;
        --tokenizer_path) TOKENIZER_PATH="$2"; shift 2 ;;
        --output_file)    OUTPUT_FILE="$2"; shift 2 ;;
        --num_samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --max_tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --skip_perplexity)  EXTRA_ARGS="$EXTRA_ARGS --skip_perplexity"; shift ;;
        --skip_fertility)   EXTRA_ARGS="$EXTRA_ARGS --skip_fertility"; shift ;;
        --quick)
            NUM_SAMPLES=20
            EXTRA_ARGS="$EXTRA_ARGS --skip_perplexity"
            shift ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "❌ Error: --model_path is required."
    usage
fi

# ── Build command ──
CMD="python ${SCRIPT_DIR}/benchmark.py \
    --model_path ${MODEL_PATH} \
    --output_file ${OUTPUT_FILE} \
    --num_perplexity_samples ${NUM_SAMPLES} \
    --max_new_tokens ${MAX_TOKENS} \
    --device ${DEVICE}"

if [ -n "$TOKENIZER_PATH" ]; then
    CMD="$CMD --tokenizer_path ${TOKENIZER_PATH}"
fi

CMD="$CMD $EXTRA_ARGS"

# ── Run ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌾 Bori-3 Benchmark Runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model:       ${MODEL_PATH}"
echo "  Tokenizer:   ${TOKENIZER_PATH:-<same as model>}"
echo "  Output:      ${OUTPUT_FILE}"
echo "  PPL Samples: ${NUM_SAMPLES}"
echo "  Max Tokens:  ${MAX_TOKENS}"
echo "  Device:      ${DEVICE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

eval $CMD

echo ""
echo "✅ Benchmark complete! Results saved to: ${OUTPUT_FILE}"
