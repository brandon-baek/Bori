# 🌾 Bori (보리): Bilingual Small Language Model (SLM) Pipeline

Bori is a highly optimized, bilingual (Korean-English) Small Language Model pipeline built upon the **SmolLM2** architecture. This repository tracks the complete evolutionary history of Bori's development, transitioning from a basic fine-tuning setup to a state-of-the-art, fully automated pre-training and alignment infrastructure.

---

## 📂 Repository Roadmap

The codebase is organized in evolutionary versions, allowing you to trace design decisions, training infrastructure updates, and model configurations over time:

```
Bori/
├── Bori-1/                        # Phase 1: Baseline Qwen2 pre-training & SFT scripts
├── Bori-2/                        # Phase 2: SmolLM2-135M, EEVE warm-up, and WSD scheduler
└── Bori-3/                        # Phase 3 (Latest): Production-Grade Infrastructure
    ├── codebase/                  # Core source code (mounted to Kaggle)
    │   ├── configs/               # Hyperparameter configurations (cpt_config.yaml)
    │   ├── scripts/               # Production-grade orchestration scripts
    │   │   ├── setup_kaggle.py    # Automated Kaggle environment setup
    │   │   ├── cpt.py             # CPT (Phase 1a Warmup & Phase 1b pre-training)
    │   │   ├── sft.py             # Supervised Fine-Tuning with 50/50 bilingual mix
    │   │   ├── benchmark.py       # Comprehensive 5-axis evaluation suite
    │   │   └── ...
    │   └── src/                   # Core modules (EEVE initialization, custom collators)
    │
    └── kaggle_runner_v3.ipynb     # Thin control-panel notebook for Kaggle execution
```

---

## 🛠️ Key Architectural Highlights (Bori-3)

### 1. Scaling to SmolLM2-360M
Bori-3 upgrades the base model from 135M to `SmolLM2-360M`, dramatically improving reasoning capability and instruction adherence while remaining trainable on free Kaggle T4 GPUs via gradient checkpointing and SDPA attention.

### 2. Vocabulary Expansion & EEVE Initialization
* **The Problem**: Pre-trained English-centric SLMs represent Korean prose very inefficiently.
* **The Solution**: We train a custom standalone Korean Byte-Level BPE tokenizer and merge it with the base tokenizer, adding **8,981 highly efficient Korean tokens** (Final Vocab: 58,133).
* **EEVE Initialization**: In `src/model.py`, newly added Korean token embeddings are initialized from the mean embeddings of their English subwords from the base tokenizer, giving the model an excellent starting approximation.

### 3. Response-Only Loss (SFT Masking)
Bori-3 implements a custom `SFTDataCollator` that strictly enforces prompt-masking (setting user/system turn labels to `-100`). The model calculates loss **only** on its own assistant responses, drastically reducing hallucinations.

### 4. 50/50 Bilingual SFT Mix
The SFT pipeline uses `datasets.interleave_datasets` to seamlessly stream a perfectly balanced 50/50 mix of Korean and English instructional data to prevent catastrophic forgetting of English capabilities:
- `konglish-synthetic-instruct` (Korean instructions)
- `korean_safe_conversation` (Korean alignment)
- `ko_wikidata_QA` (Korean factual QA)
- `no_robots` (English human-written instructions)
- `alpaca-cleaned` (English diverse instructions)

### 5. Automated Benchmark Suite
A standalone `benchmark.py` script automatically grades model checkpoints (A-F) across 5 axes: Korean/English Perplexity, Tokenizer Fertility, Repetition analysis, and hardcoded Bilingual Instruction Following.

---

## 🚀 Execution & Usage Guide

### A. Environment Setup
The repository is designed to run seamlessly on Kaggle. The `setup_kaggle.py` script handles extracting the codebase, restoring prior checkpoints across sessions, generating Accelerate configs, and pre-caching models.

### B. Full Bilingual Pre-Training (CPT)
CPT is run in two sequential stages in a single command using `cpt.py`:
1. **Phase 1a (Embedding Warm-up)**: Freezes the backbone and trains only the new Korean embeddings.
2. **Phase 1b (Full CPT)**: Unfreezes all parameters and trains the model on a mix of 85% Korean web text and 15% English replay data using a WSD (Warmup-Stable-Decay) learning rate schedule.

### C. Supervised Fine-Tuning (SFT)
Once CPT is complete, run conversational instruction alignment using `sft.py`. The control panel notebook allows switching between models and phases with single boolean toggles.

---

## 🔗 Published Models (Hugging Face)
- [Bori-2 135M Base](https://huggingface.co/brandonbaek/Bori-2-135M-Base)
- [Bori-2 135M Instruct](https://huggingface.co/brandonbaek/Bori-2-135M-Instruct)
- [Bori-1 0.6B Base](https://huggingface.co/brandonbaek/Bori-1-0.6B-Base)

## 🤝 Contributing & License
This project tracks custom experimental research on Bilingual Small Language Models. Feel free to open issues or pull requests. Distributed under the MIT License.
