# Bori (보리): Bilingual Small Language Model (SLM) Pipeline

Bori is a highly optimized, bilingual (Korean-English) Small Language Model pipeline built upon the **SmolLM2-135M** architecture. This repository tracks the complete evolutionary history of Bori's development, transitioning from a basic fine-tuning setup to a state-of-the-art, fully automated pre-training and alignment infrastructure.

---

## 📂 Repository Roadmap

The codebase is organized in evolutionary versions, allowing you to trace design decisions, training infrastructure updates, and model configurations over time:

```
Bori/
├── .gitignore                     # Custom git safety configurations
├── README.md                      # Unified documentation
├── Bori_Version1/                 # Phase 1: Baseline pre-training & SFT scripts
│   ├── bori_code/                 # Core source and execution folder
│   └── kaggle_runner.ipynb        # Kaggle integration workbook
│
├── Bori_Version2/                 # Phase 2: Refined CPT split and EEVE warm-up
│   ├── Data/                      # Intermediate fine-tuning datasets
│   ├── bori_v2/                   # Multi-GPU config & continuous pre-training (10% replay)
│   ├── run_local_dry_run.py       # Local pipeline testing script
│   └── kaggle_runner_v2.ipynb     # Kaggle multi-node workbook
│
└── Bori_Version3/                 # Phase 3 (Latest): Production-Grade Infrastructure
    ├── Data/                      # Dataset resources (DeepSeek V4 datasets)
    ├── run_local_dry_run.py       # Updated local CPU dry run integration test
    └── bori_v3/                   # Core production codebase
        ├── configs/               # Single & multi-node training configurations
        ├── src/                   # Core modules (EEVE initialization, packed dataset loaders)
        └── scripts/               # Production-grade orchestration scripts
            ├── train_korean_tokenizer.py  # Korean Byte-Level BPE training
            ├── merge_tokenizers.py        # Tokenizer merging & fertility comparisons
            ├── cpt.py                     # CPT (Phase 1a Warmup & Phase 1b WSD pre-training)
            ├── sft.py                     # Supervised Fine-Tuning with WSD & step offset
            ├── chat_ui.py                 # Gradio chat interface
            ├── test_fertility.py          # Tokenizer fertility analyzer
            └── test_inference.py          # Standalone inference validation
```

---

## 🛠️ Key Architectural Highlights (Bori v3)

### 1. Vocabulary Expansion & EEVE Initialization
* **The Problem**: Pre-trained English-centric SLMs (like SmolLM2) represent Korean prose very inefficiently, splitting single syllables into multiple bytes (high fertility).
* **The Solution**: We train a custom standalone Korean Byte-Level BPE tokenizer and merge it with the base tokenizer, adding **757 highly efficient Korean tokens** (Final vocab: `49,909`).
* **EEVE (Subword-based) Initialization**: In `src/model.py`, newly added Korean token embeddings are **not** initialized randomly. Instead, we use the **EEVE strategy**, which initializes each new token from the mean embeddings of its English subwords from the base tokenizer. This gives the model an excellent starting approximation, lowering starting loss.
* **Fertility Improvement**: Measured BPE fertility improvements on Korean prose range from **1.76x to 3.12x** vocabulary compression, meaning the model can process and generate Korean text up to 3x faster with smaller sequence lengths!

### 2. Multi-Phase Continuous W&B Charting
Standard Hugging Face training resets the step count (`state.global_step = 0`) at the beginning of each phase or script execution, causing graphs to stack or overlay confusingly in W&B.
Bori v3 introduces the **`GlobalStepCallback`**:
* **Step Offset Injection**: Accumulates steps continuously by taking a `--step_offset` argument.
* **W&B X-Axis Sync**: Logs a custom `global_step_accumulated` metric. In your W&B dashboard, simply change the panel's X-axis to `global_step_accumulated` to view Phase 1a $\rightarrow$ Phase 1b $\rightarrow$ SFT as one clean, continuous, and non-jagged optimization curve!

### 3. Automated Warmup-Stable-Decay (WSD) Scheduler
Top-tier pre-training pipelines rely on WSD schedules rather than simple cosine decay. Bori v3 provides a fully automated custom PyTorch WSD scheduler option (`--lr_scheduler_type wsd`):
* **Warmup Phase**: Ramps up learning rate linearly from 0 to peak.
* **Stable Phase**: Keeps the learning rate flat at peak to maximize optimizer progress over high-entropy web text.
* **Decay Phase**: cosinely decays learning rate down to a floor (e.g., 10%) at the very end to consolidate weights.
* **Dry Run Clamping Guardrails**: Dynamically clamps `warmup_steps` and `decay_steps` against `max_steps` to protect against negative math in short test or dry-run environments.

---

## 🚀 Execution & Usage Guide

### A. Run the Local Dry Run (End-to-End Pipeline Verification)
To verify your training scripts, tokenizer merger, custom schedulers, and callbacks locally on CPU (2-4 steps per phase):
```bash
cd Bori_Version3
python3 run_local_dry_run.py
```

### B. Full Bilingual Pre-Training (CPT) on GPU
CPT is run in two sequential stages in a single command using `cpt.py`:
1. **Phase 1a (Embedding Warm-up)**: Freezes the backbone and trains only the new Korean embeddings to align them with the model's internal representations.
2. **Phase 1b (Full CPT with WSD)**: Unfreezes all parameters and trains the model on a mix of Korean web text and 10% English replay data to prevent catastrophic forgetting.

```bash
python3 bori_v3/scripts/cpt.py \
    --lr_scheduler_type wsd \
    --max_steps 10000 \
    --warmup_steps 500 \
    --decay_steps 1500 \
    --min_lr_ratio 0.1 \
    --output_dir ./checkpoints \
    --tokenizer_path /path/to/merged_tokenizer \
    --wandb_project bori-v3-cpt
```

### C. Supervised Fine-Tuning (SFT)
Once CPT is complete, run conversational instruction alignment using `sft.py`. Ensure you carry over the accumulated step offset to maintain a unified W&B curve:
```bash
python3 bori_v3/scripts/sft.py \
    --model_path ./checkpoints/phase_1b \
    --tokenizer_path /path/to/merged_tokenizer \
    --dataset_name heegyu/open-korean-instructions \
    --lr_scheduler_type cosine \
    --step_offset 11000 \
    --output_dir ./sft_checkpoints \
    --wandb_project bori-v3-sft
```

---

## 🤝 Contributing & License
This project tracks custom experimental research on Bilingual Small Language Models. Feel free to open issues or pull requests to improve tokenization packing, memory footprint, or multi-GPU pipeline parallelisms. Distributed under the MIT License.
