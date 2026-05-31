# Changelog

All notable changes to the Bori (보리) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-05-31
### Added
- **Unified ML Tooling Ecosystem**:
  - `download_and_test.py`: Interactive CLI supporting stateful inference and dialogue templates.
  - `upload_checkpoint.py`: Comprehensive tool to validate and upload local checkpoints directly to Weights & Biases (W&B) Artifacts.
  - `kaggle_uploader.py`: Safe, import-order independent module designed to upload checkpoints directly from Kaggle Notebook instances.
  - `kaggle_chat.py`: In-notebook stateful chat runner for remote GPU acceleration.
  - `run_chat.sh`: Shell script utility to discover, launch, and activate a virtual environment wrapper.
- **Selective Artifact Downloading**:
  - Automatically filters and skips massive training files (e.g., `optimizer.pt`, `trainer_state.json`) during model checkpoint retrieval to save **~1.1 GB** of bandwidth (a 70% decrease in download size) and 3x faster transfer speeds.
- **Live Terminal Controllers**:
  - Added commands inside interactive sessions (`/mode` for instant SFT Chat to Raw Autocomplete dynamic toggling, `/clear` to wipe chat history, and `/exit` to trigger safe disk space cleanups).
- **Silent Cloud Authentication**:
  - Silent W&B authentication using Kaggle Secrets mapping to prevent cell prints.

### Changed
- Re-routed local dry-run integration testing to incorporate multi-stage operational test validations.

---

## [2.0.0] - 2026-05-30
### Added
- **SmolLM2 base integration**:
  - Transitioned the core pre-training model from scratch initialization to Continuous Pre-Training (CPT) on top of the highly optimized **SmolLM2-135M** architecture.
- **Bilingual Vocabulary Expansion**:
  - Developed a standalone Korean Byte-Level BPE tokenizer, adding **757 highly efficient Korean tokens** to achieve vocabulary compression (fertility compression from 1.76x to 3.12x).
- **EEVE Subword Embedding Initialization**:
  - Custom embedding initializer inside `src/model.py` which initializes new Korean tokens with the mean embeddings of their corresponding constituent English subwords rather than setting them randomly.
- **Continuous Pre-Training (CPT) Orchestration**:
  - Automated Phase 1a (Embedding Warm-up with frozen backbones) and Phase 1b (Full parameter unfreeze).
- **English Replay Strategy**:
  - Combined training datasets with a **10% English replay ratio** (`HuggingFaceFW/fineweb-edu-dedup`) during CPT to prevent catastrophic forgetting.
- **Warmup-Stable-Decay (WSD) LR Scheduler**:
  - PyTorch learning rate scheduler with flat peak learning rate during high-entropy training steps, alongside cosine decay consolidated structures and short dry-run clamping guardrails.
- **Apache Arrow Schema Normalization**:
  - Automatically maps and cleans heterogeneous dataset columns to a unified, single-column `"text"` layout prior to dataset interleaving to prevent Arrow schema schema errors.
- **Empty Sequence Guardrails**:
  - Explicit length and token filters to prevent attention matrix reshape crashes during training on Hugging Face / Llama models.
- **Custom HF Trainer Callbacks**:
  - `GlobalStepCallback`: Accumulates steps continuously across independent scripts (`cpt.py` -> `sft.py`) via `--step_offset` to enable a unified, non-jagged W&B learning curve.
  - `WandbCheckpointCallback`: Limits active checkpoint uploads to W&B to `max_keep=3` to save storage.
  - `SaveTokenizerCallback`: Ensures the merged tokenizer is stored directly inside HF checkpoint folders.

---

## [1.0.0] - 2026-05-18
### Added
- **Baseline Training Scripts**:
  - `pretrain.py`: Scratch pre-training script using Qwen2 architecture.
  - `sft.py`: Basic supervised fine-tuning script.
  - `test_inference.py`: Baseline inference testing.
- **Configuration & Integration**:
  - Added baseline `configs/qwen2_bori_config.json`.
  - Added Kaggle runner notebooks for remote testing.
