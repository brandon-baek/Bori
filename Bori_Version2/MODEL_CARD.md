---
language:
- ko
- en
license: mit
tags:
- causal-lm
- SLM
- continuous-pretraining
- EEVE-initialization
- WSD-scheduler
base_model: HuggingFaceTB/SmolLM2-135M
pipeline_tag: text-generation
---

# Bori-Version2 (Phase 2 Stable Release)

> [!NOTE]
> **Status: Active / Stable Release**
> This model represents the first successful, fully optimized bilingual (Korean-English) Small Language Model in the Bori lineage. It utilizes Continuous Pre-Training (CPT) and advanced vocabulary expansion.

Bori-Version2 is a highly optimized, bilingual (Korean-English) Small Language Model built by continuously pre-training the **SmolLM2-135M** architecture on mixed bilingual text corpora. It features custom token initialization, catastrophic forgetting mitigation, and a Warmup-Stable-Decay (WSD) scheduler.

---

## Model Details

- **Developed by:** brandon_baek
- **Model Type:** Causal Language Model (Transformer-based decoder)
- **Base Model:** HuggingFaceTB/SmolLM2-135M
- **Parameters:** ~135M (after vocabulary resizing)
- **Language(s):** Korean (한국어), English
- **License:** MIT

---

## Key Methodological Enhancements

### 1. Bilingual Vocabulary Expansion & Fertility Reduction
* **The Problem:** Base English-centric SLMs process Korean very inefficiently, splitting single Korean syllables into multiple raw bytes, leading to high sequence fertility and sluggish inference speeds.
* **The Solution:** Added **757 highly efficient Korean tokens** via a custom standalone Korean Byte-Level BPE tokenizer. 
* **Fertility Gains:** Achieved a **1.76x to 3.12x vocabulary compression ratio** on Korean prose, enabling Bori to process and generate Korean text up to 3x faster with smaller context windows.

### 2. EEVE Subword Embedding Initialization
* To prevent training instability and catastrophic forgetting during initial steps, newly added Korean token embeddings were **not** initialized randomly. Instead, we adopted the **EEVE strategy**, initializing each new token's input and output weights from the average pre-trained embeddings of its constituent English subwords.

### 3. Continuous Pre-Training (CPT) Orchestration
* **Phase 1a (Embedding Warm-up):** Freezes the transformer backbone and trains *only* the resized embedding parameters for 1,000 steps to align the new Korean subword semantic spaces.
* **Phase 1b (Full CPT):** Unfreezes all parameters and pre-trains on a corpus mixed with a **10% English replay dataset** (`HuggingFaceFW/fineweb-edu-dedup`) to completely safeguard the model's original English reasoning.

### 4. Warmup-Stable-Decay (WSD) LR Scheduler
* Replaced standard linear/cosine schedules with a custom **WSD Scheduler** (`--lr_scheduler_type wsd`) that ramps linearly to peak, holds the learning rate flat during the stable phase over web text, and cosinely decays at the end (annealing) to consolidate representations.

---

## Training Hyperparameters

* **Optimizer:** AdamW (`adamw_torch`)
* **Base Learning Rate (CPT):** 2e-4
* **Embedding Warm-up Learning Rate:** 1e-3
* **Warm-up Steps:** 500
* **Sequence Context Length:** 2048
* **Precision:** FP16 mixed-precision (trained under Accelerate DDP configs)

---

## Intended Use & Recommendations
* **Intended Use:** Dialogue systems, bilingual text autocomplete, lightweight agents, and local edge deployments.
* **Loading the Model:**
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_path = "brandon_baek/Bori-Version2" # Replace with your HF hub path or local folder
  
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(model_path)
  ```
