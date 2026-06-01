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

# Bori-3 (Phase 3 Stable Release)

> [!NOTE]
> **Status: Active / Stable Release**
> This model represents the production-grade bilingual (Korean-English) Small Language Model in the Bori lineage. It features production-grade infrastructure, vocabulary expansion, WSD scheduler, and full step offset sync.

Bori-3 is a highly optimized, bilingual (Korean-English) Small Language Model built by continuously pre-training and supervised fine-tuning the **SmolLM2-135M** architecture on mixed bilingual text corpora. It features custom token initialization, catastrophic forgetting mitigation, and a Warmup-Stable-Decay (WSD) scheduler.

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

- **Bilingual Vocabulary Expansion & Fertility Reduction:** Added **757 highly efficient Korean tokens** via a custom standalone Korean Byte-Level BPE tokenizer, achieving a **1.76x to 3.12x vocabulary compression ratio** on Korean prose.
- **EEVE Subword Embedding Initialization:** Adopted the **EEVE strategy**, initializing new Korean embeddings from the average pre-trained embeddings of their English subwords to prevent training instability.
- **Continuous Pre-Training (CPT) Orchestration:** Automates frozen backbone embedding warmup followed by full parameter unfreezing with **10% English replay** (`HuggingFaceFW/fineweb-edu-dedup`) to prevent catastrophic forgetting.
- **Warmup-Stable-Decay (WSD) LR Scheduler:** Replaced linear/cosine schedules with a custom **WSD Scheduler** that ramps linearly to peak, holds flat during stable web training, and cosinely decays at the end to consolidate representations.

---

## Training Hyperparameters

- **Optimizer:** AdamW (`adamw_torch`)
- **Base Learning Rate (CPT):** 2e-4
- **Embedding Warm-up Learning Rate:** 1e-3
- **Warm-up Steps:** 500
- **Sequence Context Length:** 2048
- **Precision:** FP16 mixed-precision

---

## Intended Use & Recommendations
- **Intended Use:** Dialogue systems, bilingual text autocomplete, lightweight agents, and local edge deployments.
- **Loading the Model:**
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_path = "brandon_baek/Bori-3" # Replace with your HF hub path or local folder
  
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(model_path)
  ```
