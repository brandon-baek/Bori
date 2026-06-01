---
language:
- ko
- en
license: mit
tags:
- causal-lm
- SLM
- scratch-pretraining
- experimental
model_type: qwen2
pipeline_tag: text-generation
---

# Bori-Version1 (Phase 1 Baseline)

> [!WARNING]
> **Status: Deprecated / Discontinued (Experimental Failure)**
> This model represents a highly limited, early experimental run that was terminated prematurely at step 1,000 of scratch pre-training. It is not intended for production environments.

Bori-Version1 is an experimental bilingual (Korean-English) Small Language Model built by randomly initializing a model from configuration using the **Qwen2** backbone architecture and training from scratch. 

---

## Model Details

- **Developed by:** brandon_baek
- **Model Type:** Causal Language Model (Transformer-based decoder)
- **Base Architecture:** Qwen2 (`configs/qwen2_bori_config.json`)
- **Parameters:** ~135M (Randomly initialized)
- **Language(s):** Korean (한국어), English
- **License:** MIT

---

## Training History & Limitations

### ⚠️ Pre-Training Failure Analysis
* **Premature Termination:** The pre-training run was halted prematurely at **step 1,000** due to poor optimization efficiency.
* **Initialization from Scratch:** Attempting to train a 135M parameter model completely from scratch on a limited, small-scale dataset led to slow alignment, high initial loss variance, and weak linguistic representations. 
* **Capabilities:** The model acts as a basic, highly disjointed next-token autocomplete engine. It produces grammatically incoherent texts in both languages and lacks standard structural understanding or dialogue cohesion.

### Training Hyperparameters
* **Optimizer:** Adafactor
* **Learning Rate:** 3e-4
* **Learning Rate Scheduler:** Cosine Decay
* **Batch Size:** 1
* **Gradient Accumulation Steps:** 32
* **Sequence Context Length:** 2048
* **Precision:** FP16 mixed-precision (trained on a single NVIDIA T4 GPU)

---

## Intended Use & Recommendations
* **Intended Use:** Historical tracking, pipeline auditing, and comparative evolutionary research.
* **Recommendations:** **Do not use.** It is highly recommended to use **Bori-Version2** (which employs SmolLM2 pre-trained initialization and Continuous Pre-Training) for any actual dialogue generation or token representation tasks.
