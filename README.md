# Video Retrieval — LanguageBind + LoRA Fine-tuning

Text-to-video retrieval system for Marvel series clips, built on [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) with QLoRA fine-tuning.

Given a natural language query like *"Daredevil fighting someone in a dark hallway"*, the system retrieves the most semantically relevant video clips from the dataset.

---

## Repository Structure

```
video-retrieval-final.ipynb   ← Main notebook (this is the one to use)
fork-of-load-languagebind.ipynb  ← Original exploratory version (archived)
README.md
```

---

## What Changed: `video-retrieval-final` vs `fork-of-load-languagebind`

### Bug Fixes

| Issue | Old Version | Final Version |
|---|---|---|
| **Learning rate** | `1e-3` (too aggressive, caused oscillating val loss) | `2e-4` (stable convergence) |
| **Temperature** | Fixed scalar `0.07` hardcoded in loss | **Learnable** `log_temperature` parameter trained jointly with LoRA |


### Training Improvements

- **Early stopping** (patience = 4) — old version ran all 10 epochs regardless of val loss
- **Gradient accumulation steps** doubled from 4 → 8, simulating a larger effective batch without extra VRAM
- **LR scheduler** logged per epoch so decay is visible in output

### Evaluation Suite (new in final version)

The old notebook had only a basic loss + R@1/5/10 printout. The final notebook adds a full evaluation suite that runs entirely on saved `.npy` embeddings (no GPU required):

| Section | What it measures |
|---|---|
| UMAP visualization | Are video and text embeddings geometrically aligned? |
| Cosine similarity distribution | Separation between matched vs non-matched pairs |
| Modality gap measurement | L2 distance between video and text embedding cloud centers |
| Nearest-neighbour consistency | Do visually similar clips cluster together (video-only)? |
| **Stress test — natural language queries** | Retrieval on out-of-distribution, conversational queries |
| Similarity matrix heatmap | Full N×N matrix — diagonal should dominate |
| Final test set evaluation | R@1, R@5, R@10 on held-out test split |

## Results (Final Model)

| Metric | Score |
|---|---|
| Test Loss | ~0.318 |
| Recall@1 | ~42% |
| Recall@5 | ~80% |
| Recall@10 | ~86% |

---

## Setup

### Requirements

Run on **Kaggle GPU** (T4 × 2 or P100 recommended). Key dependencies:

```bash
uv pip install transformers==4.35.0 tokenizers==0.14.0
uv pip install bitsandbytes peft==0.10.0
uv pip install decord torchvision accelerate triton
```

### Dataset

CSV file: `final_video_query_refined.csv`  
Format: `video_path, query` pairs from Marvel series clips.  
Available on Kaggle at: `gsejal/video-query-mac`

### Secrets

Set the following in Kaggle Secrets before running the HuggingFace push cell:

| Key | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace write token |

---

## How It Works

1. **Base model:** `LanguageBind/LanguageBind_Video_FT` — a CLIP-style video-language model pre-trained on image, video, audio, depth, and thermal modalities
2. **Quantization:** 4-bit NF4 (QLoRA) to fit in GPU memory
3. **Fine-tuning:** LoRA adapters on `q_proj` and `v_proj` of the transformer (~0.5% of total params)
4. **Loss:** Symmetric contrastive loss (InfoNCE) with learnable temperature
5. **Inference:** Query → text embedding → cosine similarity against all video embeddings → top-K results


