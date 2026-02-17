# Transfer Learning & Fine-Tuned Models on a Rust/Axum Backend

A comprehensive guide to fine-tuning pre-trained foundation models in Python and serving them through an Axum-based Rust backend on a CPU-only Linux server. Covers text transformers, vision models, and embedding models with full code for every step: fine-tuning, ONNX export, quantization, Rust inference, and Axum integration.

> **Last updated:** February 17, 2026

---

## Table of Contents

- [What This Covers (and What It Doesn't)](#what-this-covers-and-what-it-doesnt)
- [The Pipeline at a Glance](#the-pipeline-at-a-glance)
- [Python Fine-Tuning Toolchain](#python-fine-tuning-toolchain)
  - [transformers v5 + PEFT — LoRA and full fine-tuning](#transformers-v5--peft)
  - [sentence-transformers — embedding model fine-tuning](#sentence-transformers)
  - [SetFit — few-shot contrastive fine-tuning](#setfit)
  - [timm — vision model fine-tuning](#timm--pytorch-image-models)
  - [ultralytics — YOLO fine-tuning](#ultralytics--yolov8v11)
  - [OpenCLIP — vision-language fine-tuning](#openclip)
- [ONNX Export Pipeline](#onnx-export-pipeline)
  - [optimum: the recommended export tool](#optimum--the-recommended-export-tool)
  - [LoRA merge and export](#lora-merge-and-export)
  - [Graph optimization levels](#graph-optimization-levels)
  - [INT8 quantization](#int8-quantization)
  - [Model size reference](#model-size-reference)
- [Rust Inference Libraries](#rust-inference-libraries)
  - [ort — primary ONNX runtime](#ort--primary-onnx-runtime)
  - [candle — native HuggingFace models](#candle--native-huggingface-models)
  - [fastembed-rs — pre-packaged embeddings](#fastembed-rs--pre-packaged-embeddings)
  - [rust-bert — pre-built NLP pipelines](#rust-bert--pre-built-nlp-pipelines)
  - [tokenizers — text tokenization in Rust](#tokenizers--text-tokenization-in-rust)
  - [tch-rs + TorchScript (legacy path)](#tch-rs--torchscript-legacy-path)
- [Use Case Walkthroughs](#use-case-walkthroughs)
  - [Text classification](#1-text-classification)
  - [Sentence embeddings and semantic search](#2-sentence-embeddings--semantic-search)
  - [Image classification](#3-image-classification)
  - [Named Entity Recognition (NER)](#4-named-entity-recognition-ner)
  - [Feature extraction (frozen backbone)](#5-feature-extraction-frozen-backbone)
- [Axum Integration Patterns](#axum-integration-patterns)
- [Practical Gotchas](#practical-gotchas)
- [Decision Matrix](#decision-matrix)

---

## What This Covers (and What It Doesn't)

The [classical ML guide](./README.md#recommended-architecture-classical-ml-on-a-rustaxum-backend) covered training from scratch with sklearn, linfa, and XGBoost. This document covers a different category: **pre-trained foundation models** that are adapted to a new task via fine-tuning or feature extraction.

**Transfer learning** here means one of:
- **Full fine-tuning** — update all weights of a pre-trained model on your dataset.
- **Feature extraction** — freeze the backbone, add a new head, train only the head.
- **LoRA / PEFT** — inject low-rank adapter layers; train only adapter parameters (~0.1–1% of total weights).
- **Contrastive fine-tuning** — fine-tune with a custom metric learning objective (sentence-transformers, SetFit).

**Why this is harder than classical ML:**
1. Models are large (tens to hundreds of MB) — memory planning matters.
2. ONNX export for transformers has more moving parts (dynamic shapes, opset versions, custom ops).
3. Text models require a tokenizer in Rust, not just a feature vector.
4. First-inference warmup latency is significant.

---

## The Pipeline at a Glance

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TRANSFER LEARNING PIPELINE                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PHASE 1 — FINE-TUNING (Python, offline)                                    ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  HuggingFace Hub / timm / torchvision                               │    ║
║  │       │                                                             │    ║
║  │       ▼                                                             │    ║
║  │  Pre-trained base model  ←──── your labeled dataset                 │    ║
║  │       │                                                             │    ║
║  │  Fine-tuning strategy:                                              │    ║
║  │    ├── Full fine-tune (transformers Trainer)                        │    ║
║  │    ├── LoRA / QLoRA  (PEFT → merge_and_unload)                      │    ║
║  │    ├── Head-only     (freeze backbone, train classifier)            │    ║
║  │    └── Contrastive   (sentence-transformers, SetFit)                │    ║
║  │       │                                                             │    ║
║  │       ▼                                                             │    ║
║  │  Saved fine-tuned model  (safetensors / .pt)                        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  PHASE 2 — EXPORT (Python, offline)                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  optimum-cli export onnx  (text/vision models)                      │    ║
║  │  OR  torch.onnx.export   (timm, torchvision, YOLO, OpenCLIP)        │    ║
║  │       │                                                             │    ║
║  │       ▼                                                             │    ║
║  │  model.onnx  (FP32, opset 17)                                       │    ║
║  │       │                                                             │    ║
║  │  ORTOptimizer  (O1–O3 graph fusion)                                 │    ║
║  │       │                                                             │    ║
║  │  ORTQuantizer  (optional INT8 dynamic quantization)                 │    ║
║  │       │                                                             │    ║
║  │       ▼                                                             │    ║
║  │  model_optimized_quantized.onnx   ← artifact committed to repo      │    ║
║  │  tokenizer.json                   ← for text models                 │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  PHASE 3 — SERVING (Rust / Axum, runtime)                                   ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                     │    ║
║  │  main()                                                             │    ║
║  │    ├── ort::init()            (once per process)                    │    ║
║  │    ├── Session::from_file()   (load ONNX model)                     │    ║
║  │    ├── Tokenizer::from_file() (load tokenizer.json)                 │    ║
║  │    ├── warm_up()              (3 dummy inferences)                  │    ║
║  │    └── Arc<AppState>  ──────► Axum Router                           │    ║
║  │                                                                     │    ║
║  │  POST /predict                                                      │    ║
║  │    ├── extract request body                                         │    ║
║  │    ├── tokenize (tokenizers crate)                                  │    ║
║  │    ├── spawn_blocking {                                             │    ║
║  │    │     build input tensors                                        │    ║
║  │    │     session.run(inputs)   ← ort (ONNX Runtime)                 │    ║
║  │    │     post-process outputs                                       │    ║
║  │    │   }                                                            │    ║
║  │    └── return JSON response                                         │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  HOT-SWAP (background task)                                                  ║
║  ┌────────────────────────────────────────────────┐                         ║
║  │  watch model.onnx for changes (file mtime)     │                         ║
║  │  load new Session                              │                         ║
║  │  *model_ref.write() = Arc::new(new_session)    │                         ║
║  └────────────────────────────────────────────────┘                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Python Fine-Tuning Toolchain

### transformers v5 + PEFT

**Repositories:**
- https://github.com/huggingface/transformers
- https://github.com/huggingface/peft

**Current versions:** `transformers==5.x` | `peft==0.18.x`

**transformers v5 key changes:**
- **PyTorch-only.** TensorFlow and Flax support are being sunset. All fine-tuning and export workflows are now PyTorch-first.
- ONNX export is maintained via the companion `optimum` library (not built into transformers directly).
- Deep collaboration with ONNX Runtime, llama.cpp, and MLX for deployment interoperability.

#### Full fine-tuning (small models — feasible on CPU)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    no_cuda=True,           # CPU training
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./distilbert_finetuned")
tokenizer.save_pretrained("./distilbert_finetuned")
```

#### LoRA fine-tuning with PEFT (recommended for larger models)

LoRA (Low-Rank Adaptation) injects trainable rank-decomposition matrices into transformer layers. Only ~0.1–1% of the total parameters are trained, making fine-tuning feasible even on modest hardware.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                          # rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],   # BERT attention projection layers
    bias="none",
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 592,900 || all params: 109,876,740 || trainable%: 0.54%

# ... train with Trainer or custom loop ...

# Merge LoRA weights back into the base model for clean ONNX export
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./bert_lora_merged")
tokenizer.save_pretrained("./bert_lora_merged")
```

**`merge_and_unload()` explained:** This folds the LoRA adapter matrices `W = W_0 + BA` back into the original weight matrices and removes all adapter scaffolding. The result is a standard HuggingFace model in its original architecture — indistinguishable from a fully fine-tuned model for export purposes. After merging, export with `optimum-cli` exactly as you would a full fine-tuned model.

**PEFT strategies and when to use each:**

| Strategy | Trainable params | Use case |
|---|---|---|
| Full fine-tune | 100% | Small models (DistilBERT, MiniLM) with enough data |
| LoRA (r=8) | ~0.5% | Large models; limited VRAM/RAM; continual learning |
| QLoRA | ~0.5% (in 4-bit base) | Very large models on constrained hardware |
| IA³ | ~0.1% | Few-shot scenarios; even fewer trainable params than LoRA |
| Head-only | Head params only | Limited labeled data; strong pre-trained backbone |
| Prefix tuning | ~0.1% | Sequence-to-sequence tasks |

---

### sentence-transformers

**Repository:** https://github.com/UKPLab/sentence-transformers
**PyPI:** https://pypi.org/project/sentence-transformers/
**Current version:** `5.2.x`

Provides a high-level API for training and fine-tuning bi-encoder embedding models (SBERT variants). Models output fixed-length dense vectors suitable for semantic search, clustering, and retrieval.

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load a pre-trained sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fine-tune with labeled pairs (cosine similarity loss)
train_examples = [
    InputExample(texts=["My dog is cute", "My dog is adorable"], label=0.9),
    InputExample(texts=["My dog is cute", "The stock market fell"], label=0.1),
    # ... more pairs ...
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
)
model.save("./my_embedding_model")
```

**For retrieval tasks:** Use `MultipleNegativesRankingLoss` instead — it treats other batch items as negatives and is significantly more effective for retrieval fine-tuning:

```python
from sentence_transformers import losses
train_loss = losses.MultipleNegativesRankingLoss(model)
```

---

### SetFit

**Repository:** https://github.com/huggingface/setfit
**PyPI:** https://pypi.org/project/setfit/
**Current version:** `1.1.x`

SetFit (Sentence Transformer Fine-Tuning) enables highly efficient few-shot text classification. It fine-tunes a sentence transformer using contrastive learning on small amounts of labeled data (8–64 examples per class), then fits a logistic regression head on the resulting embeddings.

```python
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

# As few as 8 examples per class
train_dataset = Dataset.from_dict({
    "text": ["example 1", "example 2", ...],
    "label": [0, 1, ...],
})

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
args = TrainingArguments(num_epochs=1, batch_size=16)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()
model.save_pretrained("./setfit_model")
```

**ONNX export:** SetFit's sentence transformer body can be exported with `optimum-cli` using `--task feature-extraction`. The sklearn logistic regression head is exported separately as a standalone ONNX model via `sklearn-onnx`. In Rust, run both in sequence: sentence transformer → embeddings → sklearn head → class probabilities.

---

### timm — PyTorch Image Models

**Repository:** https://github.com/huggingface/pytorch-image-models
**PyPI:** https://pypi.org/project/timm/
**Current version:** `1.0.x`

The largest collection of pre-trained vision models in PyTorch (1,200+ architectures, 700+ pre-trained weights). Supports ResNet, EfficientNet, ViT, ConvNeXt, Swin Transformer, MobileNetV4, and more.

```python
import timm, torch
from torch import nn

# 1. Load pre-trained backbone, remove its classification head
backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)

# 2. Freeze backbone weights (feature extraction mode)
for param in backbone.parameters():
    param.requires_grad = False

# 3. Add custom head
num_features = backbone.num_features  # 1280 for efficientnet_b0
model = nn.Sequential(backbone, nn.Linear(num_features, num_custom_classes))

# 4. Train only the head (fast, even on CPU)
optimizer = torch.optim.Adam(model[1].parameters(), lr=1e-3)
# ... training loop ...

# 5. ONNX export
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "efficientnet_b0_custom.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
)
```

timm also ships a dedicated export script `onnx_export.py` for batch export with validation.

---

### ultralytics — YOLOv8/v11

**Repository:** https://github.com/ultralytics/ultralytics
**PyPI:** https://pypi.org/project/ultralytics/
**Current version:** `8.4.x`

```python
from ultralytics import YOLO

# Fine-tune on custom detection dataset (COCO format)
model = YOLO("yolo11n.pt")       # start from nano model (2.6M params)
results = model.train(
    data="custom_dataset.yaml",
    epochs=50,
    imgsz=640,
    device="cpu",               # CPU training for small datasets/models
    batch=16,
)

# Export to ONNX (built-in, one line)
model.export(format="onnx", opset=17, simplify=True, dynamic=True)
# Produces: runs/detect/train/weights/best.onnx
```

**YOLO model size guide for CPU:**

| Model | Params | ONNX size | CPU latency (640px) |
|---|---|---|---|
| YOLO11n | 2.6M | ~12 MB | ~50–100 ms |
| YOLO11s | 9.4M | ~40 MB | ~100–200 ms |
| YOLOv8n | 3.2M | ~14 MB | ~60–120 ms |

---

### OpenCLIP

**Repository:** https://github.com/mlfoundations/open_clip
**PyPI:** https://pypi.org/project/open-clip-torch/
**Current version:** `2.30.x`

Fine-tune CLIP-style vision-language models for custom image-text alignment, zero-shot classification, or image search.

```python
import open_clip, torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

# Fine-tune with contrastive loss on (image, text) pairs
# ... training loop with InfoNCE loss ...

# ONNX export — export image and text encoders separately
model.eval()
dummy_image = torch.randn(1, 3, 224, 224)
dummy_text = torch.randint(0, 49408, (1, 77))

torch.onnx.export(
    model.visual, dummy_image, "clip_image_encoder.onnx",
    opset_version=17, input_names=["image"], output_names=["image_features"],
    dynamic_axes={"image": {0: "batch"}, "image_features": {0: "batch"}},
)
torch.onnx.export(
    model, dummy_text, "clip_text_encoder.onnx",
    opset_version=17, input_names=["text"], output_names=["text_features"],
    dynamic_axes={"text": {0: "batch"}, "text_features": {0: "batch"}},
)
```

---

## ONNX Export Pipeline

### optimum — The Recommended Export Tool

**Repository:** https://github.com/huggingface/optimum-onnx
**PyPI:** https://pypi.org/project/optimum/
**Current version:** `optimum[onnxruntime]==1.24.x` | `optimum-onnx==0.0.x` (newer split package)

`optimum` is the Hugging Face library for optimized inference. It wraps `torch.onnx.export` with:
- Automatic dynamic axes for batch and sequence dimensions.
- Correct input/output naming that matches the model architecture.
- Task-specific graph structure (the right number of outputs for classification vs embedding vs token classification).
- Post-export graph optimization via ONNX Runtime's optimizer.
- INT8 quantization with calibration.

**Export command (recommended for all HuggingFace models):**

```bash
pip install optimum[onnxruntime]

# Text classification
optimum-cli export onnx \
  --model ./distilbert_finetuned \
  --task text-classification \
  --opset 17 \
  ./distilbert_onnx/

# Feature extraction (sentence embeddings)
optimum-cli export onnx \
  --model ./my_embedding_model \
  --task feature-extraction \
  --opset 17 \
  ./embedding_onnx/

# Token classification (NER)
optimum-cli export onnx \
  --model ./ner_model \
  --task token-classification \
  --opset 17 \
  ./ner_onnx/

# Question answering
optimum-cli export onnx \
  --model ./qa_model \
  --task question-answering \
  --opset 17 \
  ./qa_onnx/
```

The output directory will contain:
- `model.onnx` — the exported model
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` — tokenizer files needed in Rust
- `config.json` — model config (useful for label mappings)

**For non-HuggingFace models (timm, torchvision, YOLO, OpenCLIP):** Use `torch.onnx.export` directly as shown in the sections above. `optimum-cli` only handles HuggingFace `transformers` models.

---

### LoRA Merge and Export

The workflow is always: **merge first, then export as a standard model**.

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Load base + adapter
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter_checkpoint")

# Merge: folds adapter matrices into base weights, removes adapter layers
merged = peft_model.merge_and_unload()
merged.save_pretrained("./merged_for_export")

# Now export exactly as a standard fine-tuned model
# optimum-cli export onnx --model ./merged_for_export --task text-classification ./onnx/
```

After merging, there are no traces of LoRA in the architecture. The ONNX export and Rust inference are identical to a fully fine-tuned model.

---

### Graph Optimization Levels

After export, apply ONNX Runtime's graph optimizer to fuse operations and improve CPU throughput:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

model = ORTModelForSequenceClassification.from_pretrained("./distilbert_onnx")
optimizer = ORTOptimizer.from_pretrained(model)
optimizer.optimize(
    save_dir="./distilbert_optimized",
    optimization_config=OptimizationConfig(
        optimization_level=2,               # O2: fuses attention, layer norm, GELU
        optimize_for_gpu=False,
        fp16=False,
    ),
)
```

| Level | What it does | Notes |
|---|---|---|
| O1 | Basic: constant folding, unused node elimination | Always safe; portable |
| O2 | Extended: fuses attention (`com.microsoft.Attention`), GELU, LayerNorm | ORT-specific ops; use `ort`, not `tract` |
| O3 | Layout optimizations, additional fusions | ORT-specific; measure before using |
| O4 | Mixed precision (FP16/BF16) | Only useful with GPU or AVX512 BF16 |

> **Portability note:** O2+ optimization generates Microsoft-specific ONNX custom operators (`com.microsoft.Attention`, `com.microsoft.FastGelu`). These work only in `ort` (Rust), not in `tract` or `rten`. For `tract`/`rten` compatibility, use O1 only.

---

### INT8 Quantization

Dynamic INT8 quantization reduces model size by ~4x and improves throughput on CPUs with VNNI extensions (Intel Cascade Lake+, most modern server CPUs):

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("./distilbert_optimized")

# Dynamic quantization — no calibration data needed
quant_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,
    per_channel=False,   # MUST be False for 3D weight tensors (transformer FFN layers)
)
quantizer.quantize(
    save_dir="./distilbert_quantized",
    quantization_config=quant_config,
)
```

**Known INT8 issues on CPU — read before using:**

| Issue | Description | Mitigation |
|---|---|---|
| Slower than FP32 on old CPUs | AVX512 VNNI not available on CPUs older than Intel Cascade Lake (2019) | Benchmark both; deploy FP32 if INT8 is slower |
| `per_channel=True` fails | 3D weight tensors in transformer FFN layers crash ORT quantizer | Always use `per_channel=False` |
| Fused attention (O2) + INT8 | `com.microsoft.Attention` custom ops may not quantize correctly | Quantize O1-optimized models, not O2 |
| Accuracy loss | INT8 reduces precision; significant for tasks requiring fine-grained confidence | Evaluate on held-out set before deploying |

---

### Model Size Reference

| Model | Task | Params | FP32 ONNX | INT8 ONNX | CPU latency (FP32) | CPU latency (INT8) |
|---|---|---|---|---|---|---|
| `prajjwal1/bert-tiny` | classification | 4.4M | ~17 MB | ~5 MB | ~2–5 ms | ~1–3 ms |
| `all-MiniLM-L6-v2` | embeddings | 22M | ~90 MB | ~23 MB | ~10–20 ms | ~5–12 ms |
| `distilbert-base-uncased` | classification | 66M | ~260 MB | ~65 MB | ~40–80 ms | ~20–45 ms |
| `bert-base-uncased` | classification | 110M | ~440 MB | ~110 MB | ~100–200 ms | ~50–100 ms |
| `MobileNetV3-Small` | image features | 2.5M | ~10 MB | ~4 MB | ~5–10 ms | ~3–6 ms |
| `EfficientNet-B0` | image classification | 5.3M | ~22 MB | ~7 MB | ~15–25 ms | ~8–15 ms |
| `ResNet-50` | image classification | 25M | ~100 MB | ~27 MB | ~40–80 ms | ~20–40 ms |
| `ViT-base-patch16` | image classification | 86M | ~350 MB | ~88 MB | ~150–300 ms | ~80–160 ms |
| YOLO11n | object detection | 2.6M | ~12 MB | — | ~50–100 ms | — |

> Latencies are single-sample, FP32, 4-thread ORT on a modern Intel server CPU (Cascade Lake / Ice Lake class). INT8 speedup requires VNNI support; exact values vary significantly by hardware.

---

## Rust Inference Libraries

### ort — Primary ONNX Runtime

**Repository:** https://github.com/pykeio/ort
**crates.io:** https://crates.io/crates/ort
**Current version:** `2.0.0-rc.11` (wraps ONNX Runtime v1.23.2)

[![Current Crates.io Version](https://img.shields.io/crates/v/ort.svg)](https://crates.io/crates/ort)

`ort` is the primary choice for serving ONNX transformer models in Rust. It wraps Microsoft's ONNX Runtime C++ library and provides first-class async Rust ergonomics.

**Key properties for transformer workloads:**
- `Session` is `Send + Sync` — safe to share via `Arc<Session>` across Axum handlers without a Mutex.
- Supports dynamic input shapes (variable sequence lengths) natively.
- Full `ai.onnx` and `ai.onnx.ml` operator coverage.
- O2-optimized models with `com.microsoft.Attention` custom ops work correctly.
- `ort::init()` must be called exactly once per process before any `Session` is created.

```toml
[dependencies]
ort = { version = "2.0.0-rc.11", features = ["load-dynamic"] }
tokenizers = "0.22"
ndarray = "0.16"
tokio = { version = "1", features = ["full"] }
axum = "0.8"
```

**Basic transformer inference pattern:**

```rust
use ort::{Session, SessionBuilder, GraphOptimizationLevel, inputs};
use tokenizers::Tokenizer;
use ndarray::{Array2, CowArray};

// --- Startup ---
ort::init().with_name("my-service").commit()?;

let session = SessionBuilder::new()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?   // use all available cores for single-request inference
    .commit_from_file("model.onnx")?;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// --- Per-request inference ---
let encoding = tokenizer.encode("Hello, world!", true)?;
let seq_len = encoding.len();

let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

let ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
let mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;

let outputs = session.run(inputs! {
    "input_ids"      => ids_array.view(),
    "attention_mask" => mask_array.view(),
}?)?;

let logits = outputs["logits"].try_extract_tensor::<f32>()?;
// Shape: [1, num_labels] — apply softmax for probabilities
```

---

### candle — Native HuggingFace Models

**Repository:** https://github.com/huggingface/candle
**crates.io:** https://crates.io/crates/candle-core
**Current version:** `0.9.1`

[![Current Crates.io Version](https://img.shields.io/crates/v/candle-core.svg)](https://crates.io/crates/candle-core)

`candle` is HuggingFace's pure Rust ML framework. Its key advantage over `ort` for transfer learning: it can load **safetensors weights directly from HuggingFace-format checkpoints** without any ONNX export step.

**Supported transformer architectures (natively implemented):**
BERT, DistilBERT, RoBERTa, GPT-2, LLaMA 2/3, Mistral, Phi, Gemma, T5, Whisper, CLIP, Stable Diffusion, ViT, Segment Anything, and more.

**When to use candle over ort:**
- You want to skip the ONNX export step entirely.
- You're using a model from the candle examples list and want pure Rust with zero C++ dependencies.
- You need WASM deployment.

**When to use ort over candle:**
- Your fine-tuned model is not in candle's supported architecture list.
- You need O2 graph optimizations (fused attention, GELU fusion) for maximum CPU throughput.
- You're using a quantized INT8 model.

**Loading a fine-tuned BERT model in candle:**

```toml
[dependencies]
candle-core = "0.9.1"
candle-nn = "0.9.1"
candle-transformers = "0.9.1"
hf-hub = "0.3"
tokenizers = "0.22"
```

```rust
use candle_core::{Device, DType};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

let device = Device::Cpu;

// Load fine-tuned model from local directory or HuggingFace Hub
let api = Api::new()?;
let repo = api.model("./my_finetuned_distilbert".to_string());

let config: Config = serde_json::from_reader(
    std::fs::File::open("./model/config.json")?
)?;
let weights = candle_core::safetensors::load("./model/model.safetensors", &device)?;

let mut vb = candle_nn::VarBuilder::from_tensors(weights, DType::F32, &device);
let model = BertModel::load(vb, &config)?;

// Run inference
let tokens = tokenizer.encode("Hello world", true)?;
let input_ids = candle_core::Tensor::new(
    tokens.get_ids().to_vec().as_slice(),
    &device,
)?.unsqueeze(0)?;   // shape [1, seq_len]

let output = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
// output shape: [1, seq_len, hidden_size] — apply pooling for embeddings
```

**LoRA with candle:** candle does not have native LoRA adapter loading. The recommended workflow remains `merge_and_unload()` in Python before exporting to safetensors or ONNX.

---

### fastembed-rs — Pre-Packaged Embeddings

**Repository:** https://github.com/Anush008/fastembed-rs
**crates.io:** https://crates.io/crates/fastembed
**Current version:** `4.x`

[![Current Crates.io Version](https://img.shields.io/crates/v/fastembed.svg)](https://crates.io/crates/fastembed)

`fastembed-rs` (by Qdrant) is a Rust library for fast text and image embedding generation. Internally uses `ort` + `tokenizers`. Ships pre-packaged ONNX models for the most common embedding models.

**Pre-packaged models:**
- `AllMiniLML6V2` / `AllMiniLML6V2Q` (quantized) — most common embedding model
- `BGESmallENV15` / `BGESmallENV15Q`
- `NomicEmbedTextV1.5`
- Multiple multilingual models
- Vision embedding models (`Qdrant/Unicom-ViT-B-32`)

**Drop-in usage (no custom model):**

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

let model = TextEmbedding::try_new(
    InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_show_download_progress(true)
)?;

let documents = vec!["query text", "document to search"];
let embeddings = model.embed(documents, None)?;  // Vec<Vec<f32>>
// Shape: [2, 384]
```

**Custom fine-tuned model support:** Pass your own ONNX model and tokenizer files:

```rust
use fastembed::{TextEmbedding, UserDefinedEmbeddingModel, TokenizerFiles, InitOptionsUserDefined};

let custom_model = TextEmbedding::try_new_from_user_defined(
    UserDefinedEmbeddingModel {
        onnx_file: std::fs::read("./my_finetuned_model.onnx")?,
        tokenizer_files: TokenizerFiles {
            tokenizer_file: std::fs::read("./tokenizer.json")?,
            config_file: std::fs::read("./config.json")?,
            special_tokens_map_file: std::fs::read("./special_tokens_map.json")?,
            tokenizer_config_file: std::fs::read("./tokenizer_config.json")?,
        },
    },
    InitOptionsUserDefined::default(),
)?;
```

`TextEmbedding` is `Send + Sync`. Wrap in `Arc<TextEmbedding>` for Axum state sharing.

---

### rust-bert — Pre-Built NLP Pipelines

**Repository:** https://github.com/guillaume-be/rust-bert
**crates.io:** https://crates.io/crates/rust-bert
**Current version:** `0.23.0`

[![Current Crates.io Version](https://img.shields.io/crates/v/rust-bert.svg)](https://crates.io/crates/rust-bert)

`rust-bert` provides complete, ready-to-use NLP pipelines in Rust. Supports both `tch-rs` (TorchScript/libtorch) and ONNX Runtime backends. For CPU servers, use the ONNX backend — it eliminates the ~500 MB libtorch dependency.

**Supported models:** BERT, DistilBERT, RoBERTa, ALBERT, DeBERTa, GPT-2, GPT-Neo, LLaMA, T5, BART, MBART, MarianMT, PEGASUS, Electra.

**Supported pipelines:** Sequence classification, zero-shot classification, token classification (NER), question answering, text generation, summarization, translation, sentence embeddings.

**Loading a fine-tuned model:**

```rust
use rust_bert::pipelines::sequence_classification::{
    SequenceClassificationConfig, SequenceClassificationModel,
};
use rust_bert::resources::{LocalResource, ResourceProvider};
use rust_bert::RustBertError;
use std::path::PathBuf;

let config_resource  = Box::new(LocalResource { local_path: PathBuf::from("./model/config.json") });
let vocab_resource   = Box::new(LocalResource { local_path: PathBuf::from("./model/vocab.txt") });
let weights_resource = Box::new(LocalResource { local_path: PathBuf::from("./model/model.onnx") });

let config = SequenceClassificationConfig {
    model_type: rust_bert::ModelType::DistilBert,
    model_resource: weights_resource,
    config_resource,
    vocab_resource,
    ..Default::default()
};

let model = SequenceClassificationModel::new(config)?;
let outputs = model.predict(&["Great product!", "Terrible experience"])?;
```

**Axum note:** rust-bert models are synchronous (not async-native). Wrap in `tokio::task::spawn_blocking` for Axum handlers.

---

### tokenizers — Text Tokenization in Rust

**Repository:** https://github.com/huggingface/tokenizers
**crates.io:** https://crates.io/crates/tokenizers
**Current version:** `0.22.1`

[![Current Crates.io Version](https://img.shields.io/crates/v/tokenizers.svg)](https://crates.io/crates/tokenizers)

The canonical HuggingFace tokenizer library, written in Rust as its primary implementation (with Python bindings on top). Supports BPE, WordPiece, Unigram (SentencePiece), and WordLevel. Processes 1 GB of text in under 20 seconds on a single CPU.

```rust
use tokenizers::{Tokenizer, PaddingParams, PaddingStrategy, TruncationParams};

// Load from tokenizer.json (exported alongside the ONNX model)
let mut tokenizer = Tokenizer::from_file("./model/tokenizer.json")?;

// Configure padding and truncation
tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::BatchLongest,
    pad_id: 0,
    pad_token: "[PAD]".into(),
    ..Default::default()
}));
tokenizer.with_truncation(Some(TruncationParams {
    max_length: 512,
    ..Default::default()
}))?;

// Encode a batch (handles padding automatically)
let encodings = tokenizer.encode_batch(
    vec!["Hello world", "Another sentence to classify"],
    true,  // add special tokens ([CLS], [SEP])
)?;

// Extract tensors for ORT
let input_ids: Vec<Vec<i64>> = encodings.iter()
    .map(|e| e.get_ids().iter().map(|&x| x as i64).collect())
    .collect();
let attention_masks: Vec<Vec<i64>> = encodings.iter()
    .map(|e| e.get_attention_mask().iter().map(|&x| x as i64).collect())
    .collect();
let token_type_ids: Vec<Vec<i64>> = encodings.iter()
    .map(|e| e.get_type_ids().iter().map(|&x| x as i64).collect())
    .collect();
```

For **NER post-processing**, use `encoding.get_word_ids()` to map sub-word token predictions back to original word-level labels (essential for B-/I- span extraction).

For **GPT-family models**, use **tiktoken-rs** (crates.io: `tiktoken-rs`, version `0.9.1`) instead:

```rust
use tiktoken_rs::cl100k_base;  // GPT-4, GPT-3.5-turbo
let bpe = cl100k_base()?;
let tokens = bpe.encode_with_special_tokens("Hello, world!");
```

---

### tch-rs + TorchScript (Legacy Path)

**Repository:** https://github.com/LaurentMazare/tch-rs
**crates.io:** https://crates.io/crates/tch
**Current version:** `0.23.0` (requires libtorch v2.10.0)

[![Current Crates.io Version](https://img.shields.io/crates/v/tch.svg)](https://crates.io/crates/tch)

The TorchScript path exports a fine-tuned model as a `.pt` file and loads it in Rust via libtorch. **This is the legacy approach and is not recommended for new projects.** Prefer ONNX via `ort`.

**Limitations:**
- libtorch is a ~500 MB binary dependency — a significant Docker image and deployment burden.
- `torch.jit.trace` only records one execution path; transformer models with conditional branches (e.g., `if use_cache`) require `torch.jit.script`, which imposes strict type annotations.
- Variable-length attention masks can cause tracing artifacts.

If you have an existing tch-rs integration:

```python
# Python: export fine-tuned transformer as TorchScript
model.eval()
example = (
    torch.randint(0, 1000, (1, 64)),  # input_ids
    torch.ones(1, 64, dtype=torch.long),  # attention_mask
)
traced = torch.jit.trace(model, example)
traced.save("model.pt")
```

```rust
use tch::{CModule, Tensor, Kind, Device};
let model = CModule::load("model.pt")?;
let ids   = Tensor::zeros(&[1, 64], (Kind::Int64, Device::Cpu));
let mask  = Tensor::ones(&[1, 64], (Kind::Int64, Device::Cpu));
let out   = model.forward_ts(&[ids, mask])?;
```

---

## Use Case Walkthroughs

### 1. Text Classification

**Task:** Sentiment analysis, intent detection, topic classification.
**Best model for CPU:** `distilbert-base-uncased` (66M params, ~40–80 ms/sample FP32) or `all-MiniLM-L6-v2` with a classification head (22M params, ~10–20 ms/sample).

**Step 1 — Fine-tune:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# ... Trainer.train() ...
model.save_pretrained("./clf_model")
tokenizer.save_pretrained("./clf_model")
```

**Step 2 — Export and quantize:**
```bash
optimum-cli export onnx --model ./clf_model --task text-classification --opset 17 ./clf_onnx/
```
```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("./clf_onnx")
quantizer.quantize(
    save_dir="./clf_quantized",
    quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False),
)
```

**Step 3 — Serve in Rust:**
```rust
use ort::{Session, SessionBuilder, GraphOptimizationLevel, inputs};
use tokenizers::Tokenizer;
use ndarray::Array2;

fn classify(session: &Session, tokenizer: &Tokenizer, text: &str) -> Vec<f32> {
    let enc = tokenizer.encode(text, true).unwrap();
    let seq = enc.len();

    let ids  = Array2::from_shape_vec((1, seq),
        enc.get_ids().iter().map(|&x| x as i64).collect()).unwrap();
    let mask = Array2::from_shape_vec((1, seq),
        enc.get_attention_mask().iter().map(|&x| x as i64).collect()).unwrap();

    let outputs = session.run(inputs! {
        "input_ids"      => ids.view(),
        "attention_mask" => mask.view(),
    }.unwrap()).unwrap();

    let logits = outputs["logits"].try_extract_tensor::<f32>().unwrap();
    // Apply softmax
    let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}
```

---

### 2. Sentence Embeddings & Semantic Search

**Task:** Dense retrieval, semantic similarity, clustering.
**Best model for CPU:** `all-MiniLM-L6-v2` (22M params, 384-dim embeddings, ~10–20 ms/sample).

**Step 1 — Fine-tune:**
```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("all-MiniLM-L6-v2")
train_examples = [InputExample(texts=["query", "relevant doc"], label=1.0), ...]
train_loader = DataLoader(train_examples, batch_size=16, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_loader, train_loss)], epochs=2)
model.save("./embedding_model")
```

**Step 2 — Export:**
```bash
optimum-cli export onnx \
  --model ./embedding_model \
  --task feature-extraction \
  --opset 17 \
  ./embedding_onnx/
```

**Step 3 — Rust: mean pooling + L2 normalize:**
```rust
fn mean_pool_and_normalize(
    last_hidden: &ndarray::ArrayView3<f32>,  // [batch, seq, hidden]
    attention_mask: &[i64],
) -> Vec<f32> {
    let hidden_size = last_hidden.shape()[2];
    let mut pooled = vec![0.0f32; hidden_size];
    let mut count = 0.0f32;

    for (i, &mask) in attention_mask.iter().enumerate() {
        if mask == 1 {
            count += 1.0;
            for j in 0..hidden_size {
                pooled[j] += last_hidden[[0, i, j]];
            }
        }
    }
    pooled.iter_mut().for_each(|v| *v /= count);

    // L2 normalize — after this, cosine similarity == dot product
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    pooled.iter_mut().for_each(|v| *v /= norm);
    pooled
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

> **Alternative: fastembed-rs** handles all of the above (ONNX loading, tokenization, mean pooling, normalization) in 3 lines. Use it when you want zero boilerplate and `all-MiniLM-L6-v2` is sufficient.

---

### 3. Image Classification

**Task:** Multi-class image classification on custom categories.
**Best model for CPU:** `EfficientNet-B0` (5.3M params, ~15–25 ms/image) or `MobileNetV3-Small` (2.5M params, ~5–10 ms/image).

**Step 1 — Fine-tune with timm (head-only, fast):**
```python
import timm, torch
from torch import nn, optim
from torch.utils.data import DataLoader

backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
for param in backbone.parameters():
    param.requires_grad = False

model = nn.Sequential(backbone, nn.Linear(backbone.num_features, num_classes))
optimizer = optim.Adam(model[1].parameters(), lr=1e-3)
# ... training loop on DataLoader with torchvision transforms ...

model.eval()
torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "efficientnet_custom.onnx",
    opset_version=17, input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
)
```

**Step 2 — Rust image preprocessing:**
```rust
// Required Cargo deps: image = "0.25", ndarray = "0.16"
use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array4, s};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];  // ImageNet mean
const STD:  [f32; 3] = [0.229, 0.224, 0.225];  // ImageNet std

fn preprocess_image(img: DynamicImage) -> Array4<f32> {
    let img = img.resize_exact(224, 224, FilterType::Lanczos3).into_rgb8();
    let mut tensor = Array4::zeros((1, 3, 224, 224));
    for y in 0..224 {
        for x in 0..224 {
            let pixel = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (val - MEAN[c]) / STD[c];
            }
        }
    }
    tensor
}
```

---

### 4. Named Entity Recognition (NER)

**Task:** Extract person names, organizations, locations from text.
**Pre-built ONNX model available:** `optimum/bert-base-NER` on HuggingFace Hub (no fine-tuning needed for standard PER/ORG/LOC/MISC).

**Step 1 — Fine-tune on custom entity types:**
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

label_list = ["O", "B-PRODUCT", "I-PRODUCT", "B-VERSION", "I-VERSION"]
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_list),
    id2label={i: l for i, l in enumerate(label_list)},
    label2id={l: i for i, l in enumerate(label_list)},
)
# ... fine-tune with Trainer ...
model.save_pretrained("./ner_model")
```

**Step 2 — Export:**
```bash
optimum-cli export onnx \
  --model ./ner_model \
  --task token-classification \
  --opset 17 \
  ./ner_onnx/
```

**Step 3 — Rust: token-to-word mapping:**
```rust
// ORT output shape: [1, seq_len, num_labels]
// 1. Get argmax label for each token
// 2. Map sub-word token indices back to word indices via get_word_ids()
// 3. For each word, take the label of its first sub-word token
// 4. Group consecutive B-/I- spans into entity objects

let encodings = tokenizer.encode(text, true)?;
let word_ids: Vec<Option<u32>> = encodings.get_word_ids().to_vec();

// After running ORT session, predicted_labels: Vec<usize> (argmax per token)
let mut word_labels: std::collections::HashMap<u32, usize> = Default::default();
for (token_idx, word_id) in word_ids.iter().enumerate() {
    if let Some(wid) = word_id {
        // First sub-word token of each word wins
        word_labels.entry(*wid).or_insert(predicted_labels[token_idx]);
    }
}
```

---

### 5. Feature Extraction (Frozen Backbone)

No fine-tuning needed — export a pre-trained model as a feature extractor and use it as a fixed embedding function.

```bash
# Text: all-MiniLM-L6-v2 embeddings (384-dim, 22M params)
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --task feature-extraction \
  --opset 17 \
  ./minilm_onnx/

# Or use fastembed-rs in Rust — zero Python setup needed for standard models
```

```python
# Vision: MobileNetV3-Small features (576-dim, 2.5M params)
import torchvision.models as models, torch

model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()   # Remove classification head
model.eval()
torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "mobilenet_features.onnx",
    opset_version=17, input_names=["image"], output_names=["features"],
    dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
)
```

---

## Axum Integration Patterns

### Full AppState Setup with Warm-Up

```rust
use axum::{Router, routing::post, extract::State, Json};
use ort::{Session, SessionBuilder, GraphOptimizationLevel};
use tokenizers::Tokenizer;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    session: Arc<Session>,
    tokenizer: Arc<Tokenizer>,
    labels: Arc<Vec<String>>,  // id2label mapping from config.json
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize ORT environment once per process
    ort::init()
        .with_name("inference-service")
        .commit()?;

    // 2. Load tokenizer
    let tokenizer = Arc::new(Tokenizer::from_file("./model/tokenizer.json")?);

    // 3. Load ONNX session with graph optimization
    let session = Arc::new(
        SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?     // intra-op parallelism within a single inference
            .with_inter_threads(1)?     // one inference at a time (Axum handles concurrency)
            .commit_from_file("./model/model.onnx")?
    );

    // 4. Warm-up: run 3 dummy inferences to initialize ORT's thread pool and allocators
    {
        use ort::inputs;
        use ndarray::Array2;
        for _ in 0..3 {
            let dummy_ids  = Array2::<i64>::zeros((1, 32));
            let dummy_mask = Array2::<i64>::ones((1, 32));
            let _ = session.run(inputs! {
                "input_ids"      => dummy_ids.view(),
                "attention_mask" => dummy_mask.view(),
            }?)?;
        }
    }
    tracing::info!("Model warm-up complete");

    // 5. Build and start server
    let state = AppState {
        session,
        tokenizer,
        labels: Arc::new(vec!["negative".into(), "neutral".into(), "positive".into()]),
    };

    let app = Router::new()
        .route("/classify", post(classify_handler))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("Listening on :3000");
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(serde::Deserialize)]
struct ClassifyRequest { text: String }

#[derive(serde::Serialize)]
struct ClassifyResponse { label: String, score: f32, all_scores: Vec<(String, f32)> }

async fn classify_handler(
    State(state): State<AppState>,
    Json(req): Json<ClassifyRequest>,
) -> Json<ClassifyResponse> {
    let session   = state.session.clone();
    let tokenizer = state.tokenizer.clone();
    let labels    = state.labels.clone();

    // CPU-bound inference: offload to Tokio's blocking thread pool
    let result = tokio::task::spawn_blocking(move || {
        use ort::inputs;
        use ndarray::Array2;

        let enc = tokenizer.encode(req.text.as_str(), true).unwrap();
        let n = enc.len();

        let ids  = Array2::from_shape_vec((1, n),
            enc.get_ids().iter().map(|&x| x as i64).collect()).unwrap();
        let mask = Array2::from_shape_vec((1, n),
            enc.get_attention_mask().iter().map(|&x| x as i64).collect()).unwrap();

        let outputs = session.run(inputs! {
            "input_ids"      => ids.view(),
            "attention_mask" => mask.view(),
        }.unwrap()).unwrap();

        let logits = outputs["logits"].try_extract_tensor::<f32>().unwrap();
        let logits: Vec<f32> = logits.iter().copied().collect();

        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

        let best = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

        ClassifyResponse {
            label: labels[best.0].clone(),
            score: *best.1,
            all_scores: labels.iter().cloned().zip(probs).collect(),
        }
    })
    .await
    .unwrap();

    Json(result)
}
```

### Model Hot-Swap for Continuous Retraining

When models are periodically retrained and a new ONNX file is dropped onto the server:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use ort::Session;

type SharedSession = Arc<RwLock<Arc<Session>>>;

async fn model_watcher(session_ref: SharedSession, model_path: &'static str) {
    let mut last_modified = std::time::SystemTime::UNIX_EPOCH;
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        let Ok(meta) = std::fs::metadata(model_path) else { continue };
        let Ok(modified) = meta.modified() else { continue };

        if modified > last_modified {
            match SessionBuilder::new()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.commit_from_file(model_path))
            {
                Ok(new_session) => {
                    let new_arc = Arc::new(new_session);
                    // Warm up before swapping
                    // ... warmup runs ...
                    *session_ref.write().await = new_arc;
                    last_modified = modified;
                    tracing::info!("Model hot-swapped from {}", model_path);
                }
                Err(e) => tracing::error!("Failed to load new model: {}", e),
            }
        }
    }
}

// In handlers:
async fn handler(State(session_ref): State<SharedSession>, ...) {
    let session = session_ref.read().await.clone();  // Arc clone — near-zero cost
    // session has its own reference; the watcher can swap the inner Arc safely
    let _ = session.run(...);
}
```

---

## Practical Gotchas

### Dynamic Shapes
`optimum-cli` automatically sets dynamic axes on batch and sequence dimensions. In `ort`, you pass tensors of any valid shape — no special registration is needed. If you encounter shape inference errors with O3 optimized models, drop to O1:
```rust
.with_optimization_level(GraphOptimizationLevel::Level1)?
```

### Opset Version Requirements

| Transformer feature | Minimum opset |
|---|---|
| Standard attention, FFN, embeddings | opset 11 |
| Modern cast/reshape ops | opset 13 |
| `LayerNormalization` as standard op | **opset 17** (recommended minimum) |
| Grouped Query Attention (GQA) | opset 18 |
| Fused ORT custom ops (O2+) | ORT-specific (no opset restriction) |

Always export with `--opset 17` for models fine-tuned in 2025/2026.

### ORT Session Thread Safety
`Session` is `Send + Sync`. Wrap in `Arc<Session>` — no `Mutex` needed. Concurrent `session.run()` calls from multiple Axum handlers are safe; ORT allocates separate activation buffers per call internally.

### Memory Usage
ORT memory-maps the `.onnx` file. RSS (resident memory) grows as inference exercises different model parts:
- DistilBERT INT8 (~65 MB ONNX) → ~80–120 MB RSS after warm-up
- BERT-base FP32 (~440 MB ONNX) → ~450–550 MB RSS after warm-up

For a CPU-only server with multiple loaded models, budget memory carefully. Consider INT8 quantization to reduce model footprint 4×.

### `ort::init()` Must Be Called Once
```rust
// In main(), before creating any Session:
ort::init().with_name("service").commit()?;
```
Calling it multiple times or failing to call it before `Session::builder()` causes a panic or undefined behavior.

### INT8 Slowdown on Non-VNNI CPUs
INT8 dynamic quantization is faster only on CPUs with AVX512 VNNI (Intel Cascade Lake+ 2019, Ice Lake 2019, or AMD Zen4+). On older hardware or CPUs without VNNI, INT8 inference is **slower** than FP32. Always benchmark both before deploying quantized models.

---

## Decision Matrix

| Use case | Fine-tune with (Python) | Export | Serve in Rust |
|---|---|---|---|
| **Text classification, fastest CPU** | `transformers` + LoRA on `distilbert` | `optimum-cli --task text-classification --opset 17` | `ort` + `tokenizers` |
| **Text classification, minimal model** | `transformers` + full fine-tune on `bert-tiny` | `optimum-cli` | `ort` + `tokenizers` |
| **Sentence embeddings, custom domain** | `sentence-transformers` contrastive | `optimum-cli --task feature-extraction` | `ort` + mean pooling OR `fastembed-rs` custom model |
| **Sentence embeddings, off-the-shelf** | None (use pre-trained) | None | `fastembed-rs` (pre-packaged `all-MiniLM-L6-v2`) |
| **Few-shot text classification** | `SetFit` | `optimum-cli` (transformer body) + `sklearn-onnx` (head) | `ort` (two sessions in sequence) |
| **Image classification, fast CPU** | `timm` EfficientNet-B0, head-only | `torch.onnx.export` | `ort` + `image` crate |
| **Image classification, minimal** | `timm` MobileNetV3-Small, head-only | `torch.onnx.export` | `ort` + `image` crate |
| **NER (standard entities)** | None — use `optimum/bert-base-NER` | None (pre-exported ONNX on HF Hub) | `ort` + `tokenizers` + word_ids |
| **NER (custom entity types)** | `transformers` BERT token classification | `optimum-cli --task token-classification` | `ort` + `tokenizers` |
| **Object detection** | `ultralytics` YOLO11n | `model.export(format='onnx')` | `ort` + `image` crate |
| **Pre-built NLP pipeline, no boilerplate** | Fine-tune in Python | ONNX export | `rust-bert` (ONNX backend) |
| **No ONNX, native weights** | Fine-tune with HuggingFace | Save safetensors | `candle` (supported architectures only) |
| **LoRA fine-tuned model** | `peft` LoRA → `merge_and_unload()` | `optimum-cli` (merged model) | `ort` (same as full fine-tuned) |
| **Vision-language search (CLIP)** | `open_clip` contrastive | `torch.onnx.export` (two encoders) | `ort` (two sessions) |
