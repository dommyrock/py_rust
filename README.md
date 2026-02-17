# Rust Libraries for Python Interop, ONNX, and ML Workflows

A curated overview of key Rust libraries for interfacing with Python, running ONNX models, and building ML pipelines — covering use cases from Python extension authoring to high-performance production inference.

> **Last updated:** February 17, 2026

---

## Table of Contents

- [PyO3](#pyo3) — Rust ↔ Python interop
- [Monty](#monty-pydanticmonty) — Sandboxed Python interpreter for AI agents
- [ort](#ort-pykeioort) — ONNX Runtime bindings for Rust
- [tch-rs](#tch-rs-laurentmazaretch-rs) — PyTorch libtorch bindings
- [Additional Projects](#additional-projects)
  - [candle](#candle--huggingfacecandle)
  - [burn](#burn--tracel-aiburn)
  - [tract](#tract--sonostracti)
  - [tokenizers](#tokenizers--huggingfacetokenizers)
  - [linfa](#linfa--rust-mllinfa)

---

## PyO3

**Repository:** https://github.com/pyo3/pyo3
**Current version:** `0.28.1` | **Min Rust version:** 1.83
**License:** Apache-2.0 / MIT

PyO3 is the de facto standard for Rust/Python interoperability. It provides bidirectional bindings — Rust can expose native extension modules callable from Python, and Python can be embedded and driven from within a Rust binary.

### Strong Points

- **Macro-driven ergonomics** — `#[pymodule]`, `#[pyfunction]`, `#[pyclass]` attributes eliminate boilerplate when exposing Rust to Python.
- **Bidirectional interop** — both directions (Rust calling Python and Python calling Rust) are first-class citizens.
- **Free-threaded Python support** — v0.28.x supports CPython's experimental no-GIL mode (3.14t) opt-out rather than opt-in, signaling forward compatibility.
- **Stable ABI (`abi3`)** — build once, run on multiple Python versions without recompilation.
- **Maturin integration** — the companion [maturin](https://github.com/PyO3/maturin) tool handles building and publishing Rust-based Python packages to PyPI with minimal config.
- **Performance-focused API** — vectorcall dispatch, zero-cost GIL token access, and an optional reference pool disable flag for hot paths.

### Main Use Cases

- **Accelerating Python hot paths** — drop-in Rust implementations of CPU-bound functions (parsing, encoding, numerical work).
- **High-performance data engineering** — powering libraries like [Polars](https://github.com/pola-rs/polars) (DataFrames) and [connector-x](https://github.com/sfu-db/connector-x) (DB-to-DataFrame loading).
- **NLP tokenization** — [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) and [tiktoken](https://github.com/openai/tiktoken) are Rust cores exposed via PyO3.
- **Pydantic v2** — `pydantic-core` is built on PyO3 and delivered a **5× speedup** over Pydantic v1.
- **Embedding Python in Rust** — plugin and scripting systems where a Rust application hosts a Python interpreter.

### Notable Projects Using PyO3

| Project | Description |
|---|---|
| **Polars** | Fast multi-threaded DataFrame library |
| **pydantic-core** | Core validation engine of Pydantic v2 |
| **cryptography** | Widely-used Python crypto package |
| **orjson** | Fastest Python JSON library |
| **tokenizers** | HuggingFace NLP tokenizer library |
| **tiktoken** | OpenAI's fast BPE tokenizer |

---

## Monty (`pydantic/monty`)

**Repository:** https://github.com/pydantic/monty
**Current version:** `0.0.5` (February 2026) | **Status:** Experimental
**License:** MIT

Monty is a minimal, secure Python interpreter written in Rust, purpose-built for use by AI agents. It solves a critical tension in agentic AI: LLMs produce better, more efficient outputs when they can write and execute Python code — but running arbitrary LLM-generated code on a host system is a security risk, and container-based sandboxes (Docker, gVisor) add 100–500 ms of latency.

Monty's answer: **interpreter-level sandboxing with sub-microsecond startup.**

### Strong Points

- **Extreme startup speed** — under 1 µs to start an interpreter instance (vs. hundreds of milliseconds for containers).
- **Deny-by-default security model** — dangerous built-ins (`open`, `eval`, `exec`, `__import__`) are absent; `os`/`sys` return stubs; no filesystem, network, or environment access.
- **Configurable resource limits** — memory usage, allocation counts, stack depth, and execution time are all tracked and enforceable.
- **Start/Resume execution** — interpreter state can be serialized mid-execution (e.g., when a host function is called) and resumed with the result, enabling durable agentic workflows.
- **Multi-target distribution** — available as a Rust crate, a Python package (`pydantic-monty` on PyPI), JavaScript/TypeScript bindings, and WebAssembly for browser execution.
- **Ruff parser integration** — uses `ruff_python_parser` internally for battle-tested parsing.

### Main Use Cases

- **AI agent "Code Mode"** — instead of calling individual tools and sending JSON results back to the LLM, agents write short Python scripts that chain multiple tool calls and process results. Planned integration with [PydanticAI](https://ai.pydantic.dev/).
- **LLM-guided sandboxed execution** — when the interpreter returns an error for unsupported syntax, the LLM naturally rewrites code to stay within supported constraints.
- **Serverless / edge AI pipelines** — negligible cold-start cost makes it viable in latency-sensitive inference paths.

### Supported Python Subset (v0.0.5)

Core expressions, variables, functions, loops, conditionals, list comprehensions, string operations, recursion, and basic async. Class definitions and `match` statements are listed as "coming soon."

> **Note:** Monty is pre-1.0 and does not yet support third-party packages (NumPy, Pydantic, etc.).

---

## ort (`pykeio/ort`)

**Repository:** https://github.com/pykeio/ort
**Current version:** `2.0.0-rc.11` (wraps ONNX Runtime v1.23.2)
**License:** Apache-2.0 / MIT

`ort` is an ergonomic Rust wrapper around Microsoft's ONNX Runtime (C++ library). It is the primary choice for high-performance ONNX inference in Rust, supporting models originally trained in PyTorch, TensorFlow, Keras, scikit-learn, and PaddlePaddle.

### Strong Points

- **Full execution provider support** — hardware acceleration via CUDA, TensorRT, ROCm, DirectML, CoreML, OpenVINO, QNN, and WebGPU, all configurable via Cargo feature flags with automatic CPU fallback.
- **Zero-copy tensors** — `TensorRef::from_array_view` avoids unnecessary allocations on the inference hot path.
- **I/O Binding** — controls where inputs and outputs reside in memory, eliminating host↔device copies for chained GPU operations.
- **Async inference** — first-class async support for non-blocking inference in async Rust runtimes.
- **Alternative pure-Rust backends** — can swap ONNX Runtime for `tract` or `candle` backends when portability matters more than peak performance (e.g., WASM targets).
- **Model Editor API** — programmatic ONNX model creation and editing without leaving Rust.

### Supported Execution Providers

| Provider | Target |
|---|---|
| CUDA | NVIDIA GPUs |
| TensorRT | NVIDIA GPUs (preferred over CUDA) |
| ROCm | AMD GPUs |
| DirectML | Windows (any GPU via Direct3D) |
| CoreML | Apple Silicon / iOS / Neural Engine |
| OpenVINO | Intel CPUs, GPUs, VPUs |
| QNN | Qualcomm NPUs (mobile/edge) |
| WebGPU / WebNN | Browser |
| CPU | Universal fallback |

### Main Use Cases

- **Production inference services** — Hugging Face [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) uses `ort` for high-throughput embedding generation.
- **On-device / edge deployment** — no Python runtime required; small binary for resource-constrained environments.
- **Semantic search** — [Bloop](https://github.com/BloopAI/bloop) uses `ort` for AI-powered code search.
- **Content classification** — [Google Magika](https://github.com/google/magika) uses `ort` for file type detection at scale.
- **Database ML queries** — [SurrealDB](https://github.com/surrealdb/surrealdb) embeds `ort` to run ML models inside database query language.

---

## tch-rs (`LaurentMazare/tch-rs`)

**Repository:** https://github.com/LaurentMazare/tch-rs
**Current version:** `0.23.0` (requires libtorch v2.10.0)
**License:** Apache-2.0 / MIT

`tch-rs` provides safe, idiomatic Rust bindings to the PyTorch C++ API (libtorch), staying as close as possible to the original C++ API. It is the most direct path to using the full PyTorch ecosystem from Rust.

### Strong Points

- **Full autograd support** — gradient-based training works natively via `nn::VarStore`, which tracks all trainable parameters and their gradients.
- **TorchScript / JIT model loading** — export any Python/PyTorch model with `torch.jit.trace()` or `torch.jit.script()`, then load and run in Rust via `tch::CModule::load()` — no Python runtime required at inference time.
- **Complete neural network API** — `nn` module with linear layers, convolutions, LSTMs, and built-in optimizers (SGD, Adam).
- **SafeTensors support** — load and export model weights in HuggingFace's safe serialization format.
- **CUDA acceleration** — full GPU training and inference on NVIDIA hardware.
- **Flexible installation** — can reuse an existing Python PyTorch installation (`LIBTORCH_USE_PYTORCH=1`), auto-download libtorch, or link statically.

### Main Use Cases

- **Inference/deployment without Python** — the canonical workflow: train in Python, export as TorchScript `.pt`, deploy in a Rust binary.
- **Full training in Rust** — the included `min-gpt` example demonstrates GPT training from scratch entirely in Rust.
- **NLP transformer pipelines** — powers [rust-bert](https://github.com/guillaume-be/rust-bert), which provides ready-to-use BERT, RoBERTa, GPT-2, BART, DistilBERT and other transformer pipelines.
- **Image classification** — ResNet-18/34 examples included; JIT example classifies ImageNet images.

### When to Consider Alternatives

- **Binary size is critical** — libtorch is 500 MB+ as a shared library.
- **WASM or embedded targets** — libtorch does not support these; consider `candle` or `tract` instead.
- **Apple Silicon (MPS)** — support is limited and less documented than CUDA.

---

## Additional Projects

---

### candle — `huggingface/candle`

**Repository:** https://github.com/huggingface/candle
**Current version:** `0.9.1` | **License:** Apache-2.0 / MIT

Candle is HuggingFace's minimalist, PyTorch-inspired ML framework for Rust. It is designed for production inference with zero Python runtime and small binary footprint.

#### Strong Points

- **Pure Rust with GPU support** — CUDA and Metal backends; no C++ FFI dependencies for core operation.
- **WASM target** — models can run directly in the browser.
- **Extensive model zoo** — LLaMA/2, Mistral, Mixtral, Falcon, Phi, Whisper, Stable Diffusion, BERT, T5, YOLO, SAM, and more are all implemented and ready to use.
- **Native GGUF/safetensors loading** — directly loads quantized GGUF weights (used by llama.cpp) and HuggingFace safetensors without conversion.
- **HuggingFace Hub integration** — models can be pulled directly from the Hub.

#### Main Use Cases

- Deploying LLMs and vision/audio transformer models in Rust services without any Python dependency.
- Running inference on edge devices or in the browser via WASM.
- Building a complete Python-free inference stack alongside `tokenizers` and `safetensors`.

---

### burn — `tracel-ai/burn`

**Repository:** https://github.com/tracel-ai/burn
**Current version:** `0.20.0` | **License:** Apache-2.0 / MIT

Burn is the most complete Rust-native deep learning framework, covering both training and inference with a pluggable backend architecture.

#### Strong Points

- **First-class training support** — unlike candle (experimental training), burn treats training as a primary use case, with a full training dashboard for monitoring runs.
- **Pluggable backends** — `burn-wgpu` (cross-platform GPU), `burn-cuda`, `burn-ndarray` (CPU), `burn-candle`, and `burn-tch` can be swapped without changing model code.
- **ONNX import** — `burn-import` converts ONNX models to native Burn Rust code, runnable on any backend without a C++ runtime.
- **CubeCL GPU kernels** — custom MATMUL kernels that have been benchmarked competitively against NVIDIA cuBLAS.
- **Runs everywhere** — from embedded CPUs and WebAssembly to large multi-GPU clusters, using the same model definition.

#### Main Use Cases

- End-to-end deep learning pipelines written entirely in Rust (no Python anywhere in the stack).
- Hardware-agnostic model deployment — swap backends between CPU, GPU, and WASM without code changes.
- Converting and deploying ONNX models natively without depending on Microsoft's ONNX Runtime C++ library.

---

### tract — `sonos/tract`

**Repository:** https://github.com/sonos/tract
**Current version:** `0.22.0` | **License:** Apache-2.0 / MIT

Tract is a **pure Rust** ONNX and TensorFlow Lite inference engine, built and battle-tested by Sonos for production deployment in constrained audio devices.

#### Strong Points

- **Zero C++ dependencies** — unlike `ort`, tract is 100% Rust, making it safe for WASM, embedded targets, and easy to cross-compile and audit.
- **Streaming inference** — `tract-pulse` enables online/streaming inference for sequential models (RNNs, audio processing pipelines).
- **Broad ONNX conformance** — passes ~85% of ONNX backend tests; all major vision models (ResNet, VGG, Inception, DenseNet) pass.
- **NNEF support** — can read/write the Neural Network Exchange Format for interoperability.
- **Python bindings** — available since May 2025, allowing Python-side model preparation and export.

#### Main Use Cases

- **Edge and embedded inference** — deployed in Sonos smart speakers; ideal when a C++ runtime is not an option.
- **Streaming audio/speech models** — tract-pulse is specifically designed for models that process sequences in real time.
- **WASM inference** — portable deployments in browser or serverless environments where `ort` cannot be used.

---

### tokenizers — `huggingface/tokenizers`

**Repository:** https://github.com/huggingface/tokenizers
**Current version:** `0.22.2` | **License:** Apache-2.0

The HuggingFace tokenizers library is written in Rust as its canonical implementation, with Python bindings layered on top. It is the industry-standard tokenization library for transformer-based LLMs.

#### Strong Points

- **Exceptional throughput** — tokenizes 1 GB of text in under 20 seconds on a single CPU.
- **Complete pipeline** — Normalizer → PreTokenizer → Model (BPE, WordPiece, Unigram) → PostProcessor → Decoder, all configurable.
- **HuggingFace Hub integration** — loads tokenizer configs directly from any model on the Hub.
- **Offset tracking and batch encoding** — built-in support for padding, truncation, and offset mapping needed for NLP tasks.
- **WASM support** — available via the `unstable_wasm` feature.

#### Main Use Cases

- Preprocessing text in Rust inference pipelines (pairs naturally with `candle`, `ort`, or `burn`).
- Building a complete Python-free LLM inference stack: `tokenizers` + `candle`/`ort` + `safetensors`.
- Tokenization in edge or WASM applications without a Python runtime.

---

### linfa — `rust-ml/linfa`

**Repository:** https://github.com/rust-ml/linfa
**Current version:** `0.7.1` | **License:** Apache-2.0 / MIT

Linfa is a modular, scikit-learn-inspired toolkit for **classical machine learning** in Rust. It fills the gap that deep learning frameworks leave: structured/tabular data, clustering, regression, and dimensionality reduction without neural networks.

#### Strong Points

- **Consistent scikit-learn-style API** — uniform `Fit`/`Predict` interface across all algorithm sub-crates, familiar to Python ML engineers.
- **Modular sub-crates** — use only what you need: `linfa-linear`, `linfa-clustering`, `linfa-trees`, `linfa-svm`, `linfa-reduction`, `linfa-preprocessing`.
- **Optional BLAS/LAPACK acceleration** — opt-in integration with OpenBLAS, Netlib, or Intel MKL for numerical routines.
- **Pure Rust core** — no Python or C++ runtime required; easy to embed in any Rust service.

#### Main Use Cases

- Classical supervised and unsupervised ML (regression, SVM, decision trees, K-Means, DBSCAN, Gaussian mixture models) in Rust services.
- Feature engineering and preprocessing pipelines (normalization, encoding, imputation, PCA).
- Lightweight ML for microservices and embedded systems where a deep learning framework would be overkill.

---

## Quick Comparison

| Library | Focus | Training | ONNX | GPU | Pure Rust | Status |
|---|---|---|---|---|---|---|
| **PyO3** | Rust ↔ Python interop | — | — | — | Yes | Stable (v0.28) |
| **Monty** | Sandboxed Python for AI agents | — | — | — | Yes | Experimental (v0.0.5) |
| **ort** | ONNX inference (via ONNX Runtime) | — | Yes | Yes (CUDA/TRT/etc.) | No (wraps C++) | RC (v2.0.0-rc.11) |
| **tch-rs** | PyTorch libtorch bindings | Yes | Via TorchScript | Yes (CUDA) | No (wraps C++) | Stable (v0.23) |
| **candle** | DL inference (LLMs, vision, audio) | Experimental | Weight loading | Yes (CUDA/Metal) | Yes | Stable (v0.9.1) |
| **burn** | Full DL framework (train + infer) | Yes | Import via burn-import | Yes (WGPU/CUDA) | Yes | Stable (v0.20) |
| **tract** | ONNX/TF edge inference | — | Yes (~85% ops) | No (CPU-focused) | Yes | Stable (v0.22) |
| **tokenizers** | Text tokenization | — | — | — | Yes (core) | Stable (v0.22.2) |
| **linfa** | Classical ML (non-DL) | Yes (classical) | — | No | Yes | Stable (v0.7.1) |
