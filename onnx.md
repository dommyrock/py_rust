# ONNX — Open Neural Network Exchange

## What It Is

ONNX is an open standard format for representing machine learning models, co-created by Microsoft and Meta in 2017 and now governed by the Linux Foundation AI & Data (LF AI&D). A broad consortium — Microsoft, Meta, NVIDIA, Intel, AMD, Qualcomm, IBM, and others — maintains it collaboratively.

The design philosophy rests on three pillars:

- **Framework neutrality.** Train in PyTorch, TensorFlow, scikit-learn, or anything else; export once to a self-contained `.onnx` file; run it anywhere without the original framework installed at inference time.
- **Hardware abstraction.** ONNX defines computation semantics; runtimes map those semantics to the actual hardware. A single model file can target CPUs, NVIDIA GPUs, AMD GPUs, Apple Neural Engine, Qualcomm NPUs, Intel VPUs, FPGAs, and cloud accelerators via a pluggable Execution Provider (EP) interface.
- **Stable, versioned contract.** The spec maintains a versioned IR and versioned operator sets (opsets), giving model producers and consumers a clear compatibility contract as the standard evolves.

As of early 2026: ONNX library **1.20.1**, IR version 10, opset **26** in the main domain.

---

## Format Specification

ONNX models are serialized via **Protocol Buffers**, stored as `.onnx` files. For models exceeding protobuf's 2 GB limit, an *External Data* mechanism splits the graph topology into a small `.onnx` file and companion weight files on disk.

### Structure

The root object is a `ModelProto` containing:

- `ir_version` — the ONNX IR spec version
- `opset_import` — list of `(domain, version)` pairs declaring which operator sets the model uses
- `graph` — a `GraphProto` with the actual computation
- metadata fields (`model_version`, `doc_string`, `metadata_props`)

A `GraphProto` is a **directed acyclic graph (DAG)** of `NodeProto` objects plus named inputs, outputs, and initializers (constant weights). Cycles are expressed via the `Loop` and `Scan` control-flow operators, which embed inner graphs as attributes.

Each `NodeProto` carries:
- `op_type` + `domain` — which operator to invoke
- `input` / `output` — named value edges
- `attribute` — operator-specific config (floats, ints, strings, tensors, or nested graphs)

### Type System

ONNX is tensor-centric. Key element types:

| Category | Types |
|---|---|
| Float | FLOAT32, FLOAT16, BFLOAT16, FLOAT64, FLOAT8E4M3FN, FLOAT8E5M2 |
| Integer | INT4, UINT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64 |
| Other | BOOL, STRING, COMPLEX64, COMPLEX128 |

Sequences and Maps are also supported for the `ai.onnx.ml` domain (classical ML outputs like class probability dicts).

### Opsets

Operators are grouped into versioned *operator sets*. The main domain (`""`) handles neural network ops; `ai.onnx.ml` handles classical ML algorithms (decision trees, SVMs, linear models). A model declares the exact opset version it targets; all consumers must honor those semantics.

---

## The Ecosystem

**ONNX Runtime (ORT)** — Microsoft's production inference engine, written in C++ with Python, C/C++, C#, Java, JavaScript, and Swift bindings. The primary deployment vehicle for ONNX models. Used by Microsoft across Azure, Office, Windows, Bing, and Xbox.

**ONNX Model Zoo** — pre-trained, pre-converted models for image classification, object detection, NLP, speech, face detection.

**Converter tooling:**

| Framework | Tool |
|---|---|
| PyTorch | `torch.onnx` (built-in; TorchDynamo exporter recommended as of 2.5+) |
| TensorFlow / Keras | `tf2onnx` |
| scikit-learn | `skl2onnx` (~133/194 estimators) |
| XGBoost / LightGBM / CatBoost | `onnxmltools` |
| PaddlePaddle | `paddle2onnx` |

---

## Strong Points

### Framework-to-Runtime Decoupling

The most practical benefit: a model trained in any supported framework can be deployed as a single `.onnx` artifact without the training framework present at runtime. This eliminates multi-gigabyte framework dependencies from serving containers, simplifies versioning, and lets platform teams serve models without sharing framework expertise with the research team.

### Broad Hardware Acceleration via Execution Providers

ORT's EP architecture partitions the model graph across available hardware backends. Nodes that an EP cannot handle fall back to CPU automatically. Production-grade EPs include:

| EP | Target |
|---|---|
| `CUDAExecutionProvider` | NVIDIA CUDA GPUs |
| `TensorrtExecutionProvider` | NVIDIA TensorRT (maximum NVIDIA perf) |
| `OpenVINOExecutionProvider` | Intel CPU / iGPU / NPU |
| `DirectMLExecutionProvider` | Windows DirectX 12 (NVIDIA, AMD, Intel) |
| `CoreMLExecutionProvider` | Apple Neural Engine / Apple GPU |
| `QNNExecutionProvider` | Qualcomm Snapdragon NPU / Adreno |
| `DnnlExecutionProvider` | Intel oneDNN — optimized CPU kernels |
| `XNNPACKExecutionProvider` | ARM / WASM mobile CPU |
| `NnapiExecutionProvider` | Android NNAPI |
| `ROCmExecutionProvider` | AMD ROCm GPUs |
| `MIGraphXExecutionProvider` | AMD MIGraphX |
| `VitisAIExecutionProvider` | Xilinx/AMD FPGAs |
| `CANNExecutionProvider` | Huawei Ascend NPU |
| `TvmExecutionProvider` | Apache TVM |

The same ORT API and code works across all these with only the EP selection changing — this breadth has no peer among competing runtimes.

### Graph Optimizations

ORT applies a multi-level optimization pipeline at session load time:

- **Level 1 (Basic):** constant folding, redundant node elimination, `Conv + BatchNorm` weight folding, bias absorption.
- **Level 2 (Extended):** complex fusions per EP — multi-head attention fusion, BERT embedding layer fusion, GELU and LayerNorm fusions, GEMM/MatMul fusions.
- **Level 3 (Layout):** NCHWc layout conversion for better SIMD cache utilization on x86 CPUs.

The optimized graph can be serialized back to disk to skip re-optimization on subsequent loads.

### Quantization

ORT ships a mature post-training quantization (PTQ) pipeline:

- **Formats:** QDQ (QuantizeLinear/DeQuantizeLinear, recommended) and QOperator (dedicated quantized op variants). Avoid `S8S8` QOperator on x86-64 — it is slow; use QDQ instead.
- **Precision targets:** INT8/UINT8, FP16, BF16, **INT4/UINT4** (block-wise weight-only for LLMs, requires opset 21+), FP8 (H100+).
- **Static PTQ calibration modes:** MinMax, Entropy (KL divergence minimization), Percentile.
- **Dynamic quantization:** per-batch scale computation, no calibration dataset required — good for RNNs and encoders.
- **FP16:** whole-model cast for Tensor Core GPUs.
- **INT4/UINT4:** supports RTN, GPTQ, and HQQ algorithms for LLM weight compression.

### Deterministic, Portable Artifacts

An `.onnx` file is a fully self-contained artifact (graph topology + weights). The same file produces identical results across machines, OS versions, and deployment dates — unlike a training checkpoint whose behavior can shift with library updates.

### Language-Agnostic Serving

The same ORT model runs from Python, C/C++, C#, Java, JavaScript (WASM), and Swift, all using the same underlying engine. A Python-trained model can be served from a Java or C++ microservice with no further conversion.

---

## Compromises and Limitations

### Operator Coverage Gaps

- New ops from PyTorch or TensorFlow take time to be standardized into an opset. Novel attention variants (Flash Attention, grouped-query attention), custom activations, and exotic normalization schemes may have no ONNX equivalent for months or years.
- `ai.onnx.ml` is stuck at opset 6, far behind the main domain at opset 26. Sparse feature support is absent.
- `skl2onnx` covers only 133/194 sklearn estimators. NMF, LDA, and sparse-matrix models are not convertible.
- Hardware EPs implement only a subset of ONNX operators. Nodes assigned to an EP that doesn't support them silently fall back to CPU, potentially negating acceleration benefits.

### Dynamic Shape Complexity

- **Data-dependent control flow** is the core tension. Python `if/else` on tensor values is invisible to the exporter — only the path executed by the dummy input is traced. The Dynamo exporter handles more cases but symbolic shapes must still be explicitly declared.
- **Dynamic axes must be specified at export time.** Forgetting to mark a batch or sequence dimension as dynamic produces a model that fails at runtime on any shape that differs from the dummy input.
- **Loop operator constraints.** The `Loop` op requires loop-carried variables to maintain fixed rank across iterations — a real constraint for variable-length output sequences.
- **Shape inference failures** degrade graph optimization quality when shapes cannot be statically propagated through certain operator sequences.

### Training Support Is Marginal

ONNX is overwhelmingly an **inference format**. The format has defined gradient operators since IR version 7, but:

- PyTorch's training export was deprecated in PyTorch 2.6.
- **ONNX Runtime Training** (`ORTModule`) wraps a PyTorch model and intercepts the forward pass; it is a *training accelerator*, not a training framework replacement. PyTorch must still be present.
- Full training solely in ONNX, without the originating framework, is not practical.
- **The practical rule:** use ONNX for inference; train in PyTorch, JAX, or TensorFlow; use `ORTModule` only if you have a large transformer where ORT's kernel optimizations measurably beat native PyTorch.

### Opset Version Compatibility

- **Forward incompatibility.** A model at opset 20 fails on a runtime that only supports up to opset 18. Re-export at a lower opset (possibly losing features) or upgrade the runtime.
- **Historical version explosion.** Runtime implementors must support every historical version of every operator they claim to handle — a significant maintenance burden that causes many runtimes to support only recent versions, breaking older models.
- **Silent semantic substitution.** Older ORT versions could silently fall back to a different operator version with subtly different semantics when encountering an unsupported opset, producing wrong results without error.
- **Semantic drift between versions.** `BatchNormalization` changed significantly between opset 8 and 9 (training-mode outputs removed). Models straddling such boundaries require care.

### Custom Operator Complexity

When a model uses ops not in the ONNX standard, the options are:
1. **Decompose into standard ops** — preferred, but may require significant effort and can sacrifice performance.
2. **Register a custom op in ORT** (C/C++ or Python) — not portable to other runtimes/EPs; Python-based custom ops are single-threaded due to the GIL, unsuitable for high-throughput production.

### Debugging Difficulty

- No interactive breakpoints in an ONNX graph. Debugging a numerical discrepancy requires instrumenting both the original framework model and the ORT model to output intermediate activations, then comparing layer by layer.
- Graph optimizer fusions obscure the relationship between ONNX nodes and original framework modules.
- Error messages reference auto-generated ONNX value names (e.g., `onnx::Conv_23`), not Python class/line information.

### The "Export Then Freeze" Overhead

- Every architecture change requires re-export, re-validation, and re-profiling. No incremental update mechanism exists.
- Exportability to ONNX can become an implicit design constraint, discouraging model authors from using patterns (complex autograd functions, Python-native data structures as state) that ONNX cannot represent.
- In fast-iteration environments, ONNX artifacts easily become stale relative to the latest checkpoint. Process discipline is required.
- Correctness must be verified in both the training framework (unit tests) and ORT (integration tests), roughly doubling the testing surface.

---

## ONNX vs. Competing Deployment Runtimes

| Criterion | ONNX Runtime | TensorRT | TorchScript | OpenVINO |
|---|---|---|---|---|
| Hardware coverage | Universal (16+ EPs) | NVIDIA only | PyTorch backends | Intel-primary |
| Peak NVIDIA GPU perf | Good (CUDA EP) / Best (TRT EP) | Best-in-class | Good | Limited |
| Peak Intel CPU perf | Good (oneDNN EP) | N/A | Good | Best-in-class |
| Framework dep at runtime | None | None | LibTorch | None |
| Language bindings | Python, C, C++, C#, Java, JS | C++, Python | C++, Python | C++, Python, Java |
| Quantization tooling | High maturity | High | Moderate | High |
| Dynamic shape support | Moderate | Moderate (with profiles) | Good | Moderate |
| Debugging ease | Low | Low | High | Moderate |

**TensorRT** wins on peak NVIDIA latency but is NVIDIA-only and requires expensive per-GPU engine build steps. Accepts ONNX as input, so ONNX often acts as the interchange format even in TensorRT pipelines.

**TorchScript** stays native to PyTorch (no conversion step, full debugging), but requires LibTorch or TorchServe at deployment and has no equivalent to ORT's EP abstraction. De-emphasized by the PyTorch team in the 2.x era in favor of `torch.export`.

**OpenVINO** is best-in-class for Intel CPU/GPU/NPU but is Intel-ecosystem focused. Accepts ONNX input.

---

## Export Best Practices

### PyTorch

**Always use the TorchDynamo exporter** (`dynamo=True`) for new models. The legacy `torch.jit.trace`-based exporter is deprecated as of PyTorch 2.6. The Dynamo exporter uses `torch.export` for AOT tracing, handles more Python control flow, and produces cleaner graphs.

```python
import torch

model = MyModel().eval()  # .eval() is critical — sets BN and Dropout to inference mode
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    (dummy,),
    "model.onnx",
    dynamo=True,
    dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=32)}},
    opset_version=21,
    export_params=True,
)
```

Key rules:
- **Call `.eval()` before export.** Exporting in training mode bakes stochastic dropout and training-mode BatchNorm semantics into the graph.
- **Declare dynamic axes explicitly.** Any axis that varies at runtime (batch size, sequence length) must be marked. A forgotten dynamic axis produces a shape-specialized model that fails on any other shape at runtime — silently or with a cryptic error.
- **Target the highest opset your deployment ORT version supports.** Higher opsets have better operator decompositions and more optimization surface. Check the [ORT compatibility matrix](https://onnxruntime.ai/docs/reference/compatibility.html).
- **Handle custom `autograd.Function` before export.** Register ONNX export handlers via `torch.onnx.register_custom_op_symbolic()`, or decompose custom ops into standard ops first.

**Always validate numerical equivalence after export:**
```python
import onnxruntime as ort
import numpy as np

with torch.no_grad():
    pt_out = model(dummy).numpy()

sess = ort.InferenceSession("model.onnx")
ort_out = sess.run(None, {"input": dummy.numpy()})[0]

np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-5)
```

A tolerance of `1e-3` / `1e-5` is typical; tighter for simple models, looser for long chains of ops where floating-point ordering differences accumulate. Failures here often indicate a missed dynamic axis, a Python branch that wasn't traced, or an op that decomposed differently.

Also run the ONNX checker and shape inference as a sanity gate:
```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model)  # populates value_info for ORT optimizers
```

Visualize the exported graph with [Netron](https://netron.app) to verify operator mappings match your expectations — especially after fusions or decompositions you didn't anticipate.

### TensorFlow / Keras

The canonical tool is `tf2onnx` (`pip install tf2onnx`). `keras2onnx` is frozen and unmaintained.

```bash
# From SavedModel (preferred — most complete TF serialization)
python -m tf2onnx.convert \
  --saved-model /path/to/saved_model \
  --output model.onnx \
  --opset 17

# From Keras .h5
python -m tf2onnx.convert --keras model.h5 --output model.onnx --opset 17

# Models > 2 GB
python -m tf2onnx.convert --saved-model /path/to/model --output model.onnx \
  --large_model --opset 17
```

- **Prefer SavedModel format.** It is the most complete TF serialization format and gives `tf2onnx` the most conversion fidelity. Frozen GraphDefs and TFLite files are supported but have more edge cases.
- **TF1 models must be frozen first** (replace Variables with Constants) before conversion.
- Custom TF ops not representable in ONNX require a custom handler plugin for `tf2onnx` or decomposition upstream.
- Check Hub model exportability early — some TF Hub models use internal graph patterns that `tf2onnx` cannot analyze. Finding this late is expensive.

---

## Rust Integration

ONNX's primary use case in Rust is **inference**. There is no practical ONNX-based training path in Rust. The ecosystem splits cleanly between bindings to the C++ ORT runtime (full feature set, C++ dependency) and pure-Rust engines (simpler deployment, reduced operator coverage).

### `ort` — The Primary Choice

`ort` (crate: `ort`, GitHub: `pykeio/ort`) is the de-facto standard Rust interface to ONNX Runtime. Current version: **2.0.0-rc.11** (January 2026), wrapping ORT 1.23/1.24. Despite the RC label the 2.x API is stable and production-deployed by Google Magika, HuggingFace Text Embeddings Inference, SurrealDB, and Supabase. Requires Rust 1.88+.

**Cargo.toml:**
```toml
[dependencies]
ort = { version = "=2.0.0-rc.11", features = ["ndarray", "cuda"] }
ndarray = "0.16"
```

**Session creation:**
```rust
use ort::{session::Session, execution_providers::CUDAExecutionProvider};

// Optional global init (once at startup)
ort::init()
    .with_execution_providers([CUDAExecutionProvider::default().build()])
    .commit()?;

let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("model.onnx")?;
```

**Inference with ndarray (zero-copy borrow):**
```rust
use ndarray::Array4;

let image: Array4<f32> = preprocess();  // shape [1, 3, 224, 224]
let input = ort::TensorRef::from_array_view(&image)?;

let outputs = session.run(ort::inputs!["input" => input]?)?;
let logits = outputs["output"].try_extract_array::<f32>()?;
```

**Session sharing across threads** — `Session` is `Send + Sync`; wrap it in `Arc`:
```rust
let session = Arc::new(Session::builder()?.commit_from_file("model.onnx")?);
```

**Tokio integration** — ORT's `run()` is CPU-bound; keep it off the async executor:
```rust
let result = tokio::task::spawn_blocking(move || {
    session.run(ort::inputs![input_ref]?)
}).await??;
```

For native async, `session.run_async(...).await` is available when `intra_threads > 1`.

**Execution providers** — configure per session, first match wins per operator:
```rust
Session::builder()?
    .with_execution_providers([
        TensorRTExecutionProvider::default()
            .with_engine_cache(true).build(),
        CUDAExecutionProvider::default().build(),
        // CPU is always the implicit final fallback
    ])?
    .commit_from_file("model.onnx")?;
```

Add `.error_on_failure()` to any EP to prevent silent CPU fallback (useful in CI).

**Linking.** The default `download-binaries` feature fetches prebuilt ORT shared libraries at build time for x86_64/aarch64 Linux/macOS/Windows with CPU + CUDA 12/cuDNN 9. For TensorRT, ROCm, OpenVINO, or custom targets, set `ORT_LIB_LOCATION` to a local ORT build and use the `load-dynamic` feature for runtime path configuration.

**The old `onnxruntime` crate** (last release: 0.0.14, August 2021) is effectively abandoned. Do not use it for new projects.

### Full Inference Pipeline

A realistic end-to-end example: image preprocessing → inference → output extraction.

```rust
use std::sync::Arc;
use ndarray::{Array4, s};
use ort::{session::Session, GraphOptimizationLevel};

fn preprocess(img: &image::DynamicImage) -> Array4<f32> {
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // ImageNet normalization: HWC pixel data → NCHW float tensor
    let mean = [0.485f32, 0.456, 0.406];
    let std  = [0.229f32, 0.224, 0.225];

    let mut tensor = Array4::<f32>::zeros((1, 3, 224, 224));
    for (y, row) in rgb.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            for c in 0..3usize {
                tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
    }
    tensor
}

fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed
}

fn run(session: &Session, img: &image::DynamicImage) -> anyhow::Result<Vec<(usize, f32)>> {
    let tensor = preprocess(img);

    // TensorRef borrows the array — no data copy
    let input = ort::TensorRef::from_array_view(&tensor)?;
    let outputs = session.run(ort::inputs!["input" => input]?)?;

    let logits = outputs["output"].try_extract_array::<f32>()?;
    Ok(top_k(logits.as_slice().unwrap(), 5))
}

fn main() -> anyhow::Result<()> {
    let session = Arc::new(
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file("resnet50.onnx")?
    );

    let img = image::open("cat.jpg")?;
    let predictions = run(&session, &img)?;
    for (class_id, score) in predictions {
        println!("class {class_id}: {score:.4}");
    }
    Ok(())
}
```

Key points:
- `TensorRef::from_array_view` borrows the ndarray without copying — the array must stay alive for the duration of the `run()` call.
- Named inputs (`"input" => ...`) are preferred over positional — they're robust to model changes that reorder inputs.
- `try_extract_array` returns an `ArrayViewD` into ORT-owned memory. Call `.to_owned()` if you need to keep the data after the `outputs` map is dropped.
- Output names can be inspected at startup: `session.outputs.iter().for_each(|o| println!("{}", o.name))`.

---

### IO Binding — Zero-Copy GPU Inference

By default, ORT copies input tensors from CPU host memory to the GPU before each inference call, and copies outputs back. For high-throughput GPU pipelines — especially when inputs arrive already on GPU (e.g., from a CUDA preprocessing pipeline) — this copy overhead dominates latency.

`IoBinding` pins pre-allocated device buffers as named inputs and outputs, eliminating those host↔device copies entirely.

```rust
use ort::{session::Session, execution_providers::CUDAExecutionProvider, memory::Allocator};

let session = Session::builder()?
    .with_execution_providers([CUDAExecutionProvider::default().build()])?
    .commit_from_file("model.onnx")?;

// Get an allocator for the CUDA device
let cuda_alloc = session.allocator();  // device allocator from the EP

let mut binding = session.create_binding()?;

// Bind input: either from an existing GPU tensor or a host tensor that ORT uploads once
binding.bind_input("input", &input_tensor)?;

// Bind output: ORT allocates the output buffer on the GPU; no CPU copy on read
binding.bind_output_to_device("output", &cuda_alloc)?;

// Run — no host↔device copies for bound buffers
session.run_binding(&binding)?;

// Outputs stay on device; extract only if you need the data on CPU
let outputs = binding.outputs()?;
let result = outputs["output"].try_extract_array::<f32>()?; // triggers device→host copy here
```

**When IO binding matters:** latency-critical GPU serving where throughput is bottlenecked by PCIe bandwidth, not compute. For most CPU workloads it is not needed.

**Reuse bindings across calls:** rebind only when input shapes change. For a fixed-batch-size service, create the binding once at startup and reuse it for every request.

---

### Async Patterns

ORT's `run()` is synchronous and CPU-bound. There are two ways to integrate it with async Rust:

**1. `spawn_blocking` — universal, works with any EP**

The standard Tokio pattern for CPU-bound work. Moves the inference call to the blocking thread pool, preventing it from starving async tasks:

```rust
use std::sync::Arc;
use ort::session::Session;

#[derive(Clone)]
struct Classifier(Arc<Session>);

impl Classifier {
    async fn predict(&self, input: ndarray::ArrayD<f32>) -> anyhow::Result<ndarray::ArrayD<f32>> {
        let session = Arc::clone(&self.0);
        tokio::task::spawn_blocking(move || {
            let input_ref = ort::TensorRef::from_array_view(&input)?;
            let outputs = session.run(ort::inputs![input_ref]?)?;
            Ok(outputs[0].try_extract_array::<f32>()?.to_owned())
        })
        .await?
    }
}
```

The `Arc<Session>` clone is cheap — it only increments a reference count. Multiple concurrent `spawn_blocking` calls on the same session are safe; ORT handles internal thread-safety.

**2. `run_async` — native ORT async, lower overhead**

`run_async` uses ORT's internal thread pool and returns a `Future` directly, avoiding the overhead of Tokio's blocking pool. Requires `intra_threads > 1` on the session.

```rust
let session = Session::builder()?
    .with_intra_threads(4)?  // required for run_async
    .commit_from_file("model.onnx")?;

async fn infer(session: &Session, input: ort::TensorRef<'_, f32>) -> anyhow::Result<()> {
    let outputs = session.run_async(ort::inputs![input]?)?.await?;
    let arr = outputs["output"].try_extract_array::<f32>()?;
    println!("shape: {:?}", arr.shape());
    Ok(())
}
```

`run_async` is only available with the real ORT backend (not `tract` or `rten` fallback backends). The future is cancel-safe — dropping it before completion cancels the ORT inference call.

**Which to use:** prefer `run_async` for new code where you control the session config. Use `spawn_blocking` when working with third-party sessions, existing sync code, or pure-Rust backends.

---

### `tract` — Pure Rust, Zero C++ Dependencies

`tract` (Sonos, crate: `tract-onnx`, v0.22.0) is a pure-Rust ONNX inference engine with no C++ dependency. It passes ~85% of the official ONNX backend test suite and runs all common CNN architectures (ResNet, VGG, MobileNet, Inception, SqueezeNet). No GPU support; all execution is on CPU via SIMD intrinsics.

```rust
use tract_onnx::prelude::*;

let model = tract_onnx::onnx()
    .model_for_path("model.onnx")?
    .into_optimized()?
    .into_runnable()?;

let input = tvec!(Tensor::from(ndarray::arr1(&[1.0f32, 2.0, 3.0])).into());
let result = model.run(input)?;
```

**Choose `tract` when:** the target is embedded, WASM, or any platform where shipping a C++ shared library is impractical; operator coverage is sufficient; binary size matters.

**Choose `ort` when:** GPU inference, transformer-heavy models, maximum CPU performance, or access to TensorRT/DirectML is required.

---

### `rten` — Pure Rust, Near-ORT CPU Performance

`rten` (`github.com/robertknight/rten`) is a newer pure-Rust ONNX engine that directly loads `.onnx` files. On M3 Pro, CPU performance is benchmarked "in the same ballpark as ONNX Runtime" for common models. Added `Loop` support for variable-length sequences and int4/int8 quantization in 2025. Metal (macOS GPU) acceleration is on the 2026 roadmap. Effectively a modern alternative to `tract` with better transformer support.

---

### `burn` — ONNX as a Compile-Time Import

`burn` (`v0.20.1`, January 2026) takes a fundamentally different approach: `burn-import`'s `ModelGen` in `build.rs` translates an `.onnx` file into native Rust source code at compile time, using Burn's tensor API. The result is a statically typed Rust struct that runs on any Burn backend (CPU/ndarray, WebGPU, WASM, CUDA).

```rust
// build.rs
use burn_import::onnx::ModelGen;
fn main() {
    ModelGen::new()
        .input("src/model/model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

**Strengths:** the generated code is a normal Burn model that can be fine-tuned, runs on any backend including WASM, and weights can be embedded directly in the binary (`embed_states = true`).

**Limitations:** requires opset 16+; operator coverage is partial and depends on `burn-import` maturity; model structure is baked in at compile time (no runtime `.onnx` loading); if an op is unsupported, compilation fails.

Best suited when ONNX is used as a starting point for further training or WASM deployment rather than as a deployment artifact for inference-only workloads.

---

### `candle-onnx` — Loading Weights into Candle

HuggingFace's `candle-onnx` (v0.9.2) parses ONNX protobuf at runtime and dispatches ops to `candle`'s tensor kernels. Operator coverage is limited (notably lower than `tract`). Best understood as a convenience layer for loading HuggingFace Hub ONNX weights into native candle tensors, not as a general-purpose inference runtime. Requires `protoc` installed at build time.

---

### Choosing the Right Rust ONNX Approach

| | `ort` | `tract` | `rten` | `burn-import` | `candle-onnx` |
|---|---|---|---|---|---|
| GPU | Yes (CUDA, TRT, etc.) | No | No (Metal planned) | Via Burn backends | Via candle-cuda |
| Op coverage | Full | ~85% ONNX tests | Good, growing | Partial (opset 16+) | Limited |
| C++ dependency | Yes (ORT) | No | No | Depends on backend | No |
| Cross-compile | Hard | Trivial | Trivial | Depends | Mostly trivial |
| Async (`run_async`) | Yes | No | No | No | No |
| Fine-tuning after import | No | No | No | Yes (Burn training) | No |
| Best for | Production GPU/CPU inference | Embedded / WASM / simple deployment | CPU inference, near-ORT perf | ONNX as training starting point | Candle-native pipelines |

---

## References

1. **[ONNX Operator Specification](https://onnx.ai/onnx/operators/)** — canonical reference for every operator: inputs, outputs, attributes, type constraints, and the exact semantics at each opset version. Essential when debugging export mismatches or checking whether a custom op can be decomposed.

2. **[ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)** — covers each EP's requirements, configuration options, supported operator subsets, and build instructions. The right starting point when targeting a specific hardware backend.

3. **[ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance/)** — practical guide to graph optimization levels, thread configuration, IO binding, memory arena tuning, and profiling. Covers the difference between `intra_op` and `inter_op` parallelism and when each matters.

4. **[ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)** — full PTQ workflow: QDQ vs QOperator format tradeoffs, calibration modes, static vs dynamic quantization, per-channel vs per-tensor, INT4/FP16 paths, and the accuracy debugging utilities.

5. **[PyTorch ONNX Export (`torch.onnx`)](https://pytorch.org/docs/stable/onnx.html)** — documents the TorchDynamo exporter API, `dynamic_shapes` declaration, supported operator coverage, known limitations, and the migration path from the legacy `torch.jit.trace`-based exporter.
