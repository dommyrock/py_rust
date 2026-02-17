# Usecase - Classical ML training

Some guidelines when integrating custom trained ML models into existing Rust backends. 

---

## Recommended Architecture: Classical ML on a Rust/Axum Backend

This section translates the library overview above into a concrete, opinionated recommendation for a specific scenario: training **classical ML models** (regression, classification, clustering, SVM, decision trees) on tabular/structured data and serving them through an **Axum HTTP API** on a **standard CPU-only Linux server**.

The core insight is that no single library does everything well. The right answer is a **two-phase pipeline**: train offline using the best tooling available, then serve inside Rust using a lightweight runtime that loads the serialized model artifact.

---

### What to Eliminate First

Most libraries in this overview are irrelevant to this use case. Ruling them out explicitly avoids analysis paralysis:

| Library | Why it doesn't fit |
|---|---|
| **PyO3** | Builds Python extension modules or embeds Python *in* Rust. Adds complexity without benefit once a clean Python → ONNX → Rust pipeline is adopted. |
| **Monty** | Purpose-built for sandboxed LLM-generated Python execution in AI agents. No relevance to classical ML training or serving. |
| **tch-rs** | Wraps PyTorch libtorch — a ~500 MB C++ dependency designed for deep learning (autograd, neural nets). Overkill and impractical for classical ML. |
| **candle** | Inference-focused deep learning framework (LLMs, vision transformers). Has no classical ML algorithms (SVM, decision trees, regression). |
| **burn** | Full deep learning framework with excellent training support. No classical ML algorithms; designed for neural networks. |
| **tokenizers** | NLP tokenization only. Not applicable to tabular/structured data. |

The remaining candidates — **`ort`**, **`tract`**, and **`linfa`** — are the only three worth evaluating for this use case.

- [![Current Crates.io Version](https://img.shields.io/crates/v/ort.svg)](https://crates.io/crates/ort)
- [![Current Crates.io Version](https://img.shields.io/crates/v/tract-onnx.svg)](https://crates.io/crates/tract-onnx)
- [![Current Crates.io Version](https://img.shields.io/crates/v/linfa.svg)](https://crates.io/crates/linfa)

---

### Phase 1 — Training (offline, Python)

**Recommendation: Python + scikit-learn or XGBoost, exported to ONNX**

Training is a one-time or periodic offline step, completely decoupled from the Rust codebase. Python is the right tool here:

- **scikit-learn** alone covers 40+ classical estimators (linear/logistic regression, SVMs, decision trees, random forests, gradient boosting, K-Means, DBSCAN, GMMs, PCA, and more).
- **XGBoost / LightGBM** add best-in-class gradient boosted tree implementations when needed.
- Best-in-class ecosystem for EDA, cross-validation, feature pipelines, and hyperparameter tuning.
- Export any trained model to a static `.onnx` file via `sklearn-onnx` or `onnxmltools`.

The ONNX file is a portable artifact — commit it to the repository, store it in object storage, or embed it in the binary. The Rust code never knows or cares how it was trained.

```python
# Example: train a classifier, export to ONNX
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

clf = RandomForestClassifier().fit(X_train, y_train)

initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

### Phase 2 — Serving (Rust + Axum)

#### Primary: `ort` — async ONNX inference

**`ort`** (v2.0.0-rc.11) is the primary recommendation for serving inside Axum:

- **First-class async inference** — maps directly to Axum's async handler model; inference never blocks the executor.
- **Zero-copy tensors** via `TensorRef::from_array_view` — no unnecessary allocations on the inference hot path.
- **CPU execution provider** is the default and requires zero additional configuration for a Linux CPU server.
- **Production-proven** — SurrealDB embeds `ort` for ML queries; Hugging Face TEI uses it for high-throughput inference.

Load the ONNX session once at startup and share it across all handlers via Axum's `State` extractor:

```rust
use std::sync::Arc;
use axum::{extract::State, routing::post, Router};
use ort::{Session, inputs};
use ndarray::Array2;

#[derive(Clone)]
struct AppState {
    model: Arc<Session>,
}

async fn predict(
    State(state): State<AppState>,
    // extract your request body here
) -> impl axum::response::IntoResponse {
    let input: Array2<f32> = /* build from request */;

    let outputs = state.model
        .run(inputs!["float_input" => input.view()].unwrap())
        .unwrap();

    let predictions = outputs["output_label"]
        .try_extract_tensor::<i64>()
        .unwrap();

    axum::Json(predictions.as_slice().unwrap().to_vec())
}

#[tokio::main]
async fn main() {
    let session = Arc::new(
        Session::builder().unwrap()
            .commit_from_file("model.onnx").unwrap()
    );

    let state = AppState { model: session };

    let app = Router::new()
        .route("/predict", post(predict))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

```toml
# Cargo.toml
[dependencies]
axum = "0.8"
ort = { version = "2.0.0-rc.11", features = ["load-dynamic"] }
ndarray = "0.16"
tokio = { version = "1", features = ["full"] }
```

#### Alternative: `tract` — if zero C++ dependencies is required

If the C++ ONNX Runtime dependency that `ort` wraps is unacceptable (strict build pipeline, minimal Docker image, air-gapped environment):

- **`tract`** (v0.22.0) is 100% pure Rust with no C++ FFI — same ONNX input format.
- Passes ~85% of ONNX conformance tests; all major scikit-learn exported models (linear, SVM, tree-based) are covered.
- Slightly lower peak throughput than `ort`, negligible for classical ML model sizes on a CPU server.
- No native async API — wrap the blocking call in `tokio::task::spawn_blocking` inside Axum handlers:

```rust
use axum::{extract::State, Json};
use std::sync::Arc;
use tract_onnx::prelude::*;

type OnnxModel = Arc<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>;

async fn predict(State(model): State<OnnxModel>) -> Json<Vec<f32>> {
    let result = tokio::task::spawn_blocking(move || {
        let input = tvec!(Tensor::from(ndarray::arr2(&[[1.0f32, 2.0, 3.0, 4.0]])).into());
        model.run(input).unwrap()
    })
    .await
    .unwrap();

    Json(result[0].to_array_view::<f32>().unwrap().iter().copied().collect())
}
```

#### Alternative: `linfa` — full Rust, no Python, no ONNX

If Python in the training pipeline is undesirable (single-language monorepo, no Python tooling allowed):

- **`linfa`** (v0.7.1) provides a scikit-learn-style `Fit`/`Predict` API in pure Rust.
- Supports: linear/logistic regression, SVM, decision trees, K-Means, DBSCAN, GMM, PCA — no ONNX conversion needed.
- Serialize trained models with `serde` and load them at startup like any other configuration artifact.
- Limitation: narrower algorithm selection and less tooling for hyperparameter search and data exploration compared to Python.

---

### Decision Matrix

| Scenario | Train with | Serve with |
|---|---|---|
| **Recommended (pragmatic)** | Python + scikit-learn/XGBoost → ONNX | `ort` in Axum (async, zero-copy) |
| **No C++ deps in Rust binary** | Python + scikit-learn/XGBoost → ONNX | `tract` in Axum (pure Rust, `spawn_blocking`) |
| **Full Rust, no Python** | `linfa` (train in Rust, `serde` persist) | Direct in Axum handler (no ONNX needed) |
| **Full Rust + ONNX portability** | `linfa` → export to ONNX | `ort` or `tract` in Axum |
