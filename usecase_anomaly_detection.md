# Anomaly Detection, Trend Prediction & Time Series on a Rust/Axum Backend

A deep-dive reference for training time series and anomaly detection models — either with Python offline or natively in Rust — and serving them inside an Axum-based Rust backend on a CPU-only Linux server.

> **Last updated:** February 17, 2026

---

## Table of Contents

- [Overview: The Three-Layer Problem](#overview-the-three-layer-problem)
- [Python Libraries for Training](#python-libraries-for-training)
  - [Anomaly Detection](#anomaly-detection-python)
  - [Time Series Forecasting & Trend Prediction](#time-series-forecasting--trend-prediction-python)
  - [Feature Engineering for Time Series](#feature-engineering-for-time-series)
- [ONNX Export Compatibility Matrix](#onnx-export-compatibility-matrix)
- [Rust Libraries for Native Inference](#rust-libraries-for-native-inference)
  - [augurs — the primary Rust time series toolkit](#augurs--grafanaaugurs)
  - [anomaly_detection — ankane port](#anomaly_detection--ankaneanomalydetectionrs)
  - [ort — ONNX Runtime bindings](#ort--for-sklearn-exported-models)
  - [Other Rust crates](#other-rust-crates)
- [Alternative Approaches for Non-ONNX Models](#alternative-approaches-for-non-onnx-models)
  - [Prophet](#prophet--augurs-prophet-or-pre-computed-forecasts)
  - [ARIMA / SARIMA](#arima--sarima--parameter-extraction--rust-inference)
  - [EllipticEnvelope](#ellipticenvelope--mahalanobis-distance-in-nalgebra)
  - [STUMPY / Matrix Profile](#stumpy--matrix-profile--pre-computed-anomaly-scores)
- [Axum Integration Patterns](#axum-integration-patterns)
  - [Sliding window streaming detection](#pattern-1-sliding-window-streaming-anomaly-detection)
  - [Model hot-swap on retraining](#pattern-2-atomic-model-hot-swap-on-retraining)
  - [Threshold-based vs model-based](#pattern-3-threshold-based-vs-model-based-hybrid)
- [Decision Matrix](#decision-matrix)
- [SaaS Alternatives](#saas-alternatives)

---

## Overview: The Three-Layer Problem

Time series and anomaly detection workloads introduce challenges not present in classical ML:

1. **Temporal dependency** — data points are not i.i.d.; the order and spacing of observations matters.
2. **Seasonality and trends** — normal behavior changes by hour, day, and week; a model trained naively will flag normal seasonal peaks as anomalies.
3. **Concept drift** — the data distribution shifts over time; a model trained last quarter may be wrong today.

The architecture from the [classical ML recommendation](./README.md#recommended-architecture-classical-ml-on-a-rustaxum-backend) still applies — train offline, serve in Rust — but the export paths are more varied, and some models have no ONNX path at all.

**The expanded pattern:**

```
[Python: train offline]
  ├── sklearn anomaly models  ──→ ONNX (sklearn-onnx) ──→ ort in Axum
  ├── Prophet                 ──→ JSON params          ──→ augurs-prophet in Axum
  ├── ARIMA/SARIMA            ──→ coefficient JSON     ──→ reimplemented recursion in Rust
  ├── darts / tsai (DL)       ──→ ONNX (torch.onnx)   ──→ ort in Axum
  └── STUMPY                  ──→ anomaly scores JSON  ──→ lookup / baseline in Rust

[Rust: native, no Python at all]
  ├── augurs-mstl + augurs-ets         → decomposition + ETS forecasting
  ├── augurs-prophet                   → Prophet-equivalent inference
  ├── augurs-changepoint               → BOCPD changepoint detection
  ├── augurs-outlier                   → group-level outlier detection
  └── anomaly_detection (ankane)       → S-H-ESD on seasonal time series
```

---

## Python Libraries for Training

### Anomaly Detection (Python)

---

#### scikit-learn — Anomaly Detection Estimators

**Repository:** https://github.com/scikit-learn/scikit-learn
**Current version:** `1.6.1`

scikit-learn ships four dedicated anomaly/outlier detection estimators. All follow the standard `fit()` / `predict()` / `score_samples()` / `decision_function()` API. Three of the four are exportable to ONNX.

| Estimator | Type | ONNX exportable | Notes |
|---|---|---|---|
| `IsolationForest` | Ensemble (random trees) | **Yes** (with `zipmap=False`) | Best general-purpose anomaly detector; scales well; no distribution assumptions |
| `OneClassSVM` | Kernel method | **Yes** (with `zipmap=False`) | Good for low-dimensional, dense data; expensive at large scale |
| `LocalOutlierFactor` (`novelty=True`) | Density-based | **Yes** | Must be fitted with `novelty=True` for `predict()` to work on new data |
| `EllipticEnvelope` | Gaussian assumption | **No** | No sklearn-onnx converter exists; must reimplement inference manually |

**Key export detail:** sklearn-onnx by default wraps classifier outputs in a `ZipMap` ONNX operator that converts the output tensor to a list of dictionaries. `ZipMap` is not supported by most Rust ONNX runtimes. Always disable it:

```python
from sklearn.ensemble import IsolationForest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

clf = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
clf.fit(X_train)

initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(
    clf,
    initial_types=initial_type,
    options={id(clf): {"zipmap": False}},           # <-- critical
    target_opset={"": 17, "ai.onnx.ml": 2},
)

with open("isolation_forest.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

`IsolationForest.predict()` returns `+1` (normal) or `-1` (anomaly). `decision_function()` returns the raw anomaly score (negative = more anomalous). The `score_samples()` output is the most useful for thresholding.

---

#### PyOD — Python Outlier Detection

**Repository:** https://github.com/yzhao062/pyod
**Current version:** `2.0.6`
**Paper:** PyOD 2 (ACM 2025)

PyOD is the most comprehensive Python outlier detection library with **45+ algorithms** across probabilistic, linear model, proximity-based, ensemble, and deep learning categories.

**Selected algorithms relevant to time series anomaly detection:**

| Algorithm | Type | Description |
|---|---|---|
| `IForest` | Ensemble | Wraps sklearn IsolationForest with PyOD interface |
| `HBOS` | Statistical | Histogram-Based Outlier Score; extremely fast; online-updatable histograms |
| `COPOD` | Probabilistic | Copula-Based Outlier Detection; parameter-free; good for high-dimensional data |
| `ECOD` | Probabilistic | Empirical Cumulative distribution-based; no hyperparameters; fast |
| `LOF` | Proximity | Local Outlier Factor; same as sklearn variant |
| `OCSVM` | Kernel | One-Class SVM; same as sklearn variant |
| `AutoEncoder` | Neural | Simple autoencoder reconstruction error anomaly detection (PyTorch) |
| `DeepSVDD` | Neural | Deep Support Vector Data Description |

**ONNX support:** PyOD does not have a built-in ONNX export. For sklearn-compatible models (`IForest`, `HBOS`, `LOF`, `OCSVM`), the underlying `clf.detector_` attribute is a sklearn estimator and can be exported via `sklearn-onnx`. For neural network models (`AutoEncoder`, `DeepSVDD`), use `torch.onnx.export()`.

**Serialization:** Use `joblib` or `pickle`. For production, prefer `joblib`:
```python
from joblib import dump, load
dump(clf, "pyod_model.joblib")
clf = load("pyod_model.joblib")
```

---

#### STUMPY — Matrix Profile

**Repository:** https://github.com/stumpy-dev/stumpy
**Current version:** `1.13.0`

STUMPY implements the **Matrix Profile** algorithm, which computes the distance between every subsequence of a time series and its nearest neighbor. Anomalies appear as local maxima in the matrix profile — regions where no similar subsequence exists.

**When to use:** Time series data where anomalies are unusual shapes/patterns (not just extreme values). Detects novel subsequences, motifs (repeating patterns), and discord (most unusual subsequence).

```python
import stumpy
import numpy as np

ts = np.loadtxt("timeseries.csv")
window_size = 60  # match your expected anomaly duration

mp = stumpy.stump(ts, m=window_size)  # shape: (n - m + 1, 4)
# mp[:, 0] = matrix profile values (distances). High = anomalous.
# mp[:, 1] = matrix profile index (nearest neighbor index)

anomaly_threshold = np.percentile(mp[:, 0], 95)
anomaly_indices = np.where(mp[:, 0] > anomaly_threshold)[0]
```

**ONNX export:** Not applicable. STUMPY is a pure algorithm with no trainable model. The output is a numpy array of distance scores.

**How to use results in Rust:** Run STUMPY in Python on a schedule, export the matrix profile and threshold as a JSON or Parquet file. The Rust service reads this as a baseline reference. For real-time scoring, compute the z-normalized Euclidean distance from a new incoming window to the nearest neighbor in a precomputed reference set using `ndarray` in Rust.

---

### Time Series Forecasting & Trend Prediction (Python)

---

#### Prophet (Meta/Facebook)

**Repository:** https://github.com/facebook/prophet
**Current version:** `1.1.6`

Prophet models time series as an **additive model**: `y(t) = trend(t) + seasonality(t) + holidays(t) + noise`. It is designed for business metrics with strong seasonal effects.

**Components:**
- **Trend:** Piecewise linear or logistic growth; changepoints are automatically detected or manually specified.
- **Seasonality:** Fourier series approximation; default Fourier order 10 for yearly, 3 for weekly.
- **Holidays:** User-specified date effects with individual priors.

**ONNX export:** **Not supported.** Prophet's fitting procedure relies on Stan (PyStan or CmdStan) for MAP estimation. There is no sklearn-onnx or onnxmltools converter.

**Best options for Rust deployment:** See [Alternative Approaches — Prophet](#prophet--augurs-prophet-or-pre-computed-forecasts).

**Model serialization (Python-side):**
```python
import json
with open("prophet_model.json", "w") as f:
    f.write(model.to_json())
# Reload in Python: model = model_from_json(json.load(open(...)))
```

---

#### statsmodels — ARIMA, SARIMA, Holt-Winters

**Repository:** https://github.com/statsmodels/statsmodels
**Current version:** `0.14.4`

The most complete Python library for classical time series econometric models.

**Relevant models:**

| Model | Class | Use case |
|---|---|---|
| ARIMA | `ARIMA` | General non-seasonal series; requires stationarity |
| SARIMA | `SARIMAX(seasonal_order=...)` | Seasonal series (monthly/weekly patterns) |
| SARIMAX | `SARIMAX(exog=...)` | With exogenous variables (e.g., temperature → energy use) |
| Holt-Winters | `ExponentialSmoothing` | Simple trend + seasonality without ARIMA components |

**ONNX export:** **Not supported.** No converters exist in sklearn-onnx or onnxmltools for statsmodels objects.

**Best options for Rust:** See [Alternative Approaches — ARIMA](#arima--sarima--parameter-extraction--rust-inference).

**Model serialization:**
```python
# Save fitted model
result = model.fit()
result.save("arima_model.pkl")

# Extract parameters for manual Rust inference
params = {
    "arparams": result.arparams.tolist(),   # AR coefficients
    "maparams": result.maparams.tolist(),   # MA coefficients
    "sigma2": float(result.sigma2),
    "d": d,                                  # integration order
    "fittedvalues": result.fittedvalues.tolist(),
    "resid": result.resid.tolist(),
}
```

---

#### darts — Time Series Forecasting Library

**Repository:** https://github.com/unit8co/darts
**Current version:** `0.32.0`

darts provides a unified interface for 30+ forecasting models from ARIMA to transformers. Most relevant: it supports **ONNX export for PyTorch-based models**.

**ONNX export (torch models only):**
```python
from darts.models import NBEATSModel, TFTModel, TCNModel

model = NBEATSModel(input_chunk_length=24, output_chunk_length=6)
model.fit(training_series)
model.to_onnx("nbeats_model.onnx")  # available for all torch forecasting models
```

Classical stat models in darts (ARIMA, ETS, Theta) do not support ONNX export — use their parameter serialization or `augurs` equivalents in Rust.

---

#### tsai — State-of-the-Art Time Series DL

**Repository:** https://github.com/timeseriesAI/tsai
**Current version:** `0.3.9`

tsai is a library for time series classification, regression, and anomaly detection using deep learning (built on fastai + PyTorch). Includes InceptionTime, ROCKET, TSiT (ViT for time series), PatchTST, and more.

**ONNX export:**
```python
from tsai.export import get_ts_learner
learner = get_ts_learner(...)
learner.export("model.pkl")  # fastai format
# Or export via torch.onnx directly from the underlying model:
import torch
dummy_input = torch.zeros(1, n_channels, seq_len)
torch.onnx.export(learner.model, dummy_input, "model.onnx", opset_version=17)
```

Exported models run in `ort` in Rust as standard neural network inference.

---

#### Merlion — Salesforce Unified AIOps Library

**Repository:** https://github.com/salesforce/Merlion
**Current version:** `2.0.0`

Merlion provides a unified framework for anomaly detection and forecasting with AutoML capabilities. Supports ensembles, anomaly benchmarking, and dashboard visualization.

**ONNX support:** **None.** Merlion models serialize to a custom JSON format via `model.save()`. No ONNX path exists.

**Verdict for this use case:** Use Merlion for experimentation and model selection in Python. For production serving in Rust, extract the winning algorithm type and re-implement via the ONNX or parameter-extraction paths above.

---

### Feature Engineering for Time Series

#### tsfresh — Automated Time Series Feature Extraction

**Repository:** https://github.com/blue-yonder/tsfresh
**Current version:** `0.21.0`

tsfresh extracts hundreds of statistical features from raw time series windows: FFT coefficients, autocorrelation, approximate entropy, wavelet energy, trend line parameters, etc. It integrates as a sklearn `Transformer`, so a `Pipeline(tsfresh_transformer, IsolationForest)` is ONNX-exportable as a unit.

```python
from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from skl2onnx import convert_sklearn

pipeline = Pipeline([
    ("features", RelevantFeatureAugmenter(column_id="id", column_sort="time")),
    ("clf", IsolationForest(contamination=0.05, random_state=42)),
])
pipeline.fit(X_train_windows, y_train)

# Export the full pipeline
initial_type = [("input", FloatTensorType([None, X_train_windows.shape[1]]))]
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={"zipmap": False},
)
```

**Note:** The feature extraction step must be pre-computed before passing to the ONNX model in Rust (i.e., the ONNX model expects the feature vector, not raw windows). Replicate the exact same feature extraction in Rust for serving — this is a maintenance risk; document it carefully.

---

## ONNX Export Compatibility Matrix

This matrix covers which models can reach `ort` (or `tract`) in Rust via ONNX:

| Python model | ONNX export tool | Export works | `ort` (Rust) | `tract` (Rust) | Notes |
|---|---|---|---|---|---|
| `IsolationForest` | sklearn-onnx | **Yes** (`zipmap=False`) | Yes | Uncertain — test required | Uses `ai.onnx.ml` domain ops |
| `OneClassSVM` | sklearn-onnx | **Yes** (`zipmap=False`) | Yes | Uncertain | `SVMRegressor` operator |
| `LOF` (`novelty=True`) | sklearn-onnx | **Yes** | Yes | Likely partial | Standard + `TopK` ops |
| `EllipticEnvelope` | sklearn-onnx | **No** | N/A | N/A | No converter exists |
| `sklearn` Pipeline | sklearn-onnx | **Yes** | Yes | Yes (simple ops) | ONNX ops depend on steps |
| PyOD sklearn-backed models | sklearn-onnx | **Yes** (extract `detector_`) | Yes | Uncertain | Same as their sklearn equivalents |
| PyOD neural models (`AutoEncoder`) | torch.onnx.export | **Yes** | Yes | Yes | Standard DL ops |
| Prophet | — | **No** | N/A | N/A | Stan-based; no ONNX path |
| statsmodels ARIMA | — | **No** | N/A | N/A | Parameter extraction only |
| Holt-Winters (`ExponentialSmoothing`) | — | **No** | N/A | N/A | Parameter extraction only |
| darts PyTorch models | `model.to_onnx()` | **Yes** | Yes | Yes | Standard DL ops |
| tsai models | torch.onnx.export | **Yes** | Yes | Yes | Standard DL ops |
| STUMPY | N/A | **No** | N/A | N/A | Algorithm, not a model |

> **Runtime recommendation:** Use `ort` (not `tract`) for any ONNX model originating from sklearn-onnx. Models in the `ai.onnx.ml` domain require full ONNX Runtime support which only `ort` provides. `tract` is better suited for pure neural network ONNX models (standard `ai.onnx` domain ops only).

---

## Rust Libraries for Native Inference

### augurs — `grafana/augurs`

**Repository:** https://github.com/grafana/augurs
**crates.io:** https://crates.io/crates/augurs
**Current version:** `0.10.1` (January 2026)
**License:** Apache-2.0 / MIT
**WASM support:** Yes

`augurs` is the most important Rust library for this use case. Developed by Grafana Labs, it is a modular time series toolkit that powers Grafana Cloud's own ML features. It covers forecasting, outlier detection, changepoint detection, and an experimental Prophet reimplementation — all in pure Rust with no C++ dependencies and optional WASM targets.

[![Current Crates.io Version](https://img.shields.io/crates/v/augurs.svg)](https://crates.io/crates/augurs)

#### Sub-crates

| Crate | Purpose | Status |
|---|---|---|
| `augurs-mstl` | MSTL decomposition (Multi-Seasonal STL) + trend/seasonality extraction | Stable |
| `augurs-ets` | Exponential Smoothing State Space (ETS) forecasting; automatic parameter selection | Stable |
| `augurs-prophet` | Prophet reimplementation in Rust; uses CmdStan for fitting, pure Rust for inference | Stable |
| `augurs-outlier` | Outlier detection across a group of time series; DBSCAN-based and MAD-based | Stable |
| `augurs-changepoint` | Bayesian Online Changepoint Detection (BOCPD) with Normal-Gamma conjugate prior | Stable |
| `augurs-dtw` | Dynamic Time Warping distance; used by clustering and outlier detection | Stable |
| `augurs-clustering` | DBSCAN clustering using DTW distance on time series | Stable |

#### augurs-mstl + augurs-ets: Forecasting Pipeline

```toml
[dependencies]
augurs = { version = "0.10.1", features = ["mstl", "ets"] }
```

```rust
use augurs::{
    ets::AutoETS,
    mstl::MSTLModel,
};

// Fit MSTL + ETS on a daily time series with weekly seasonality
let data: Vec<f64> = vec![/* your historical data */];
let seasonality_periods = vec![7usize];  // weekly

let ets = AutoETS::non_seasonal();
let mut model = MSTLModel::new(seasonality_periods, ets);
model.fit(&data).expect("fit failed");

let forecast = model.predict(24, 0.95).expect("predict failed");
// forecast.point    → point predictions
// forecast.lower    → lower confidence bound (if level provided)
// forecast.upper    → upper confidence bound
```

#### augurs-prophet: Prophet Equivalent in Rust

The fitting step uses Stan's MAP optimizer. For production, run fitting in Python (`augurs-prophet` also has Python bindings via PyPI) and load the resulting parameters into the Rust inference path:

```toml
[dependencies]
augurs = { version = "0.10.1", features = ["prophet", "prophet-cmdstan"] }
```

```rust
use augurs::prophet::{Prophet, TrainingData, PredictionData};

// Option 1: fit in Rust using CmdStan binary
let mut prophet = Prophet::new(Default::default(), cmdstan_optimizer);
prophet.fit(&training_data, Default::default())?;
let predictions = prophet.predict(Some(&future_data))?;

// Option 2: load pre-fitted parameters from Python export
// (load the Stan JSON output from Python training, feed into Prophet inference)
```

#### augurs-outlier: Group-Level Outlier Detection

Detects which time series in a collection are outliers relative to the group — useful for per-endpoint, per-host, or per-tenant anomaly detection:

```rust
use augurs::outlier::{OutlierDetector, DbscanDetector};

let detector = DbscanDetector::with_sensitivity(0.5).unwrap();
let series: &[&[f64]] = &[&[1.0, 2.0, 1.5], &[1.2, 2.1, 1.6], &[100.0, 200.0, 150.0]];
let result = detector.preprocess(series).unwrap()
    .detect().unwrap();

for (i, outlier_info) in result.series_results.iter().enumerate() {
    if outlier_info.is_outlier {
        println!("Series {} is an outlier", i);
    }
}
```

#### augurs-changepoint: Bayesian Online Changepoint Detection

Detects when the statistical properties of a time series change (trend shifts, variance changes):

```rust
use augurs::changepoint::{ArgpcpDetector, Detector};

let detector = ArgpcpDetector::default();
let data = vec![1.0, 1.1, 0.9, 1.0, 1.2, 5.0, 5.1, 4.9, 5.2];
let result = detector.detect_changepoints(&data);
println!("Changepoints at indices: {:?}", result.indices);
```

---

### anomaly_detection — `ankane/AnomalyDetection.rs`

**Repository:** https://github.com/ankane/AnomalyDetection.rs
**crates.io:** https://crates.io/crates/anomaly_detection
**Current version:** `0.3.x`
**License:** GPL-3.0

A Rust port of Twitter's AnomalyDetection R package. Uses the **Seasonal Hybrid ESD (S-H-ESD)** test, which is built on STL decomposition followed by a Generalized ESD (Extreme Studentized Deviate) statistical test.

**Best for:** Univariate time series with known weekly/daily seasonality and an approximately known number of anomalies.

```toml
[dependencies]
anomaly_detection = "0.3"
```

```rust
use anomaly_detection::{AnomalyDetector, Direction};

let series: Vec<f64> = vec![/* your time series */];
let period = 168;  // weekly seasonality at hourly resolution = 7 * 24

let detector = AnomalyDetector::builder()
    .alpha(0.05)            // significance level
    .max_anoms(0.1)         // max 10% of points can be anomalies
    .direction(Direction::Both)
    .period(period)
    .build();

let result = detector.detect(&series);
for anomaly in &result.anomalies {
    println!("Anomaly at index {}: value {}", anomaly.index, anomaly.value);
}
```

**Limitation:** GPL-3.0 license — check compatibility with your project's license.

---

### ort — For sklearn-Exported Models

See the [README.md Recommended Architecture](./README.md#recommended-architecture-classical-ml-on-a-rustaxum-backend) for full `ort` integration details. The pattern is identical for anomaly detection: export from sklearn-onnx with `zipmap=False`, load via `Arc<Session>`, share across Axum handlers via `State`.

```rust
// IsolationForest inference: negative score = anomaly
let outputs = session.run(inputs!["float_input" => features.view()]?)?;
let scores: ArrayViewD<f32> = outputs["variable"].try_extract_tensor()?;
let is_anomaly = scores[0] < -0.05;  // threshold tuned on validation data
```

---

### Other Rust Crates

#### s2gpp — Series2Graph++ (Multivariate)
**crates.io:** https://lib.rs/crates/s2gpp

Rust implementation of graph-based anomaly detection for **multivariate** time series. Uses graph representations of trajectory patterns. Niche but self-contained. Useful when you have 2–10 correlated metrics and need to detect joint anomalies.

#### scirs2-series — SciRS2 Time Series Module
**crates.io:** https://crates.io/crates/scirs2-series
**Version:** `0.1.0-alpha.1` (part of SciRS2 v0.1.5, February 2026)

Part of the SciRS2 project (a SciPy port to Rust). Provides ARIMA-family models, autocorrelation, and trend analysis. **Alpha stage — not yet production-ready.** Monitor for stabilization; this could become a useful alternative to `augurs-ets` for ARIMA-style models.

#### rten — Pure Rust ONNX Inference
**Repository:** https://github.com/robertknight/rten
**crates.io:** https://crates.io/crates/rten

A pure Rust ONNX inference engine focused on neural network operators. CPU performance benchmarks competitively with ONNX Runtime on many architectures (2025). Best suited for DL-based time series models (darts/tsai exports). Does **not** focus on `ai.onnx.ml` operators, so not a substitute for `ort` with sklearn-exported models.

---

## Alternative Approaches for Non-ONNX Models

### Prophet → augurs-prophet or Pre-Computed Forecasts

**Option A — augurs-prophet (recommended for native Rust inference)**

Fit the model once in Python using `augurs-prophet`'s Python bindings (which use the same Stan optimizer as the original Prophet), then serialize the fitted parameters to JSON. Load in Rust for inference:

```bash
pip install augurs
python -c "
from augurs.prophet import Prophet
import json
model = Prophet()
model.fit(df)
# Export Stan params
with open('prophet_params.json', 'w') as f:
    json.dump(model.params_to_dict(), f)
"
```

```rust
// Pure Rust Prophet inference from fitted parameters
use augurs::prophet::{Prophet, Params};
let params: Params = serde_json::from_str(&std::fs::read_to_string("prophet_params.json")?)?;
let prophet = Prophet::from_params(params, Default::default())?;
let forecast = prophet.predict(Some(&future_dates))?;
```

**Option B — Pre-computed forecast table (simplest for batch use cases)**

Run Prophet in Python on a schedule. Export `yhat`, `yhat_lower`, `yhat_upper` per timestamp to a Parquet or JSON file. The Rust service reads this as a lookup table. Anomaly detection is then: `|actual - yhat| > k * (yhat_upper - yhat)`.

```python
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_parquet("forecast.parquet")
```

**Option C — Manual reconstruction from JSON export**

Prophet's additive model can be reimplemented in ~150 lines of Rust. After fitting, call `model.to_json()` and extract:
- `k`, `m` — trend slope and offset
- `delta` — changepoint magnitudes (array)
- `changepoints_t` — changepoint positions in normalized time
- `beta` — Fourier series coefficients for seasonality
- `y_scale`, `t_scale` — normalization constants

The inference then computes:
```
trend(t) = (k + a(t)·delta) * t + (m + a(t)·(-s·delta))
seasonality(t) = X(t) · beta       // X = Fourier feature matrix
y(t) = trend(t) + seasonality(t)
```

---

### ARIMA / SARIMA → Parameter Extraction + Rust Inference

**Step 1: Extract parameters in Python**

```python
from statsmodels.tsa.arima.model import ARIMA
import json, numpy as np

result = ARIMA(series, order=(2, 1, 2)).fit()
params = {
    "ar_params": result.arparams.tolist(),
    "ma_params": result.maparams.tolist(),
    "d": 1,
    "mean": float(result.params.get("const", 0.0)),
    "sigma2": float(result.sigma2),
    "last_observations": series[-max(p, q+1):].tolist(),
    "last_residuals": result.resid[-max(p, q+1):].tolist(),
}
with open("arima_params.json", "w") as f:
    json.dump(params, f)
```

**Step 2: Implement inference in Rust**

```rust
use std::collections::VecDeque;

struct ArimaForecaster {
    ar_params: Vec<f64>,
    ma_params: Vec<f64>,
    mean: f64,
    observations: VecDeque<f64>,
    residuals: VecDeque<f64>,
}

impl ArimaForecaster {
    fn predict_next(&mut self) -> f64 {
        let p = self.ar_params.len();
        let q = self.ma_params.len();
        let mut yhat = self.mean;

        for (i, phi) in self.ar_params.iter().enumerate() {
            if let Some(y) = self.observations.iter().rev().nth(i) {
                yhat += phi * y;
            }
        }
        for (j, theta) in self.ma_params.iter().enumerate() {
            if let Some(eps) = self.residuals.iter().rev().nth(j) {
                yhat += theta * eps;
            }
        }
        yhat
    }

    fn update(&mut self, actual: f64) {
        let predicted = self.predict_next();
        let residual = actual - predicted;
        self.observations.push_back(actual);
        self.residuals.push_back(residual);
        if self.observations.len() > self.ar_params.len() { self.observations.pop_front(); }
        if self.residuals.len() > self.ma_params.len() { self.residuals.pop_front(); }
    }
}
```

For SARIMA or differenced (d > 0) series, maintain the last d+1 observations and undo differencing in the inference step.

> **Maintenance risk:** The Rust inference must replicate the exact Python preprocessing (differencing, scaling). Keep a regression test with known inputs/outputs from Python.

---

### EllipticEnvelope → Mahalanobis Distance in nalgebra

`EllipticEnvelope` fits a multivariate Gaussian and flags points whose Mahalanobis distance exceeds a threshold. It has no sklearn-onnx converter, but the inference is ~20 lines of Rust using `nalgebra`:

```python
# Python: extract parameters after fitting
import json, numpy as np
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.05).fit(X_train)
params = {
    "location": clf.location_.tolist(),          # mean vector
    "precision": clf.precision_.tolist(),        # inverse covariance matrix
    "threshold": float(clf.threshold_),          # decision boundary
    "offset": float(clf.offset_),
}
with open("elliptic_envelope.json", "w") as f:
    json.dump(params, f)
```

```rust
use nalgebra::{DMatrix, DVector};

struct EllipticEnvelope {
    location: DVector<f64>,
    precision: DMatrix<f64>,
    threshold: f64,
}

impl EllipticEnvelope {
    fn mahalanobis_sq(&self, x: &DVector<f64>) -> f64 {
        let diff = x - &self.location;
        (&diff.transpose() * &self.precision * &diff)[(0, 0)]
    }

    fn is_anomaly(&self, x: &[f64]) -> bool {
        let x_vec = DVector::from_vec(x.to_vec());
        self.mahalanobis_sq(&x_vec) > self.threshold
    }
}
```

---

### STUMPY / Matrix Profile → Pre-Computed Anomaly Scores

STUMPY is a pure algorithm — there is no model to export. The practical pattern:

1. **Offline (Python, cron job):** Run `stumpy.stump()` on recent history. Export the matrix profile (distances) and the reference motif set to Parquet.
2. **Online (Rust, per-request):** For a new incoming window, compute the z-normalized Euclidean distance to the nearest neighbor in the exported reference set using `ndarray`. Flag if distance > threshold.

For **streaming** anomaly detection with a matrix profile approach, use STUMPY's `stumpy.stumpi()` (incremental matrix profile). Run it in a Python sidecar that writes updated thresholds to a shared file or database row that Rust reads.

---

## Axum Integration Patterns

### Pattern 1: Sliding Window Streaming Anomaly Detection

Maintain a per-metric circular buffer. On each ingested data point, extract features from the window and score with the loaded ONNX model.

```rust
use std::{collections::VecDeque, sync::Arc};
use axum::{extract::State, Json, routing::post, Router};
use dashmap::DashMap;
use ort::{Session, inputs};
use ndarray::Array2;
use tokio::sync::RwLock;

#[derive(Clone)]
struct AppState {
    model: Arc<Session>,
    buffers: Arc<DashMap<String, VecDeque<f64>>>,
}

const WINDOW_SIZE: usize = 60;

fn extract_features(window: &VecDeque<f64>) -> Vec<f32> {
    let values: Vec<f64> = window.iter().copied().collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    // Add more features: percentiles, slope, autocorrelation lag-1, etc.
    vec![mean as f32, variance.sqrt() as f32, min as f32, max as f32, range as f32]
}

async fn ingest(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let metric_id = payload["metric"].as_str().unwrap_or("default").to_string();
    let value = payload["value"].as_f64().unwrap_or(0.0);

    let mut buffer = state.buffers.entry(metric_id).or_insert_with(VecDeque::new);
    buffer.push_back(value);
    if buffer.len() > WINDOW_SIZE { buffer.pop_front(); }

    if buffer.len() < WINDOW_SIZE {
        return Json(serde_json::json!({"status": "buffering", "n": buffer.len()}));
    }

    let features = extract_features(&buffer);
    let input = Array2::from_shape_vec((1, features.len()), features).unwrap();

    let outputs = state.model.run(inputs!["float_input" => input.view()].unwrap()).unwrap();
    let score: f32 = outputs["variable"].try_extract_tensor::<f32>().unwrap()[0];

    Json(serde_json::json!({
        "score": score,
        "is_anomaly": score < -0.05,
    }))
}
```

**Key design decisions:**
- Use `DashMap` (concurrent HashMap) to avoid a global `RwLock` on the buffers — each metric's buffer is independently locked.
- Feature extraction in Rust must exactly replicate the Python feature extraction used during training. Keep both implementations tested against the same golden dataset.
- Window size must match the `window_size` parameter used during model training.

---

### Pattern 2: Atomic Model Hot-Swap on Retraining

Periodically retrain in Python (Python sidecar, cron job, or CI pipeline). The Rust service monitors for a new model file and swaps it atomically without downtime:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use ort::Session;

type SharedModel = Arc<RwLock<Arc<Session>>>;

// Background task: watch for model updates
async fn model_watcher(model_ref: SharedModel, model_path: &str) {
    let mut last_modified = std::time::SystemTime::UNIX_EPOCH;
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        if let Ok(meta) = std::fs::metadata(model_path) {
            if let Ok(modified) = meta.modified() {
                if modified > last_modified {
                    match Session::builder().and_then(|b| b.commit_from_file(model_path)) {
                        Ok(new_session) => {
                            *model_ref.write().await = Arc::new(new_session);
                            last_modified = modified;
                            tracing::info!("Model hot-swapped successfully");
                        }
                        Err(e) => tracing::error!("Failed to load new model: {}", e),
                    }
                }
            }
        }
    }
}

// In handlers: take a read lock (non-blocking in the common case)
async fn predict(State(model_ref): State<SharedModel>, ...) {
    let model = model_ref.read().await.clone();  // Arc clone — cheap
    let outputs = model.run(...)?;
}
```

---

### Pattern 3: Threshold-Based vs. Model-Based Hybrid

For most production scenarios, combining a fast statistical check with a model-based scorer reduces both false positives and latency:

```rust
struct AnomalyDetector {
    // Statistical baseline (updated incrementally)
    ewma: f64,
    ewmstd: f64,
    alpha: f64,          // smoothing factor, e.g. 0.1
    sigma_threshold: f64, // e.g. 3.0

    // Model-based scorer (loaded ONNX session)
    model: Option<Arc<ort::Session>>,
    model_threshold: f32,
}

impl AnomalyDetector {
    fn update_and_score(&mut self, value: f64) -> AnomalyResult {
        // Gate 1: EWMA control chart (O(1), no allocation)
        let diff = value - self.ewma;
        self.ewma += self.alpha * diff;
        self.ewmstd = ((1.0 - self.alpha) * (self.ewmstd.powi(2) + self.alpha * diff.powi(2))).sqrt();
        let z_score = diff.abs() / (self.ewmstd + 1e-10);

        let statistical_anomaly = z_score > self.sigma_threshold;

        // Gate 2: model-based score (only when statistical check passes or borderline)
        let model_score = if statistical_anomaly || z_score > self.sigma_threshold * 0.7 {
            self.run_model_inference(value)
        } else {
            None
        };

        AnomalyResult {
            z_score,
            statistical_anomaly,
            model_score,
            is_anomaly: statistical_anomaly || model_score.map_or(false, |s| s < self.model_threshold),
        }
    }
}
```

**Statistical methods implementable in Rust without any library:**

| Method | Description | Best for |
|---|---|---|
| Rolling z-score | `(x - μ) / σ` on a fixed-size window | Approximately normal metrics |
| EWMA control chart | Exponentially weighted μ and σ | Non-stationary trends; quick adaptation |
| IQR / MAD | Median-based; robust to outliers | Heavy-tailed or skewed distributions |
| Seasonal decomposition | Subtract rolling seasonal mean before z-score | Strongly seasonal metrics |
| Welford's online variance | Incremental mean + variance in O(1) | Any streaming metric |

---

## Decision Matrix

| Use case | Train with | Serve / detect in Rust |
|---|---|---|
| **Anomaly detection, single metric, simple** | — | Rolling z-score or EWMA (pure Rust, ~10 lines) |
| **Anomaly detection, seasonal time series** | — | `anomaly_detection` crate (S-H-ESD) or `augurs-mstl` residuals |
| **Anomaly detection, multivariate features** | Python: `IsolationForest` → sklearn-onnx (`zipmap=False`) | `ort` in Axum (async, zero-copy) |
| **Anomaly detection, multivariate, no C++ deps** | Python: `IsolationForest` → sklearn-onnx (`zipmap=False`) | `tract` in Axum (test compatibility first) |
| **Outlier detection across N time series** | — | `augurs-outlier` (DBSCAN or MAD across series group) |
| **Short-term forecasting + confidence bands** | — | `augurs-mstl` + `augurs-ets` (pure Rust) |
| **Prophet-equivalent forecasting, native** | Fit via augurs-prophet Python bindings or CmdStan | `augurs-prophet` in Axum |
| **ARIMA forecasting, native** | Python `statsmodels` → JSON params | Rust AR recursion or `scirs2-series` (alpha) |
| **Changepoint detection** | — | `augurs-changepoint` (BOCPD, pure Rust) |
| **Deep learning time series (DL features needed)** | Python: darts/tsai → `torch.onnx.export` | `ort` or `rten` in Axum |
| **Full Rust, no Python anywhere** | `augurs` suite for forecasting; `anomaly_detection` crate for anomaly | Direct in Axum handler |
| **Unusual pattern detection (motifs/discords)** | Python: STUMPY → pre-compute scores | Read scores from cache/file in Axum |

---

## SaaS Alternatives

The following hosted services provide anomaly detection as a managed product. They are relevant if you want to avoid building and maintaining the ML pipeline described above, at the cost of vendor dependency and data egress.

---

### Amazon CloudWatch Anomaly Detection

**What it is:** Per-metric anomaly detection on CloudWatch metrics using a proprietary ML model (algorithm not disclosed) that automatically accounts for hourly, daily, and weekly seasonality.

**Integration:** Enable per-metric in the AWS Console, CLI, or CloudFormation. Use `ANOMALY_DETECTION_BAND` in metric math. Create CloudWatch Alarms that fire when a metric exits the predicted band.

**Pricing:** $0.10 per metric per month for anomaly detection; each resulting alarm consumes 3 standard metric units (~$0.30/month per alarm in US-East).

**Pros:**
- Zero setup; deeply integrated with all AWS services.
- Handles seasonality and trend changes automatically.
- Fires CloudWatch Alarms natively — integrates with SNS, Lambda, PagerDuty.

**Cons:**
- Black-box algorithm — no customization of sensitivity or model type.
- Only applies to CloudWatch metrics; custom application metrics must be emitted as CloudWatch custom metrics first (additional cost: $0.30/metric/month for custom metrics).
- Cannot export the model or predictions programmatically beyond the band values.
- No multivariate or cross-metric correlation.

---

### ~~AWS Lookout for Metrics~~

> **Status: End-of-life October 9, 2025.** Do not use for new projects. Migrate to CloudWatch Anomaly Detection.

---

### Datadog Anomaly Detection

**What it is:** A function applied to any Datadog metric in monitors and dashboards. Provides three algorithms:
- **Basic:** Rolling quantile bounds; no seasonality; adapts quickly to level shifts.
- **Agile:** Seasonal ARIMA (day-of-week + time-of-day). Adjusts reasonably quickly to intentional level shifts.
- **Robust:** Seasonal decomposition with very stable bands. Does not adjust to intentional level shifts quickly.

**Integration:** All metrics must be ingested into Datadog (agent, API, or Prometheus remote write). Configure anomaly monitors via the Datadog Monitor UI or Terraform provider. An AIOps layer correlates anomalies from multiple monitors into incidents.

**Pricing:** No separate line item; included in Datadog Infrastructure and APM subscriptions. Datadog pricing is consumption-based and contract-driven at scale. At small scale (~10 hosts), expect $15–$30/host/month; costs scale quickly.

**Pros:**
- Turnkey; excellent visualization and alerting.
- Correlates anomalies with traces and logs via AIOps.
- Three algorithm choices give some customization.

**Cons:**
- All metric data must be sent to Datadog; significant cost at scale.
- No control over model internals or custom features.
- Strong vendor lock-in; difficult to migrate away.
- Anomaly bands are not accessible programmatically beyond alert triggers.

---

### ~~Azure Anomaly Detector~~

> **Status: Retiring October 1, 2026.** New resource creation disabled September 20, 2023. Do not start new projects. Migrate to Azure AI Foundry or a custom solution.

At peak (2021–2023), Azure Anomaly Detector provided a well-designed REST API using SR-CNN (Spectral Residual + CNN) for univariate detection and a graph attention network for multivariate (MVAD). It was genuinely best-in-class for its API design.

---

### Grafana Cloud ML + Sift

**What it is:** Grafana Cloud's built-in ML features, powered internally by the `augurs` Rust library:
- **Metric forecasting:** Trains models on Prometheus/Grafana metrics and provides predicted bands for dynamic alerting thresholds.
- **Outlier detection:** Identifies which time series in a group deviates from cohort behavior.
- **Sift:** Automated incident investigation assistant that runs heuristic + ML checks across metrics, logs, and traces when an incident is declared.

**Integration:** Native to Grafana Cloud; configured via the Grafana ML App plugin or API, pointing at existing data sources (Prometheus, Loki, Tempo).

**Pricing:** **Included at no additional cost** across all Grafana Cloud tiers.
- Free: 10k active series, 50 GB logs/traces per month.
- Pro: $19/month platform fee + usage overages.

**Pros:**
- Included in Grafana Cloud cost; effectively free.
- Sift provides automated root cause analysis — valuable for operational metrics use cases.
- Powered by `augurs` (the same Rust library recommended for self-hosted solutions).

**Cons:**
- Requires Grafana Cloud (not self-hosted Grafana).
- Limited to metrics already in Grafana's data sources.
- No custom model training or algorithm selection.
- Less flexible than a custom solution for application-level anomaly detection on arbitrary data.

---

### Elastic (Kibana) ML

**What it is:** X-Pack ML provides unsupervised time series anomaly detection directly on Elasticsearch data. Uses a proprietary temporal Bayesian normalization algorithm. Job types: single metric, multi-metric, population (per-entity analysis), and advanced.

**Integration:** Data must be in Elasticsearch. Configure anomaly detection jobs via the Kibana ML UI or Elastic REST API. Results are written to `.ml-anomalies-*` indices and visualized in Kibana Anomaly Explorer. Anomaly scores (0–100) trigger Kibana Alerts.

**Pricing:** ML features are available at **Platinum tier and above**.
- Platinum: ~$125–131/month (baseline managed cluster pricing, scales with usage).
- ML is not available on the Basic or Standard tiers.

**Pros:**
- Zero-code anomaly detection if data is already in Elasticsearch.
- Population-level analysis (per-user, per-host anomaly scoring) is powerful and not easily replicated with simple models.
- Good visualization and Kibana integration.

**Cons:**
- Requires Platinum subscription; significant incremental cost if not already on Platinum.
- Data must be in Elasticsearch — not suitable for arbitrary application data without a pipeline.
- Algorithm is proprietary and not configurable.
- No model export or portability.

---

### Anodot

**What it is:** A purpose-built anomaly detection and business monitoring platform. Uses a proprietary ensemble of 30+ ML models that learn normal behavior for each metric autonomously. Specializes in real-time detection and incident correlation across business metrics (revenue, conversion, performance).

**Integration:** Push API (REST) for custom metrics; pre-built connectors for AWS, GCP, Azure, Datadog, Prometheus.

**Pricing:** Custom enterprise pricing; not publicly listed.

**Pros:**
- Strong incident correlation — groups related anomalies into business incidents.
- Purpose-built for business metrics; good at capturing business impact, not just technical spikes.
- Real-time detection with claimed 95% catch rate.

**Cons:**
- Opaque pricing — typically expensive at enterprise scale.
- Black-box models; no transparency into algorithm selection or tuning.
- Requires sending all metric data to Anodot's cloud.

---

### New Relic AI / Intelligent Alerting

**What it is:** New Relic's "Applied Intelligence" layer applies ML-based anomaly detection to APM, infrastructure, browser, and custom metrics. Feeds into alert correlation and incident intelligence for noise reduction.

**Integration:** 780+ integrations; data via New Relic agent, Prometheus, OpenTelemetry, or REST API.

**Pricing:** Consumption-based ($0.35/GB ingested after free tier). Free tier: 100 GB/month ingest. Anomaly detection is included in the platform.

**Pros:**
- Broadest integration ecosystem of any observability platform.
- Anomaly detection is free within existing ingest costs.
- Good correlation across APM + infrastructure + logs.

**Cons:**
- Anomaly detection is a secondary feature, not the core product — less sophisticated than dedicated tools.
- Limited algorithmic customization.
- At moderate ingest volumes, cost grows faster than specialized alternatives.

---

### SaaS vs. Self-Hosted: Key Trade-offs

| Dimension | SaaS | Self-hosted (this doc's approach) |
|---|---|---|
| **Setup time** | Hours to days | Days to weeks |
| **Maintenance** | Vendor-managed | Your team |
| **Algorithm control** | None / limited | Full |
| **Custom features** | Not possible | Any feature you can engineer |
| **Data privacy** | Data leaves your infrastructure | Data stays in-house |
| **Cost at low scale** | Often cheap/free | Engineering time |
| **Cost at high scale** | Expensive (per metric/per GB) | Fixed (compute only) |
| **Multivariate anomalies** | Limited (Datadog/Elastic only) | Full (IsolationForest, MVAD reimpl.) |
| **Model portability** | Zero | Full |
| **Cold start** | None | Needs historical data for training |
| **Seasonal adaptation** | Automatic | Manual retraining schedule needed |

**Recommendation:** Use SaaS for operational/infrastructure metrics (Grafana Cloud ML is free; CloudWatch Anomaly Detection is $0.10/metric). Build the self-hosted pipeline described in this document for **application-domain anomaly detection** — detecting fraud, user behavior anomalies, data quality issues, or business metric anomalies where you need custom features, full algorithm control, and data privacy.
