# Vulnerability Detection & Feature Extraction — CVE/SEC Domain

A thorough guide to building ML models for common vulnerability detection, security scanning, and compromised package identification. Covers dataset acquisition from NVD/OSV, deep CVSS vector parsing, feature engineering, exploratory data analysis (EDA) to identify the most important variables, model training in Python, and serving predictions through an Axum Rust backend.

> **Last updated:** February 17, 2026

---

## Table of Contents

- [Problem Framing: What Are We Predicting?](#problem-framing-what-are-we-predicting)
- [Data Sources](#data-sources)
  - [NVD REST API v2](#nvd-rest-api-v2)
  - [OSV — Open Source Vulnerabilities](#osv--open-source-vulnerabilities)
  - [EPSS — Exploit Prediction Scoring System](#epss--exploit-prediction-scoring-system)
  - [Kaggle / HuggingFace CVE Datasets](#kaggle--huggingface-cve-datasets)
- [Raw CVE Record Structure](#raw-cve-record-structure)
- [CVSS Vector String Deep Dive](#cvss-vector-string-deep-dive)
  - [CVSS v3.1 Metrics](#cvss-v31-metrics)
  - [CVSS v4.0 Metrics](#cvss-v40-metrics)
  - [Parsing the Vector String in Python](#parsing-the-vector-string-in-python)
- [Feature Engineering](#feature-engineering)
  - [Structured Features from CVSS](#structured-features-from-cvss)
  - [CWE Features](#cwe-features)
  - [Text Features from Description](#text-features-from-description)
  - [Temporal Features](#temporal-features)
  - [Reference and Metadata Features](#reference-and-metadata-features)
  - [Package / CPE Scope Features](#package--cpe-scope-features)
  - [Full Feature Pipeline](#full-feature-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Target Variable: Severity Distribution](#target-variable-severity-distribution)
  - [CVSS Metric Distributions](#cvss-metric-distributions)
  - [CWE Category Analysis](#cwe-category-analysis)
  - [Score Distributions Over Time](#score-distributions-over-time)
  - [Correlation Heatmap](#correlation-heatmap)
  - [Description Length vs Severity](#description-length-vs-severity)
  - [Attack Vector vs Exploit Rate](#attack-vector-vs-exploit-rate)
- [Feature Importance Analysis](#feature-importance-analysis)
  - [Tree-Based Importance (XGBoost / RandomForest)](#tree-based-importance-xgboost--randomforest)
  - [SHAP Values](#shap-values)
  - [Key Findings: Most Predictive Features](#key-findings-most-predictive-features)
- [ML Tasks and Model Training](#ml-tasks-and-model-training)
  - [Task 1 — Severity Classification (CVSS Score Bucket)](#task-1--severity-classification-cvss-score-bucket)
  - [Task 2 — Exploit Probability Prediction (EPSS Regression)](#task-2--exploit-probability-prediction-epss-regression)
  - [Task 3 — Compromised Package Detection (Binary Classification)](#task-3--compromised-package-detection-binary-classification)
  - [Task 4 — CWE Category Prediction from Description (NLP)](#task-4--cwe-category-prediction-from-description-nlp)
- [ONNX Export and Rust Serving](#onnx-export-and-rust-serving)
- [Decision Matrix](#decision-matrix)

---

## Problem Framing: What Are We Predicting?

The security domain presents several distinct ML tasks, each with different feature sets and label sources:

| Task | Input | Label | Practical Use |
|---|---|---|---|
| **Severity classification** | CVE description + metadata | CVSS score bucket (None/Low/Medium/High/Critical) | Triage automation — auto-route to correct team |
| **Exploit probability** | CVE features + CVSS metrics | EPSS probability (0–1) | Patch prioritization — which CVEs are actively exploited |
| **Package vulnerability scan** | Package name + version + dependency tree | CVE list / clean flag | CI/CD gate — block deployments with known CVEs |
| **CWE prediction** | CVE description (free text) | CWE-ID (e.g., CWE-79, CWE-89) | Enrich incomplete records; assist static analysis |
| **Compromised package detection** | Package manifest + registry metadata | Binary (clean / compromised) | Supply chain security — detect typosquatted or backdoored packages |

This document focuses primarily on the CVE dataset tasks (severity, exploit prediction, CWE tagging) and the package scanning task, since they share infrastructure and pipeline stages.

---

## Data Sources

### NVD REST API v2

The National Vulnerability Database (NIST) exposes a public REST API. No authentication is needed for basic queries; an API key raises the rate limit from 5 to 50 requests per rolling 30 seconds.

```python
import requests, time

NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
HEADERS = {"apiKey": "YOUR_NVD_API_KEY"}   # omit for unauthenticated (slower)

def fetch_nvd_page(start_index: int, results_per_page: int = 2000) -> dict:
    params = {
        "startIndex": start_index,
        "resultsPerPage": results_per_page,
    }
    resp = requests.get(NVD_BASE, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_all_cves() -> list[dict]:
    """Full NVD dump — ~230k+ records as of 2026."""
    records = []
    data = fetch_nvd_page(0)
    total = data["totalResults"]
    records.extend(data["vulnerabilities"])

    start = len(records)
    while start < total:
        time.sleep(0.6)   # stay within rate limit (unauthenticated: 1 req/6s)
        page = fetch_nvd_page(start)
        batch = page["vulnerabilities"]
        if not batch:
            break
        records.extend(batch)
        start += len(batch)
        print(f"Fetched {start}/{total}")

    return records
```

**Incremental updates** — use `lastModStartDate` and `lastModEndDate` to pull only records modified since your last sync:

```python
from datetime import datetime, timedelta

def fetch_recent_cves(since_hours: int = 24) -> list[dict]:
    since = (datetime.utcnow() - timedelta(hours=since_hours)).strftime(
        "%Y-%m-%dT%H:%M:%S.000"
    )
    params = {
        "lastModStartDate": since,
        "lastModEndDate": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": 2000,
    }
    resp = requests.get(NVD_BASE, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["vulnerabilities"]
```

---

### OSV — Open Source Vulnerabilities

OSV (`https://osv.dev`) provides a package-centric vulnerability database that maps CVEs and GHSA advisories to exact package version ranges. It is the authoritative source for the **compromised package** detection task.

```python
import requests

OSV_QUERY = "https://api.osv.dev/v1/query"

def check_package_vulns(ecosystem: str, package: str, version: str) -> list[dict]:
    """
    ecosystem: 'PyPI', 'npm', 'crates.io', 'Maven', 'Go', 'Hex', ...
    Returns list of OSV advisory objects affecting this exact version.
    """
    body = {
        "version": version,
        "package": {"name": package, "ecosystem": ecosystem},
    }
    resp = requests.post(OSV_QUERY, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json().get("vulns", [])

# Example: check requests 2.28.0 on PyPI
vulns = check_package_vulns("PyPI", "requests", "2.28.0")
print(f"Found {len(vulns)} known vulnerabilities")
```

**Bulk dataset download** — OSV provides full dumps per ecosystem at `https://osv-vulnerabilities.storage.googleapis.com/{ECOSYSTEM}/all.zip`.

---

### EPSS — Exploit Prediction Scoring System

EPSS (by FIRST.org) assigns each CVE a daily-updated probability that it will be exploited in the wild within the next 30 days. It is the most important external signal for exploit prioritization tasks.

```python
import requests, pandas as pd

def fetch_epss_scores() -> pd.DataFrame:
    """Download the full EPSS dataset (~230k rows, ~15 MB CSV)."""
    url = "https://epss.cyentia.com/epss_scores-current.csv.gz"
    df = pd.read_csv(url, compression="gzip", skiprows=1)
    # Columns: cve, epss, percentile
    df.columns = ["cve_id", "epss_score", "epss_percentile"]
    df["epss_score"] = df["epss_score"].astype(float)
    df["epss_percentile"] = df["epss_percentile"].astype(float)
    return df

epss_df = fetch_epss_scores()
# epss_score: float 0–1 (probability of exploitation within 30 days)
# epss_percentile: float 0–1 (rank among all CVEs — more stable for comparisons)
```

> EPSS scores are the ground-truth label for the exploit prediction task and also one of the highest-signal features when available. When training with it as a feature (not label), use a lagged version to avoid data leakage.

---

### Kaggle / HuggingFace CVE Datasets

For fast iteration without writing a crawler:

| Dataset | Records | Notes |
|---|---|---|
| `manavkhambhayata/cve-2024-database-exploits-cvss-os` (Kaggle) | ~26k 2024 CVEs | Pre-parsed CVSS fields, exploit flag, OS breakdown |
| `andrewkronser/cve-common-vulnerabilities-and-exposures` (Kaggle) | ~170k historical | CVSS v2 + v3 metrics pre-flattened |
| `stasvinokur/cve-and-cwe-dataset-1999-2025` (HuggingFace) | ~250k | Includes CWE linkage, very clean |

```python
# HuggingFace dataset — requires: pip install datasets
from datasets import load_dataset

ds = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025", split="train")
df = ds.to_pandas()
print(df.columns.tolist())
# ['cve_id', 'description', 'cvss_score', 'cvss_vector', 'cwe_id', 'published_date', ...]
```

---

## Raw CVE Record Structure

Understanding the raw NVD JSON structure is essential before any feature extraction. A minimal annotated example:

```json
{
  "cve": {
    "id": "CVE-2024-12345",
    "sourceIdentifier": "cve@mitre.org",
    "published": "2024-03-15T14:23:00.000",
    "lastModified": "2024-03-20T09:00:00.000",
    "vulnStatus": "Analyzed",
    "descriptions": [
      {
        "lang": "en",
        "value": "A heap buffer overflow in libfoo before 1.2.3 allows remote
                  attackers to execute arbitrary code via a crafted HTTP request."
      }
    ],
    "metrics": {
      "cvssMetricV31": [{
        "source": "nvd@nist.gov",
        "type": "Primary",
        "cvssData": {
          "version": "3.1",
          "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
          "attackVector": "NETWORK",
          "attackComplexity": "LOW",
          "privilegesRequired": "NONE",
          "userInteraction": "NONE",
          "scope": "UNCHANGED",
          "confidentialityImpact": "HIGH",
          "integrityImpact": "HIGH",
          "availabilityImpact": "HIGH",
          "baseScore": 9.8,
          "baseSeverity": "CRITICAL"
        },
        "exploitabilityScore": 3.9,
        "impactScore": 5.9
      }]
    },
    "weaknesses": [
      { "source": "nvd@nist.gov", "type": "Primary",
        "description": [{ "lang": "en", "value": "CWE-122" }] }
    ],
    "configurations": [
      {
        "nodes": [{
          "operator": "OR",
          "cpeMatch": [{
            "vulnerable": true,
            "criteria": "cpe:2.3:a:vendor:libfoo:*:*:*:*:*:*:*:*",
            "versionEndExcluding": "1.2.3"
          }]
        }]
      }
    ],
    "references": [
      { "url": "https://github.com/vendor/libfoo/security/advisories/GHSA-xxxx",
        "source": "cve@mitre.org", "tags": ["Vendor Advisory"] },
      { "url": "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
        "source": "nvd@nist.gov", "tags": ["Third Party Advisory"] }
    ]
  }
}
```

**Fields that are most useful for ML:**

| Field Path | Type | Notes |
|---|---|---|
| `cve.id` | string | Identifier; derive year from it |
| `cve.published` | ISO datetime | Age of CVE; publication year |
| `cve.lastModified` | ISO datetime | Staleness signal |
| `cve.vulnStatus` | enum | `Analyzed`, `Modified`, `Awaiting Analysis`, `Rejected` — signals completeness |
| `cve.descriptions[lang=en].value` | free text | Primary NLP input |
| `cvssMetricV31[type=Primary].cvssData.*` | structured | 8 metrics + 2 sub-scores + base score |
| `cvssMetricV31[type=Primary].vectorString` | string | Compact encoding of all metrics |
| `weaknesses[type=Primary].description[lang=en].value` | string (CWE-ID) | Category feature |
| `configurations[].nodes[].cpeMatch[].criteria` | CPE string | Vendor, product, version scope |
| `references[].tags` | list[string] | `Exploit`, `Patch`, `Vendor Advisory`, `Third Party Advisory` |

---

## CVSS Vector String Deep Dive

The CVSS vector string is the single most information-dense field in a CVE record. It encodes all exploitability and impact metrics in a compact, parseable format. Understanding it is the foundation of feature engineering for this domain.

### CVSS v3.1 Metrics

Format: `CVSS:3.1/AV:{}/AC:{}/PR:{}/UI:{}/S:{}/C:{}/I:{}/A:{}`

| Metric | Abbreviation | Values | What It Measures |
|---|---|---|---|
| Attack Vector | AV | N (Network) / A (Adjacent) / L (Local) / P (Physical) | How far away can an attacker be? Network = remotely exploitable over internet |
| Attack Complexity | AC | L (Low) / H (High) | Are there conditions beyond the attacker's control that must exist? |
| Privileges Required | PR | N (None) / L (Low) / H (High) | What privileges must the attacker already have? |
| User Interaction | UI | N (None) / R (Required) | Does a user (other than the attacker) need to take action? |
| Scope | S | U (Unchanged) / C (Changed) | Can the vulnerability impact resources beyond its authorization scope? |
| Confidentiality Impact | C | N (None) / L (Low) / H (High) | Impact to confidentiality of the vulnerable system |
| Integrity Impact | I | N (None) / L (Low) / H (High) | Impact to integrity |
| Availability Impact | A | N (None) / L (Low) / H (High) | Impact to availability (denial of service) |

The **exploitabilityScore** (0–3.9) and **impactScore** (0–6.0) are derived sub-scores. The **baseScore** (0–10) is the final composite. NVD maps this to severity buckets: None (0.0), Low (0.1–3.9), Medium (4.0–6.9), High (7.0–8.9), Critical (9.0–10.0).

---

### CVSS v4.0 Metrics

CVSS v4.0 (released October 2023) introduces significant structural changes. The vector format is: `CVSS:4.0/AV:{}/AC:{}/AT:{}/PR:{}/UI:{}/VC:{}/VI:{}/VA:{}/SC:{}/SI:{}/SA:{}`

**Key changes from v3.1:**

| Change | v3.1 | v4.0 |
|---|---|---|
| Scope → split into two systems | Scope (S: U/C) | Vulnerable System (VC/VI/VA) + Subsequent System (SC/SI/SA) |
| Attack Requirements added | — | AT: None / Present |
| User Interaction values extended | N / R | N (None) / P (Passive) / A (Active) |
| Threat metric replaces Temporal | Temporal group | Exploit Maturity (E): X / A / P / U |
| Supplemental metrics added | — | Safety (S), Automatable (AU), Recovery (R), Value Density (V), Provider Urgency (U), Response Effort (RE) |

**Full v4.0 Base Metric Table:**

| Group | Metric | Abbr | Values |
|---|---|---|---|
| Exploitability | Attack Vector | AV | N / A / L / P |
| Exploitability | Attack Complexity | AC | L / H |
| Exploitability | Attack Requirements | AT | N / P |
| Exploitability | Privileges Required | PR | N / L / H |
| Exploitability | User Interaction | UI | N / P / A |
| Vulnerable System Impact | Confidentiality | VC | N / L / H |
| Vulnerable System Impact | Integrity | VI | N / L / H |
| Vulnerable System Impact | Availability | VA | N / L / H |
| Subsequent System Impact | Confidentiality | SC | N / L / H |
| Subsequent System Impact | Integrity | SI | N / L / H |
| Subsequent System Impact | Availability | SA | N / L / H |

**Threat Metrics (optional, modifies final score):**

| Metric | Abbr | Values | Meaning |
|---|---|---|---|
| Exploit Maturity | E | X / A / P / U | X=Not Defined, A=Attacked (exploited in wild), P=PoC available, U=No evidence |

**Supplemental Metrics (informational, do not change score):**

| Metric | Abbr | Values |
|---|---|---|
| Safety | S | X / N / P |
| Automatable | AU | X / N / Y |
| Provider Urgency | U | X / Clear / Green / Amber / Red |
| Recovery | R | X / A / U / I |
| Value Density | V | X / D / C |
| Vulnerability Response Effort | RE | X / L / M / H |

The **Automatable** (AU) metric is particularly valuable for ML: `Y` means an attacker can automate all MITRE ATT&CK kill chain steps — this is a strong predictor of rapid exploitation.

---

### Parsing the Vector String in Python

```python
import re
from dataclasses import dataclass, field

# --- CVSS v3.1 parser ---

CVSS31_METRIC_MAP = {
    "AV": {"N": 0, "A": 1, "L": 2, "P": 3},
    "AC": {"L": 0, "H": 1},
    "PR": {"N": 0, "L": 1, "H": 2},
    "UI": {"N": 0, "R": 1},
    "S":  {"U": 0, "C": 1},
    "C":  {"N": 0, "L": 1, "H": 2},
    "I":  {"N": 0, "L": 1, "H": 2},
    "A":  {"N": 0, "L": 1, "H": 2},
}

def parse_cvss31_vector(vector: str) -> dict:
    """
    Input:  'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'
    Output: {'AV': 'N', 'AC': 'L', ..., 'AV_ord': 0, 'AC_ord': 0, ...}
    """
    if not vector or not vector.startswith("CVSS:3."):
        return {}

    metrics = {}
    for part in vector.split("/")[1:]:   # skip 'CVSS:3.1' prefix
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        metrics[key] = val
        if key in CVSS31_METRIC_MAP and val in CVSS31_METRIC_MAP[key]:
            metrics[f"{key}_ord"] = CVSS31_METRIC_MAP[key][val]

    return metrics


# --- CVSS v4.0 parser ---

CVSS40_BASE_METRICS = ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"]

CVSS40_METRIC_MAP = {
    "AV":  {"N": 0, "A": 1, "L": 2, "P": 3},
    "AC":  {"L": 0, "H": 1},
    "AT":  {"N": 0, "P": 1},
    "PR":  {"N": 0, "L": 1, "H": 2},
    "UI":  {"N": 0, "P": 1, "A": 2},
    "VC":  {"N": 0, "L": 1, "H": 2},
    "VI":  {"N": 0, "L": 1, "H": 2},
    "VA":  {"N": 0, "L": 1, "H": 2},
    "SC":  {"N": 0, "L": 1, "H": 2},
    "SI":  {"N": 0, "L": 1, "H": 2},
    "SA":  {"N": 0, "L": 1, "H": 2},
    # Threat
    "E":   {"X": -1, "U": 0, "P": 1, "A": 2},
    # Supplemental
    "AU":  {"X": -1, "N": 0, "Y": 1},
    "R":   {"X": -1, "A": 0, "U": 1, "I": 2},
    "V":   {"X": -1, "D": 0, "C": 1},
    "RE":  {"X": -1, "L": 0, "M": 1, "H": 2},
}

def parse_cvss40_vector(vector: str) -> dict:
    """Parses a CVSS v4.0 vector string into a flat feature dict."""
    if not vector or not vector.startswith("CVSS:4.0"):
        return {}

    metrics = {}
    for part in vector.split("/")[1:]:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        metrics[key] = val
        if key in CVSS40_METRIC_MAP and val in CVSS40_METRIC_MAP[key]:
            metrics[f"{key}_ord"] = CVSS40_METRIC_MAP[key][val]

    return metrics


# --- Unified extractor that handles both v3.1 and v4.0 ---

def extract_cvss_features(nvd_metrics: dict) -> dict:
    """
    nvd_metrics: the 'metrics' block from one NVD CVE record.
    Returns a flat feature dict with prefixed keys.
    """
    features = {}

    # Prefer v4.0 if present, fall back to v3.1, then v2
    if "cvssMetricV40" in nvd_metrics:
        raw = nvd_metrics["cvssMetricV40"]
        source = next((m for m in raw if m["type"] == "Primary"), raw[0])
        parsed = parse_cvss40_vector(source["cvssData"]["vectorString"])
        features.update({f"v4_{k}": v for k, v in parsed.items()})
        features["cvss_version"] = 4
        features["base_score"] = source["cvssData"].get("baseScore")
        features["base_severity"] = source["cvssData"].get("baseSeverity", "")

    elif "cvssMetricV31" in nvd_metrics:
        raw = nvd_metrics["cvssMetricV31"]
        source = next((m for m in raw if m["type"] == "Primary"), raw[0])
        parsed = parse_cvss31_vector(source["cvssData"]["vectorString"])
        features.update({f"v3_{k}": v for k, v in parsed.items()})
        features["cvss_version"] = 3
        features["base_score"] = source["cvssData"].get("baseScore")
        features["base_severity"] = source["cvssData"].get("baseSeverity", "")
        features["exploitability_score"] = source.get("exploitabilityScore")
        features["impact_score"] = source.get("impactScore")

    elif "cvssMetricV2" in nvd_metrics:
        raw = nvd_metrics["cvssMetricV2"]
        source = raw[0]
        features["cvss_version"] = 2
        features["base_score"] = source["cvssData"].get("baseScore")
        features["exploitability_score"] = source.get("exploitabilityScore")
        features["impact_score"] = source.get("impactScore")
        features["v2_acInsufInfo"] = int(source.get("acInsufInfo", False))
        features["v2_obtainAllPrivilege"] = int(source.get("obtainAllPrivilege", False))
        features["v2_obtainUserPrivilege"] = int(source.get("obtainUserPrivilege", False))
        features["v2_userInteractionRequired"] = int(source.get("userInteractionRequired", False))

    return features
```

---

## Feature Engineering

### Structured Features from CVSS

The parsed CVSS metrics feed directly into the model as ordinal-encoded integers. The encoding scheme above (`_ord` suffix) converts categorical levels into meaningful numeric scales. The ordering is intentional — for Attack Vector, `N=0` (most severe / remotely exploitable) is not the natural sort; you may want to reverse it to `N=3` for models that benefit from monotonic relationships:

```python
# Reversed severity ordering for AV: Network is most dangerous = highest value
AV_SEVERITY = {"P": 0, "L": 1, "A": 2, "N": 3}
IMPACT_SEVERITY = {"N": 0, "L": 1, "H": 2}

def cvss_features_for_model(parsed: dict, prefix: str = "v3_") -> dict:
    """
    Convert parsed CVSS dict to model-ready features.
    All features are integers. Missing = -1 (sentinel for imputation).
    """
    p = lambda key: parsed.get(f"{prefix}{key}_ord", -1)
    return {
        "av_severity":   AV_SEVERITY.get(parsed.get(f"{prefix}AV", ""), -1),
        "ac_ordinal":    p("AC"),    # 0=Low(easy), 1=High(hard)
        "pr_ordinal":    p("PR"),    # 0=None, 1=Low, 2=High
        "ui_ordinal":    p("UI"),    # 0=None, 1=Required (v3) or Passive/Active (v4)
        "scope_changed": p("S"),     # v3.1 only: 0=Unchanged, 1=Changed
        "c_impact":      p("C"),     # confidentiality
        "i_impact":      p("I"),     # integrity
        "a_impact":      p("A"),     # availability
        # Composite impact: max of CIA for quick approximation
        "max_impact":    max(
            IMPACT_SEVERITY.get(parsed.get(f"{prefix}C", "N"), 0),
            IMPACT_SEVERITY.get(parsed.get(f"{prefix}I", "N"), 0),
            IMPACT_SEVERITY.get(parsed.get(f"{prefix}A", "N"), 0),
        ),
        # RCE proxy: Network AV + None PR + High CIA — strong indicator
        "rce_proxy": int(
            parsed.get(f"{prefix}AV") == "N"
            and parsed.get(f"{prefix}PR") == "N"
            and parsed.get(f"{prefix}C") == "H"
            and parsed.get(f"{prefix}I") == "H"
        ),
    }
```

---

### CWE Features

CWE-IDs encode the root cause weakness class. They are hierarchical — CWE-89 (SQL Injection) is a child of CWE-943 (Improper Neutralization). For ML, use two encodings:

```python
import pandas as pd

# Top-level CWE categories with known exploit/severity associations
HIGH_SEVERITY_CWES = {
    "CWE-787",   # Out-of-bounds Write — often RCE
    "CWE-122",   # Heap Buffer Overflow
    "CWE-416",   # Use After Free
    "CWE-78",    # OS Command Injection
    "CWE-502",   # Deserialization of Untrusted Data
    "CWE-94",    # Code Injection
}

HIGH_EXPLOIT_CWES = {
    "CWE-89",    # SQL Injection
    "CWE-79",    # Cross-site Scripting
    "CWE-20",    # Improper Input Validation
    "CWE-22",    # Path Traversal
    "CWE-306",   # Missing Authentication
    "CWE-287",   # Improper Authentication
}

MEMORY_SAFETY_CWES = {"CWE-787", "CWE-125", "CWE-122", "CWE-416", "CWE-119", "CWE-190"}

def extract_cwe_features(weaknesses: list[dict]) -> dict:
    primary_cwes = []
    for w in weaknesses:
        if w.get("type") == "Primary":
            for d in w.get("description", []):
                if d["lang"] == "en" and d["value"].startswith("CWE-"):
                    primary_cwes.append(d["value"])

    if not primary_cwes:
        return {
            "cwe_id": "CWE-UNKNOWN",
            "cwe_numeric": -1,
            "is_high_severity_cwe": 0,
            "is_high_exploit_cwe": 0,
            "is_memory_safety_cwe": 0,
            "cwe_count": 0,
        }

    primary = primary_cwes[0]
    numeric = int(primary.replace("CWE-", "")) if primary != "CWE-noinfo" else -1

    return {
        "cwe_id": primary,
        "cwe_numeric": numeric,
        "is_high_severity_cwe": int(primary in HIGH_SEVERITY_CWES),
        "is_high_exploit_cwe": int(primary in HIGH_EXPLOIT_CWES),
        "is_memory_safety_cwe": int(primary in MEMORY_SAFETY_CWES),
        "cwe_count": len(primary_cwes),
    }
```

For tree-based models, use label-encoded CWE IDs. For neural models, embed CWE into a learned categorical embedding.

---

### Text Features from Description

The English CVE description is the richest free-text signal. Beyond raw NLP, security-specific keyword patterns carry high predictive signal:

```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Keyword groups with known severity/exploit associations
EXPLOIT_KEYWORDS = [
    r"\bremote code execution\b", r"\bRCE\b", r"\barbitrary code\b",
    r"\bprivilege escalation\b", r"\broot\b.*\bexecute\b",
    r"\bshell\b", r"\bbackdoor\b",
]
DOS_KEYWORDS = [
    r"\bdenial.of.service\b", r"\bDoS\b", r"\bcrash\b", r"\bexhaust\b",
    r"\bloop\b.*\binfinite\b",
]
AUTH_BYPASS_KEYWORDS = [
    r"\bbypass\b.*\bauth", r"\bunauthenticated\b", r"\bwithout.*authentication\b",
    r"\bno.*credentials\b",
]
INFO_DISCLOSURE_KEYWORDS = [
    r"\binformation disclosure\b", r"\bsensitive.*data\b", r"\bpassword.*exposed\b",
    r"\bmemory.*leak\b", r"\bpath.*traversal\b",
]

def extract_text_features(description: str) -> dict:
    desc_lower = description.lower()
    words = desc_lower.split()

    return {
        "desc_len_chars": len(description),
        "desc_len_words": len(words),
        "has_rce_keywords":     int(any(re.search(p, desc_lower) for p in EXPLOIT_KEYWORDS)),
        "has_dos_keywords":     int(any(re.search(p, desc_lower) for p in DOS_KEYWORDS)),
        "has_auth_bypass":      int(any(re.search(p, desc_lower) for p in AUTH_BYPASS_KEYWORDS)),
        "has_info_disclosure":  int(any(re.search(p, desc_lower) for p in INFO_DISCLOSURE_KEYWORDS)),
        # Version mentions are correlated with scope clarity (better CVE records)
        "version_mention_count": len(re.findall(r"\d+\.\d+(?:\.\d+)*", description)),
        # CVE cross-references in description
        "cross_cve_refs": len(re.findall(r"CVE-\d{4}-\d+", description)),
        # Technical specificity markers
        "has_heap_mention":   int("heap" in desc_lower),
        "has_stack_mention":  int("stack" in desc_lower),
        "has_buffer_mention": int("buffer" in desc_lower),
        "has_memory_mention": int("memory" in desc_lower),
        "has_sql_mention":    int("sql" in desc_lower),
        "has_xss_mention":    int("xss" in desc_lower or "cross-site scripting" in desc_lower),
    }


# For high-cardinality TF-IDF features (use as complement to above):
def build_tfidf_vectorizer(descriptions: list[str]) -> TfidfVectorizer:
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf.fit(descriptions)
    return tfidf
```

---

### Temporal Features

Publication date and modification patterns are underused but informative:

```python
from datetime import datetime

def extract_temporal_features(published: str, last_modified: str) -> dict:
    pub = datetime.fromisoformat(published.replace("Z", "+00:00"))
    mod = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    now = datetime.now(tz=pub.tzinfo)

    age_days = (now - pub).days
    mod_delta_days = (mod - pub).days

    return {
        "pub_year":          pub.year,
        "pub_month":         pub.month,
        "pub_quarter":       (pub.month - 1) // 3 + 1,
        "pub_day_of_week":   pub.weekday(),     # Monday = 0
        "age_days":          age_days,
        "age_bucket":        min(age_days // 365, 10),  # years old, capped at 10
        # Rapid modification after publication = active exploit research
        "days_to_first_mod": mod_delta_days,
        "was_modified_quickly": int(mod_delta_days <= 7),
        # Years since CVE epoch (1999) — coarse temporal drift capture
        "years_since_1999":  pub.year - 1999,
    }
```

---

### Reference and Metadata Features

References reveal exploit maturity and vendor response:

```python
EXPLOIT_DB_DOMAINS = {"exploit-db.com", "packetstormsecurity.com", "seclists.org"}
VENDOR_ADVISORY_TAGS = {"Vendor Advisory", "Patch", "Mitigation"}
POC_TAGS = {"Exploit", "Proof of Concept"}

def extract_reference_features(references: list[dict]) -> dict:
    tags_flat = [t for ref in references for t in ref.get("tags", [])]
    urls = [ref.get("url", "") for ref in references]

    has_exploitdb = any(any(d in url for d in EXPLOIT_DB_DOMAINS) for url in urls)
    has_github_poc = any("github.com" in url and ("PoC" in url or "poc" in url or "exploit" in url.lower()) for url in urls)

    return {
        "ref_count":             len(references),
        "has_vendor_advisory":   int(any(t in VENDOR_ADVISORY_TAGS for t in tags_flat)),
        "has_patch_reference":   int("Patch" in tags_flat),
        "has_exploit_reference": int(any(t in POC_TAGS for t in tags_flat)),
        "has_exploitdb_ref":     int(has_exploitdb),
        "has_github_poc":        int(has_github_poc),
        "has_nvd_ref":           int(any("nvd.nist.gov" in url for url in urls)),
        "has_github_advisory":   int(any("github.com/advisories" in url for url in urls)),
        # Number of distinct domains referenced (breadth of coverage)
        "distinct_ref_domains":  len({
            url.split("/")[2] for url in urls if "//" in url
        }),
    }
```

---

### Package / CPE Scope Features

The CPE (Common Platform Enumeration) strings in `configurations` define what software and versions are affected. These are critical for the package scanning task:

```python
import re

def extract_cpe_features(configurations: list[dict]) -> dict:
    all_cpe = []
    for config in configurations:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable"):
                    all_cpe.append(match)

    cpe_criteria = [c.get("criteria", "") for c in all_cpe]

    # Determine affected product types from CPE part field:
    # cpe:2.3:<part>:<vendor>:<product>:<version>:...
    # part: a=application, o=operating_system, h=hardware
    parts = []
    for cpe in cpe_criteria:
        segs = cpe.split(":")
        if len(segs) > 2:
            parts.append(segs[2])   # 'a', 'o', or 'h'

    # Version range breadth
    unbounded_count = sum(
        1 for c in all_cpe
        if c.get("versionEndIncluding") is None and c.get("versionEndExcluding") is None
    )

    return {
        "cpe_entry_count":       len(all_cpe),
        "affects_applications":  int("a" in parts),
        "affects_os":            int("o" in parts),
        "affects_hardware":      int("h" in parts),
        "cpe_vendor_count":      len({c.split(":")[3] for c in cpe_criteria if len(c.split(":")) > 3}),
        "cpe_product_count":     len({c.split(":")[4] for c in cpe_criteria if len(c.split(":")) > 4}),
        "has_unbounded_versions": int(unbounded_count > 0),
        # Wide scope (many products) is associated with higher real-world impact
        "is_wide_scope":         int(len(all_cpe) > 10),
    }
```

---

### Full Feature Pipeline

Combining all extractors into a single `pandas` DataFrame row:

```python
import pandas as pd

def build_feature_row(cve_record: dict) -> dict:
    cve = cve_record["cve"]

    desc = next(
        (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
        ""
    )

    features = {"cve_id": cve["id"]}
    features.update(extract_cvss_features(cve.get("metrics", {})))
    features.update(extract_cwe_features(cve.get("weaknesses", [])))
    features.update(extract_text_features(desc))
    features.update(extract_temporal_features(cve["published"], cve["lastModified"]))
    features.update(extract_reference_features(cve.get("references", [])))
    features.update(extract_cpe_features(cve.get("configurations", [])))
    features["description"] = desc   # keep raw text for NLP models
    features["vuln_status"] = cve.get("vulnStatus", "Unknown")

    return features


def build_feature_dataframe(raw_cve_records: list[dict]) -> pd.DataFrame:
    rows = [build_feature_row(r) for r in raw_cve_records]
    df = pd.DataFrame(rows)

    # Type coercions
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
    df["exploitability_score"] = pd.to_numeric(df.get("exploitability_score"), errors="coerce")
    df["impact_score"] = pd.to_numeric(df.get("impact_score"), errors="coerce")
    df["pub_year"] = df["pub_year"].astype("Int64")

    # Severity label (5-class target for classification)
    def score_to_severity(s):
        if pd.isna(s):   return "Unknown"
        if s == 0.0:     return "None"
        if s < 4.0:      return "Low"
        if s < 7.0:      return "Medium"
        if s < 9.0:      return "High"
        return "Critical"

    df["severity_label"] = df["base_score"].apply(score_to_severity)

    return df
```

---

## Exploratory Data Analysis

### Target Variable: Severity Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_severity_distribution(df: pd.DataFrame):
    order = ["None", "Low", "Medium", "High", "Critical", "Unknown"]
    counts = df["severity_label"].value_counts().reindex(order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette="RdYlGn_r")
    axes[0].set_title("CVE Count by Severity (absolute)")
    axes[0].set_ylabel("Count")

    # Proportion over publication year — detect distribution shift
    yearly = df.groupby(["pub_year", "severity_label"]).size().unstack(fill_value=0)
    yearly_pct = yearly.div(yearly.sum(axis=1), axis=0)
    yearly_pct[["Critical", "High", "Medium", "Low"]].plot.area(
        ax=axes[1], colormap="RdYlGn_r", alpha=0.8
    )
    axes[1].set_title("Severity Distribution Over Time (stacked %)")

    plt.tight_layout()
    plt.savefig("eda_severity_distribution.png", dpi=150)
```

**What to look for:**
- Class imbalance: Medium and High dominate; None and Critical are minority classes. Plan for `class_weight='balanced'` or SMOTE.
- Temporal drift: the proportion of Critical CVEs has grown year-over-year since 2016, partly due to CVSS scoring inflation. Models trained on 2015 data degrade on current data.

---

### CVSS Metric Distributions

```python
def plot_cvss_metric_distributions(df: pd.DataFrame):
    cvss_cat_cols = [
        "v3_AV", "v3_AC", "v3_PR", "v3_UI", "v3_S",
        "v3_C",  "v3_I",  "v3_A"
    ]
    cvss_cat_cols = [c for c in cvss_cat_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(cvss_cat_cols):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col.replace("v3_", ""))

    plt.suptitle("CVSS v3.1 Metric Distributions")
    plt.tight_layout()
    plt.savefig("eda_cvss_metrics.png", dpi=150)
```

**Key findings from real NVD data:**
- `AV`: ~70% of CVEs have `N` (Network) — network-accessible vulnerabilities dominate.
- `PR`: ~55% require no privileges (`N`) — most CVEs are unauthenticated.
- `C/I/A`: Impact distributions are strongly bimodal (either None or High).
- `AC`: ~85% are `L` (Low complexity) — attackers rarely need special conditions.

These distributions reveal **heavy skew toward the dangerous end** of each metric. A model that predicts "Network, Low, None, None" for all metrics would be directionally correct most of the time — highlighting the importance of using recall and F1 rather than accuracy.

---

### CWE Category Analysis

```python
def plot_top_cwes(df: pd.DataFrame, top_n: int = 25):
    cwe_counts = df["cwe_id"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(y=cwe_counts.index, x=cwe_counts.values, ax=ax)
    ax.set_title(f"Top {top_n} CWE IDs in NVD dataset")
    ax.set_xlabel("Count")

    plt.tight_layout()
    plt.savefig("eda_top_cwes.png", dpi=150)

def analyze_cwe_severity_profile(df: pd.DataFrame) -> pd.DataFrame:
    """For each CWE, compute mean CVSS score and exploit rate."""
    return (
        df.groupby("cwe_id")
        .agg(
            count=("cve_id", "count"),
            mean_score=("base_score", "mean"),
            mean_exploitability=("exploitability_score", "mean"),
            critical_rate=("severity_label", lambda x: (x == "Critical").mean()),
            high_or_critical_rate=("severity_label", lambda x: x.isin(["High", "Critical"]).mean()),
        )
        .query("count >= 50")
        .sort_values("critical_rate", ascending=False)
        .reset_index()
    )
```

**Notable CWE severity profiles:**
- `CWE-787` (Out-of-bounds Write) — median score ~8.8, ~45% Critical
- `CWE-78` (OS Command Injection) — median score ~9.0, ~55% Critical
- `CWE-79` (XSS) — median score ~6.1, mostly Medium — high volume but lower severity
- `CWE-416` (Use After Free) — median score ~8.5, ~40% Critical, high exploit rate
- `CWE-22` (Path Traversal) — median score ~7.5, moderate exploit rate
- `CWE-noinfo` / `CWE-UNKNOWN` — must be handled as a categorical NA; impute or use embedding

---

### Score Distributions Over Time

```python
def plot_score_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rolling median CVSS score per year
    yearly_score = df.groupby("pub_year")["base_score"].median().reset_index()
    axes[0].plot(yearly_score["pub_year"], yearly_score["base_score"], marker="o")
    axes[0].set_title("Median CVSS Base Score by Year")
    axes[0].set_ylim(0, 10)

    # Score distribution as violin
    recent = df[df["pub_year"] >= 2018]
    sns.violinplot(
        data=recent, x="pub_year", y="base_score",
        ax=axes[1], cut=0, inner="quartile"
    )
    axes[1].set_title("CVSS Base Score Distribution (2018–2026)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("eda_score_trends.png", dpi=150)
```

**Why this matters:** CVSS v3 was introduced in 2016, v3.1 in 2019, v4.0 in 2023. Score distributions shift at these boundaries — not because vulnerabilities changed, but because the scoring rubric changed. A single model trained across all years without a `cvss_version` feature will learn spurious temporal correlations.

---

### Correlation Heatmap

```python
import numpy as np

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "base_score", "exploitability_score", "impact_score",
        "av_severity", "ac_ordinal", "pr_ordinal", "ui_ordinal",
        "c_impact", "i_impact", "a_impact", "max_impact",
        "cwe_numeric", "is_high_severity_cwe", "is_memory_safety_cwe",
        "ref_count", "has_exploit_reference", "has_patch_reference",
        "age_days", "desc_len_words", "has_rce_keywords",
        "rce_proxy", "epss_score",
    ]
    available = [c for c in numeric_cols if c in df.columns]

    corr = df[available].corr(method="spearman")   # Spearman for ordinal features

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlation — CVE Feature Set")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=150)
```

**Key correlation patterns typically observed:**
- `base_score` ↔ `impact_score`: Pearson ~0.85 — impact dominates the final score
- `base_score` ↔ `exploitability_score`: Pearson ~0.55 — secondary factor
- `av_severity` ↔ `exploitability_score`: ~0.60
- `max_impact` ↔ `base_score`: ~0.80
- `has_rce_keywords` ↔ `base_score`: ~0.45 (text is predictive)
- `has_exploit_reference` ↔ `epss_score`: ~0.50 (references are predictive for EPSS)

---

### Description Length vs Severity

```python
def plot_desc_length_vs_severity(df: pd.DataFrame):
    order = ["Low", "Medium", "High", "Critical"]
    subset = df[df["severity_label"].isin(order)]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=subset, x="severity_label", y="desc_len_words",
        order=order, ax=ax, showfliers=False
    )
    ax.set_title("Description Length (words) by Severity")
    plt.tight_layout()
    plt.savefig("eda_desc_length_severity.png", dpi=150)
```

**Finding:** Critical CVEs tend to have slightly longer descriptions (median ~55 words vs ~45 for Low), but the distributions overlap heavily. Description length alone is a weak feature — semantic content matters far more.

---

### Attack Vector vs Exploit Rate

```python
def plot_av_vs_epss(df: pd.DataFrame):
    if "epss_score" not in df.columns:
        print("EPSS data not joined — skipping")
        return

    av_map = {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"}
    subset = df[df["v3_AV"].notna()].copy()
    subset["AV_label"] = subset["v3_AV"].map(av_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        data=subset, x="AV_label", y="epss_score",
        order=["Network", "Adjacent", "Local", "Physical"],
        ax=axes[0], showfliers=False
    )
    axes[0].set_title("EPSS Score by Attack Vector")

    sns.boxplot(
        data=subset, x="AV_label", y="base_score",
        order=["Network", "Adjacent", "Local", "Physical"],
        ax=axes[1], showfliers=False
    )
    axes[1].set_title("CVSS Base Score by Attack Vector")

    plt.tight_layout()
    plt.savefig("eda_av_vs_epss.png", dpi=150)
```

**Finding:** Network-accessible (`AV:N`) CVEs have both higher CVSS scores and dramatically higher EPSS scores than Local or Physical CVEs. This single feature alone is one of the strongest separators between the "patch now" and "patch later" populations.

---

## Feature Importance Analysis

### Tree-Based Importance (XGBoost / RandomForest)

```python
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "av_severity", "ac_ordinal", "pr_ordinal", "ui_ordinal", "scope_changed",
    "c_impact", "i_impact", "a_impact", "max_impact", "rce_proxy",
    "is_high_severity_cwe", "is_high_exploit_cwe", "is_memory_safety_cwe",
    "ref_count", "has_exploit_reference", "has_patch_reference", "has_github_poc",
    "age_days", "pub_year", "was_modified_quickly",
    "desc_len_words", "has_rce_keywords", "has_dos_keywords", "has_auth_bypass",
    "version_mention_count",
    "cpe_entry_count", "affects_os", "is_wide_scope",
]

def train_importance_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(-1)
    y = df["severity_label"].map(
        {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    ).fillna(2)

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    clf.fit(X, y)
    return clf, available

def plot_feature_importance(clf, feature_names: list[str]):
    importance = pd.Series(clf.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(10, 10))
    importance.plot.barh(ax=ax)
    ax.set_title("XGBoost Feature Importance (gain) — CVE Severity")
    plt.tight_layout()
    plt.savefig("feature_importance_xgb.png", dpi=150)
```

---

### SHAP Values

SHAP (SHapley Additive exPlanations) provides per-instance explanations and reliable global importance rankings even for correlated features:

```python
import shap

def compute_shap_analysis(clf, X: pd.DataFrame):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)  # shape: [n_classes, n_samples, n_features]

    # Summary plot — global importance across all classes
    shap.summary_plot(
        shap_values,
        X,
        class_names=["None", "Low", "Medium", "High", "Critical"],
        max_display=20,
        show=False,
    )
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")

    # For binary exploit prediction task (class=Critical):
    shap.summary_plot(
        shap_values[4],   # Critical class SHAP values
        X,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.savefig("shap_critical_class.png", dpi=150, bbox_inches="tight")

    return shap_values
```

---

### Key Findings: Most Predictive Features

Based on SHAP analysis and peer literature, the features rank roughly as follows for **severity classification** and **exploit prediction**:

**Tier 1 — Highest signal (always include):**

| Feature | Why It Matters |
|---|---|
| `c_impact` / `i_impact` / `a_impact` | Impact sub-metrics are the direct inputs to CVSS base score |
| `max_impact` | Composite proxy for worst-case impact |
| `av_severity` | Network-accessible CVEs score ~2.5 pts higher on average |
| `pr_ordinal` | No-privilege-required CVEs have 3× higher EPSS than High-privilege |
| `exploitability_score` | NVD-computed sub-score; top-3 predictor for EPSS |
| `impact_score` | NVD-computed sub-score; top-3 predictor for severity |
| `rce_proxy` | Boolean engineered from AV:N + PR:N + C:H + I:H — directly identifies critical RCE class |

**Tier 2 — High signal (include in non-trivial models):**

| Feature | Why It Matters |
|---|---|
| `is_memory_safety_cwe` | Memory safety bugs (buffer overflow, UAF) are over-represented among exploited CVEs |
| `has_exploit_reference` | Presence of PoC reference is one of the strongest predictors of active exploitation |
| `has_github_poc` | GitHub PoC presence often precedes EPSS spikes by 2–7 days |
| `scope_changed` (v3.1) | `S:C` indicates blast radius extends beyond vulnerable component |
| `ac_ordinal` | Low complexity → 2× more likely to be exploited |
| `has_rce_keywords` | Text feature independently predictive even controlling for CVSS metrics |
| `was_modified_quickly` | CVEs updated within 7 days of publication often have active researcher interest |

**Tier 3 — Moderate signal (include in full-featured models):**

| Feature | Why It Matters |
|---|---|
| `pub_year` | Score inflation, changing CWE distributions, CVSS version shifts |
| `cpe_entry_count` | More affected configurations = broader real-world exposure |
| `affects_os` | OS-level vulnerabilities carry higher exploit value |
| `desc_len_words` | Longer, more specific descriptions correlate with more complete assessments |
| `cross_cve_refs` | Related CVEs often indicate a vulnerability family with active research |
| `ui_ordinal` | User interaction required reduces exploitability substantially |
| `age_days` | Old unpatched CVEs become more dangerous over time |
| `cwe_id` (label-encoded) | Per-CWE severity and exploit rate profiles are informative |

**Tier 4 — Low / conditional signal:**

| Feature | Notes |
|---|---|
| `ref_count` | Weakly correlated; dominated by exploit_reference signal |
| `has_patch_reference` | Negatively correlated with exploit probability (patched = lower risk) |
| `desc_len_chars` | Redundant with `desc_len_words` |
| `pub_day_of_week` | Weak: vendors coordinate disclosures on Tuesdays ("Patch Tuesday") |
| TF-IDF features | High cardinality; only add value when CWE/CVSS fields are missing |

---

## ML Tasks and Model Training

### Task 1 — Severity Classification (CVSS Score Bucket)

**Use case:** Auto-triage new CVEs without waiting for NVD's CVSS assignment (NVD can lag 2–4 weeks for new CVEs in `Awaiting Analysis` status). Predict severity from description + partial metadata.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import numpy as np

STRUCTURED_FEATURES = [
    "av_severity", "ac_ordinal", "pr_ordinal", "ui_ordinal",
    "c_impact", "i_impact", "a_impact", "max_impact", "rce_proxy",
    "is_high_severity_cwe", "is_memory_safety_cwe",
    "has_exploit_reference", "has_patch_reference",
    "age_days", "pub_year", "desc_len_words", "has_rce_keywords",
    "version_mention_count", "ref_count",
]

def train_severity_classifier(df: pd.DataFrame):
    available = [c for c in STRUCTURED_FEATURES if c in df.columns]

    # Filter to records with known severity (have been CVSS-scored by NVD)
    labeled = df[df["severity_label"].isin(["Low", "Medium", "High", "Critical"])].copy()
    X = labeled[available].fillna(-1)

    le = LabelEncoder()
    y = le.fit_transform(labeled["severity_label"])

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    print(f"CV F1-weighted: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)
    return clf, le, available
```

**Realistic performance (NVD dataset, ~150k labeled):** F1-weighted ~0.82–0.87. The main confusion is between adjacent severity buckets (Medium/High), not between None/Critical.

---

### Task 2 — Exploit Probability Prediction (EPSS Regression)

**Use case:** Predict the probability that a given CVE will be exploited within 30 days. Training label = current EPSS score. This is a regression task (output 0–1).

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

EPSS_FEATURES = [
    "av_severity", "ac_ordinal", "pr_ordinal", "ui_ordinal",
    "c_impact", "i_impact", "a_impact", "max_impact", "rce_proxy",
    "is_memory_safety_cwe", "is_high_exploit_cwe",
    "has_exploit_reference", "has_github_poc", "has_exploitdb_ref",
    "was_modified_quickly", "age_days",
    "has_rce_keywords", "has_auth_bypass",
    "cpe_entry_count", "affects_os",
    "pub_year", "exploitability_score",
]

def train_epss_regressor(df: pd.DataFrame):
    available = [c for c in EPSS_FEATURES if c in df.columns]
    subset = df[df["epss_score"].notna()].copy()

    X = subset[available].fillna(-1)
    y = subset["epss_score"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        random_state=42,
    )

    reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE: {mae:.4f}")

    # Log-transform EPSS (very right-skewed) often improves MAE:
    # y_log = np.log1p(y); predict log space, expm1 back

    return reg, available
```

> Note: EPSS itself is updated daily by FIRST.org's model. Training a custom regressor on NVD features to approximate EPSS is useful when you need offline scoring or want to score very new CVEs that EPSS hasn't yet processed. For production prioritization, consume EPSS directly from `epss.cyentia.com`.

---

### Task 3 — Compromised Package Detection (Binary Classification)

**Use case:** Given a package name and version in a dependency manifest (requirements.txt, Cargo.toml, package.json), predict whether it is affected by any known CVE. This is a lookup + ML hybrid:

```python
import pandas as pd
from dataclasses import dataclass

@dataclass
class PackageScanResult:
    package: str
    version: str
    ecosystem: str
    is_vulnerable: bool
    matching_cves: list[str]
    max_cvss_score: float
    has_exploit: bool
    recommended_fix: str | None


def scan_package(
    package: str,
    version: str,
    ecosystem: str,
    osv_client,           # callable: (eco, pkg, ver) → list[osv_advisory]
    epss_lookup: dict,    # cve_id → float
) -> PackageScanResult:
    """
    Phase 1: exact OSV lookup (rule-based, high precision)
    Phase 2: enrich matched advisories with CVSS and EPSS signals
    """
    advisories = osv_client(ecosystem, package, version)

    if not advisories:
        return PackageScanResult(
            package=package, version=version, ecosystem=ecosystem,
            is_vulnerable=False, matching_cves=[], max_cvss_score=0.0,
            has_exploit=False, recommended_fix=None,
        )

    cve_ids = []
    scores = []
    has_exploit = False
    fix_versions = []

    for adv in advisories:
        # Extract CVE aliases
        for alias in adv.get("aliases", []):
            if alias.startswith("CVE-"):
                cve_ids.append(alias)
                epss = epss_lookup.get(alias, 0.0)
                scores.append(epss)

        # EPSS-based exploit signal
        if any(epss_lookup.get(c, 0) > 0.1 for c in cve_ids):
            has_exploit = True

        # Extract fix version from affected ranges
        for affected in adv.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if "fixed" in event:
                        fix_versions.append(event["fixed"])

    return PackageScanResult(
        package=package, version=version, ecosystem=ecosystem,
        is_vulnerable=True,
        matching_cves=list(set(cve_ids)),
        max_cvss_score=max(scores, default=0.0),
        has_exploit=has_exploit,
        recommended_fix=fix_versions[0] if fix_versions else None,
    )
```

**For supply chain / typosquatting detection** (ML layer on top of exact lookup), features from the package registry metadata are the most useful:

```python
# Feature set for typosquatted / backdoored package classifier
PACKAGE_RISK_FEATURES = [
    "days_since_first_publish",   # new packages are higher risk
    "days_since_last_update",     # dormant packages that suddenly update = risk
    "download_count_30d",         # low downloads = less reviewed
    "maintainer_count",           # single-maintainer packages are higher risk
    "has_install_scripts",        # postinstall/preinstall hooks = code exec at install time
    "dependency_count",           # deeply nested deps = larger attack surface
    "is_typosquat_candidate",     # Levenshtein dist < 2 from popular package
    "has_known_vulnerable_dep",   # transitive CVE exposure
    "source_repo_stars",          # low-star repos = less scrutiny
    "version_bump_size",          # major version bumps from unknown authors = risk
]
```

---

### Task 4 — CWE Category Prediction from Description (NLP)

**Use case:** ~15% of NVD records have `CWE-noinfo` or missing CWE. A text classifier can predict the most likely CWE category to enrich these records and improve downstream feature quality.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Top 20 CWE classes cover ~80% of all CVEs — treat as multi-class classification
TOP_CWES = [
    "CWE-787", "CWE-79", "CWE-89", "CWE-20", "CWE-125",
    "CWE-416", "CWE-22", "CWE-78", "CWE-476", "CWE-190",
    "CWE-119", "CWE-798", "CWE-77", "CWE-306", "CWE-862",
    "CWE-287", "CWE-276", "CWE-502", "CWE-269", "CWE-94",
]

def build_cwe_classification_dataset(df: pd.DataFrame) -> Dataset:
    labeled = df[df["cwe_id"].isin(TOP_CWES)].copy()
    label2id = {cwe: i for i, cwe in enumerate(TOP_CWES)}
    labeled["label"] = labeled["cwe_id"].map(label2id)
    return Dataset.from_pandas(labeled[["description", "label"]].dropna())


def finetune_cwe_classifier(dataset: Dataset):
    """
    Fine-tune DistilBERT for CWE prediction.
    Export to ONNX for Rust serving (see usecase_transfer_learning.md).
    """
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["description"],
            truncation=True,
            max_length=256,   # CVE descriptions rarely exceed 200 tokens
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    split = tokenized.train_test_split(test_size=0.15, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TOP_CWES),
        id2label={i: c for i, c in enumerate(TOP_CWES)},
        label2id={c: i for i, c in enumerate(TOP_CWES)},
    )

    args = TrainingArguments(
        output_dir="./cwe_classifier",
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./cwe_classifier_final")
    tokenizer.save_pretrained("./cwe_classifier_final")

    # Export to ONNX:
    # optimum-cli export onnx --model ./cwe_classifier_final \
    #   --task text-classification --opset 17 ./cwe_onnx/
```

---

## ONNX Export and Rust Serving

The export and serving pattern follows the same pipeline as the classical ML and transfer learning guides. A single Axum service can host all four models behind different routes:

```
POST /v1/severity      → XGBoost ONNX (structured features)
POST /v1/exploit-score → XGBoost ONNX regression (structured features)
POST /v1/scan-package  → OSV lookup + EPSS join (rule-based + data enrichment)
POST /v1/predict-cwe   → DistilBERT ONNX (text classification)
```

```python
# Export the XGBoost severity classifier to ONNX (sklearn-onnx path)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# Wrap XGBoost in a sklearn Pipeline to use skl2onnx
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("clf", xgb.XGBClassifier(...)),    # trained model
])
pipeline.fit(X_train, y_train)

initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type, zipmap=False)

with open("severity_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

```rust
// Rust: CVE severity scoring endpoint in Axum
use axum::{extract::State, Json};
use ort::{Session, inputs};
use ndarray::Array2;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct CveSeverityRequest {
    av_severity: f32,
    ac_ordinal: f32,
    pr_ordinal: f32,
    ui_ordinal: f32,
    c_impact: f32,
    i_impact: f32,
    a_impact: f32,
    max_impact: f32,
    rce_proxy: f32,
    // ... remaining STRUCTURED_FEATURES
}

#[derive(Serialize)]
struct CveSeverityResponse {
    predicted_severity: String,
    confidence: f32,
    probabilities: Vec<(String, f32)>,
}

const SEVERITY_LABELS: [&str; 4] = ["Low", "Medium", "High", "Critical"];

async fn predict_severity(
    State(session): State<Arc<Session>>,
    Json(req): Json<CveSeverityRequest>,
) -> Json<CveSeverityResponse> {
    let session = session.clone();

    let result = tokio::task::spawn_blocking(move || {
        // Build feature vector — order must match training feature order
        let features: Vec<f32> = vec![
            req.av_severity, req.ac_ordinal, req.pr_ordinal, req.ui_ordinal,
            req.c_impact, req.i_impact, req.a_impact, req.max_impact, req.rce_proxy,
            // ... all STRUCTURED_FEATURES in training order
        ];

        let n_features = features.len();
        let input = Array2::from_shape_vec((1, n_features), features).unwrap();

        let outputs = session
            .run(inputs!["float_input" => input.view()].unwrap())
            .unwrap();

        // XGBoost with zipmap=False outputs probabilities as float tensor
        let probs = outputs["probabilities"]
            .try_extract_tensor::<f32>()
            .unwrap();
        let probs: Vec<f32> = probs.iter().copied().collect();

        let best_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        CveSeverityResponse {
            predicted_severity: SEVERITY_LABELS[best_idx].to_string(),
            confidence: probs[best_idx],
            probabilities: SEVERITY_LABELS
                .iter()
                .map(|s| s.to_string())
                .zip(probs)
                .collect(),
        }
    })
    .await
    .unwrap();

    Json(result)
}
```

For the CWE text classifier, use the full transformer serving pattern from `usecase_transfer_learning.md` — it is a standard DistilBERT sequence classification ONNX model and requires no special handling beyond what is already covered there.

---

## Decision Matrix

| Scenario | Data input | Model | Serving |
|---|---|---|---|
| **Auto-triage new CVEs (no NVD CVSS yet)** | Description text + partial metadata | DistilBERT fine-tuned on CWE → XGBoost severity | `ort` (two-stage: NLP → tabular) |
| **Severity classification (CVSS available)** | Parsed CVSS + CWE + reference features | XGBoost / GBM → ONNX via skl2onnx | `ort` in Axum, structured features |
| **Exploit probability (EPSS approximation)** | CVSS + reference + text keywords | XGBoost regression → ONNX | `ort` in Axum, regression output |
| **Package vulnerability scan (exact)** | Package name + version + ecosystem | OSV API lookup (rule-based, no ML) | Direct HTTP client in Axum |
| **Package vulnerability scan (enriched)** | OSV result + EPSS join | EPSS lookup table (CSV in memory) | `HashMap<String, f32>` in Axum State |
| **Supply chain / typosquatting detection** | Registry metadata features | Random Forest binary classifier → ONNX | `ort` in Axum |
| **CWE prediction from description** | Free text description | DistilBERT fine-tuned on NVD records | `ort` + `tokenizers` in Axum |
| **CVE search / semantic similarity** | CVE description embeddings | all-MiniLM-L6-v2 → ONNX or fastembed-rs | `fastembed-rs` + vector DB (Qdrant) |
