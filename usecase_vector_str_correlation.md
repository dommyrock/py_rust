# Vulnerability Correlation Across Affected Products

A research guide for discovering, quantifying, and modelling correlations between products, vendors, and software components sharing common vulnerabilities. Covers bipartite graph construction from CPE/NVD data, statistical co-occurrence analysis, graph centrality and community detection, dependency propagation, node embeddings, knowledge graph approaches, and serving correlation queries via a Rust/Axum backend using `petgraph`.

> **Last updated:** February 17, 2026

---

## Table of Contents

- [Problem Framing: What Correlations Matter?](#problem-framing-what-correlations-matter)
- [Data Preparation: Building the Joined Entity Table](#data-preparation-building-the-joined-entity-table)
  - [Parsing CPE strings into structured entities](#parsing-cpe-strings-into-structured-entities)
  - [Building the CVE → Product edge list](#building-the-cve--product-edge-list)
  - [Joining EPSS and CWE signals](#joining-epss-and-cwe-signals)
- [Co-occurrence Matrix Analysis](#co-occurrence-matrix-analysis)
  - [Vendor-vendor co-occurrence](#vendor-vendor-co-occurrence)
  - [Product-product co-occurrence](#product-product-co-occurrence)
  - [Similarity metrics: Jaccard, PMI, Lift](#similarity-metrics-jaccard-pmi-lift)
  - [Statistical significance: chi-squared and Fisher's exact test](#statistical-significance-chi-squared-and-fishers-exact-test)
- [EDA: Key Correlation Questions and Visualisations](#eda-key-correlation-questions-and-visualisations)
  - [Which vendors share the most CVEs?](#which-vendors-share-the-most-cves)
  - [Which products cluster by CWE profile?](#which-products-cluster-by-cwe-profile)
  - [How correlated are vendor disclosure timelines?](#how-correlated-are-vendor-disclosure-timelines)
  - [Which products are highest-centrality nodes?](#which-products-are-highest-centrality-nodes)
- [Graph Construction and Analysis with NetworkX](#graph-construction-and-analysis-with-networkx)
  - [Bipartite graph: CVE ↔ Product](#bipartite-graph-cve--product)
  - [Projected product-product graph](#projected-product-product-graph)
  - [Centrality measures](#centrality-measures)
  - [Community detection (Louvain)](#community-detection-louvain)
- [Dependency Propagation Analysis](#dependency-propagation-analysis)
  - [SBOM parsing: CycloneDX and SPDX](#sbom-parsing-cyclonedx-and-spdx)
  - [Transitive CVE exposure scoring](#transitive-cve-exposure-scoring)
  - [Cargo ecosystem (Rust): rustsec + cargo-audit](#cargo-ecosystem-rust-rustsec--cargo-audit)
- [Temporal Correlation Analysis](#temporal-correlation-analysis)
  - [Cross-correlation of vendor disclosure timelines](#cross-correlation-of-vendor-disclosure-timelines)
  - [Coordinated disclosure detection](#coordinated-disclosure-detection)
- [Node Embeddings for Product Similarity](#node-embeddings-for-product-similarity)
  - [Node2Vec on the product-CVE graph](#node2vec-on-the-product-cve-graph)
  - [Product nearest-neighbour search](#product-nearest-neighbour-search)
  - [Link prediction: will product X be affected next?](#link-prediction-will-product-x-be-affected-next)
- [Knowledge Graph Approaches](#knowledge-graph-approaches)
  - [CWE–CVE–CPE unified graph](#cwecvecpe-unified-graph)
  - [Neo4j property graph and Cypher queries](#neo4j-property-graph-and-cypher-queries)
- [Rust Backend: petgraph + Axum](#rust-backend-petgraph--axum)
  - [Building and querying the product graph in Rust](#building-and-querying-the-product-graph-in-rust)
  - [cargo-audit integration for live scanning](#cargo-audit-integration-for-live-scanning)
- [Decision Matrix](#decision-matrix)

---

## Problem Framing: What Correlations Matter?

"Correlation across products affected by vulnerabilities" encompasses several distinct analytical questions. Each drives a different technique:

| Question | Correlation Type | Primary Signal | Technique |
|---|---|---|---|
| Which vendors are hit by the same CVEs? | Vendor–vendor co-occurrence | Shared CVE IDs in CPE | Co-occurrence matrix, Jaccard |
| Which products share a vulnerability profile? | Product–product similarity | CWE-ID + CVSS metrics | Cosine similarity, clustering |
| If library A is vulnerable, is library B also? | Product pair association | CVE co-occurrence | PMI, lift, chi-squared |
| What is the blast radius of a single CVE? | Direct + transitive exposure | Dependency graph | Graph BFS/DFS, propagation score |
| Do vendors coordinate or disclose independently? | Temporal cross-correlation | Publication date per vendor | Time series cross-correlation |
| Which products sit at the center of the CVE graph? | Graph centrality | Product degree / betweenness | PageRank, betweenness centrality |
| Which products form clusters of shared risk? | Community structure | Edge weights in product graph | Louvain, Girvan–Newman |
| Given product X is vulnerable, predict product Y? | Link prediction | Node embedding similarity | Node2Vec + nearest neighbor |

The questions compound: a high-centrality product (e.g., OpenSSL) with many co-occurrences that also sits on many transitive dependency paths is a fundamentally different risk category than a peripheral product with isolated CVEs.

---

## Data Preparation: Building the Joined Entity Table

### Parsing CPE Strings into Structured Entities

CPE 2.3 strings encode vendor, product, and version in a fixed schema. They are the connective tissue between CVE records and real-world software:

```
cpe:2.3:<part>:<vendor>:<product>:<version>:<update>:<edition>:<language>:<sw_edition>:<target_sw>:<target_hw>:<other>
```

```python
from dataclasses import dataclass
import re

@dataclass
class CpeEntity:
    part: str          # 'a'=application, 'o'=os, 'h'=hardware
    vendor: str
    product: str
    version: str       # '*' means any version
    version_end_excl: str | None   # from NVD cpeMatch
    version_end_incl: str | None

def parse_cpe(cpe_string: str) -> CpeEntity | None:
    """Parse a CPE 2.3 URI into its components."""
    if not cpe_string.startswith("cpe:2.3:"):
        return None
    parts = cpe_string.split(":")
    if len(parts) < 6:
        return None
    return CpeEntity(
        part=parts[2],
        vendor=parts[3],
        product=parts[4],
        version=parts[5],
        version_end_excl=None,
        version_end_incl=None,
    )

def normalize_vendor(vendor: str) -> str:
    """Canonical form for vendor names (NVD uses underscores and lowercase)."""
    return vendor.lower().replace("_", " ").strip()

def normalize_product(vendor: str, product: str) -> str:
    """Fully-qualified product identifier for dedup across renamed vendors."""
    return f"{normalize_vendor(vendor)}/{product.lower()}"
```

---

### Building the CVE → Product Edge List

The core data structure for all correlation analysis is a sparse edge list: one row per (CVE, product) pair.

```python
import pandas as pd
from itertools import product as itertools_product

def build_cve_product_edges(raw_cve_records: list[dict]) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (cve_id, product_fqn) pair.
    product_fqn = "vendor/product" — version-agnostic for cross-version analysis.
    """
    rows = []

    for rec in raw_cve_records:
        cve = rec["cve"]
        cve_id = cve["id"]
        pub = cve.get("published", "")
        base_score = None

        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics:
                src = next(
                    (m for m in metrics[key] if m.get("type") == "Primary"),
                    metrics[key][0],
                )
                base_score = src["cvssData"].get("baseScore")
                break

        # CWE
        cwes = [
            d["value"]
            for w in cve.get("weaknesses", [])
            if w.get("type") == "Primary"
            for d in w.get("description", [])
            if d["lang"] == "en" and d["value"].startswith("CWE-")
        ]
        primary_cwe = cwes[0] if cwes else "CWE-UNKNOWN"

        # CPE entities
        seen_products = set()
        for config in cve.get("configurations", []):
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    if not match.get("vulnerable", False):
                        continue
                    cpe = parse_cpe(match.get("criteria", ""))
                    if cpe is None:
                        continue
                    fqn = normalize_product(cpe.vendor, cpe.product)
                    if fqn in seen_products:
                        continue
                    seen_products.add(fqn)
                    rows.append({
                        "cve_id": cve_id,
                        "vendor": normalize_vendor(cpe.vendor),
                        "product": cpe.product.lower(),
                        "product_fqn": fqn,
                        "part": cpe.part,
                        "published": pub,
                        "base_score": base_score,
                        "primary_cwe": primary_cwe,
                    })

    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df["pub_year"] = df["published"].dt.year
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
    return df
```

---

### Joining EPSS and CWE Signals

```python
def enrich_edges(
    edges: pd.DataFrame,
    epss_df: pd.DataFrame,           # columns: cve_id, epss_score, epss_percentile
    cwe_meta: dict[str, dict],       # cwe_id → {name, category, is_memory_safety, ...}
) -> pd.DataFrame:
    df = edges.merge(epss_df, on="cve_id", how="left")
    df["epss_score"] = df["epss_score"].fillna(0.0)

    df["cwe_category"] = df["primary_cwe"].map(
        lambda c: cwe_meta.get(c, {}).get("category", "Unknown")
    )
    df["is_memory_safety"] = df["primary_cwe"].map(
        lambda c: cwe_meta.get(c, {}).get("is_memory_safety", False)
    ).astype(int)

    return df
```

At this point you have a flat DataFrame with columns:

```
cve_id | vendor | product | product_fqn | part | published | pub_year |
base_score | primary_cwe | cwe_category | epss_score | is_memory_safety
```

This is the foundation for every analysis that follows.

---

## Co-occurrence Matrix Analysis

### Vendor-Vendor Co-occurrence

Two vendors co-occur on a CVE when both appear in its CPE configuration list. This happens for:
- Shared upstream libraries (both products bundle the same vulnerable component)
- Platform + application pairings (OS vendor + browser vendor for privilege-escalation chains)
- Re-branded products (acquired companies with identical codebases)

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

def build_vendor_cooccurrence(edges: pd.DataFrame, min_cve_count: int = 20) -> pd.DataFrame:
    """
    Build a symmetric vendor × vendor co-occurrence matrix.
    Cell [i, j] = number of CVEs that affect both vendor i and vendor j.
    """
    # Keep only vendors with >= min_cve_count CVEs (noise reduction)
    vendor_counts = edges.groupby("vendor")["cve_id"].nunique()
    active_vendors = vendor_counts[vendor_counts >= min_cve_count].index.tolist()
    filtered = edges[edges["vendor"].isin(active_vendors)]

    # Group each CVE → set of vendors
    cve_vendors = (
        filtered.groupby("cve_id")["vendor"]
        .apply(set)
        .reset_index()
    )
    # Keep only CVEs affecting >= 2 vendors (otherwise no co-occurrence)
    cve_vendors = cve_vendors[cve_vendors["vendor"].apply(len) >= 2]

    # Build binary matrix: CVE × Vendor
    mlb = MultiLabelBinarizer(classes=sorted(active_vendors))
    binary = mlb.fit_transform(cve_vendors["vendor"])   # shape [n_cves, n_vendors]

    # Co-occurrence = Vᵀ · V  (symmetric, diagonal = vendor's own CVE count)
    cooc = binary.T @ binary
    np.fill_diagonal(cooc, 0)

    cooc_df = pd.DataFrame(cooc, index=mlb.classes_, columns=mlb.classes_)
    return cooc_df


def top_vendor_pairs(cooc_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Unstack and return the strongest vendor pairs sorted by co-occurrence count."""
    stacked = (
        cooc_df.where(np.triu(np.ones(cooc_df.shape, dtype=bool), k=1))
        .stack()
        .reset_index()
    )
    stacked.columns = ["vendor_a", "vendor_b", "shared_cves"]
    return stacked.sort_values("shared_cves", ascending=False).head(top_n)
```

---

### Product-Product Co-occurrence

At product granularity the matrix becomes much sparser but more precise:

```python
def build_product_cooccurrence(
    edges: pd.DataFrame,
    min_cve_count: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns: (cooc_df, product_labels)
    cooc_df: symmetric product × product co-occurrence matrix.
    """
    prod_counts = edges.groupby("product_fqn")["cve_id"].nunique()
    active = prod_counts[prod_counts >= min_cve_count].index.tolist()
    filtered = edges[edges["product_fqn"].isin(active)]

    cve_products = (
        filtered.groupby("cve_id")["product_fqn"]
        .apply(set)
        .reset_index()
    )
    cve_products = cve_products[cve_products["product_fqn"].apply(len) >= 2]

    mlb = MultiLabelBinarizer(classes=sorted(active))
    binary = mlb.fit_transform(cve_products["product_fqn"])

    cooc = binary.T @ binary
    np.fill_diagonal(cooc, 0)

    labels = mlb.classes_.tolist()
    cooc_df = pd.DataFrame(cooc, index=labels, columns=labels)
    return cooc_df, labels
```

---

### Similarity Metrics: Jaccard, PMI, Lift

Raw co-occurrence counts favour frequent products regardless of actual association strength. Normalised metrics are essential for fair comparison:

```python
def compute_jaccard(cooc: np.ndarray, marginals: np.ndarray) -> np.ndarray:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|
    marginals[i] = number of CVEs affecting product i (diagonal of raw cooc)
    """
    n = len(marginals)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            union = marginals[i] + marginals[j] - cooc[i, j]
            if union > 0:
                jaccard[i, j] = jaccard[j, i] = cooc[i, j] / union
    return jaccard


def compute_pmi(cooc: np.ndarray, marginals: np.ndarray, total_cves: int) -> np.ndarray:
    """
    Pointwise Mutual Information: log( P(A,B) / (P(A) * P(B)) )
    Positive PMI (PPMI) clips negative values to 0.
    Interpretation: PMI > 0 means the pair co-occurs more than expected by chance.
    """
    n = len(marginals)
    pmi = np.zeros((n, n))
    pa = marginals / total_cves     # P(product_i)
    for i in range(n):
        for j in range(i + 1, n):
            pab = cooc[i, j] / total_cves   # P(A and B)
            if pab > 0 and pa[i] > 0 and pa[j] > 0:
                raw_pmi = np.log2(pab / (pa[i] * pa[j]))
                pmi[i, j] = pmi[j, i] = max(raw_pmi, 0)   # PPMI
    return pmi


def compute_lift(cooc: np.ndarray, marginals: np.ndarray, total_cves: int) -> np.ndarray:
    """
    Lift: P(A,B) / (P(A) * P(B))  — multiplicative version of PMI.
    Lift = 1: independent.  Lift > 1: positively correlated.
    More interpretable than PMI for non-statisticians.
    """
    pa = marginals / total_cves
    n = len(pa)
    lift = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            expected = pa[i] * pa[j] * total_cves
            if expected > 0:
                lift[i, j] = lift[j, i] = cooc[i, j] / expected
    return lift


# Vectorised version (much faster for large matrices):
def compute_lift_vectorised(cooc: np.ndarray, marginals: np.ndarray, total_cves: int) -> np.ndarray:
    pa = marginals / total_cves
    expected = np.outer(pa, pa) * total_cves
    np.fill_diagonal(expected, 1)   # avoid division by zero on diagonal
    lift = cooc / expected
    np.fill_diagonal(lift, 0)
    return lift
```

---

### Statistical Significance: Chi-Squared and Fisher's Exact Test

High lift on rare products can be a fluke. Always test for significance:

```python
from scipy.stats import chi2_contingency, fisher_exact

def significance_test(
    cooc_count: int,
    marginal_a: int,
    marginal_b: int,
    total_cves: int,
    method: str = "chi2",           # 'chi2' for large samples, 'fisher' for small counts
) -> tuple[float, float]:
    """
    Returns (test_statistic, p_value) for the null hypothesis that
    product A and product B co-occur at the rate expected by chance.

    Contingency table:
                   Has B   No B
    Has A     [ a      b  ]
    No A      [ c      d  ]
    """
    a = cooc_count
    b = marginal_a - cooc_count         # has A but not B
    c = marginal_b - cooc_count         # has B but not A
    d = total_cves - marginal_a - marginal_b + cooc_count  # neither

    table = [[a, b], [c, d]]

    if method == "fisher":
        stat, p = fisher_exact(table, alternative="greater")
    else:
        stat, p, _, _ = chi2_contingency(table, correction=False)

    return stat, p


def build_significant_pairs(
    cooc_df: pd.DataFrame,
    min_cooc: int = 5,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """
    Build a filtered edge table of statistically significant product pairs.
    Applies Bonferroni correction for multiple comparisons.
    """
    labels = cooc_df.index.tolist()
    marginals = cooc_df.values.diagonal().copy()   # before fill_diagonal
    # Recompute marginals: marginal[i] = CVEs affecting product i
    marginals = {
        labels[i]: int(cooc_df.iloc[i].sum()) + int(cooc_df.index.get_loc(labels[i]))
        for i in range(len(labels))
    }

    n_pairs = len(labels) * (len(labels) - 1) // 2
    adjusted_alpha = alpha / n_pairs   # Bonferroni

    rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            c = int(cooc_df.loc[a, b])
            if c < min_cooc:
                continue
            total = int(cooc_df.sum().sum()) // 2 + len(labels) * 5  # approximate
            _, p = significance_test(c, marginals[a], marginals[b], total, method="chi2")
            if p < adjusted_alpha:
                rows.append({
                    "product_a": a,
                    "product_b": b,
                    "shared_cves": c,
                    "p_value": p,
                    "significant": True,
                })

    return pd.DataFrame(rows).sort_values("shared_cves", ascending=False)
```

---

## EDA: Key Correlation Questions and Visualisations

### Which Vendors Share the Most CVEs?

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_vendor_cooccurrence_heatmap(cooc_df: pd.DataFrame, top_n: int = 30):
    """Heatmap of the top-N most co-occurring vendors."""
    # Select top-N vendors by total co-occurrence weight
    total_cooc = cooc_df.sum(axis=1)
    top_vendors = total_cooc.nlargest(top_n).index.tolist()
    subset = cooc_df.loc[top_vendors, top_vendors]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        subset,
        annot=False,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
    )
    ax.set_title(f"Vendor Co-occurrence Heatmap (top {top_n} vendors by total shared CVEs)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    plt.savefig("eda_vendor_cooccurrence_heatmap.png", dpi=150)


def plot_top_vendor_pairs_bar(cooc_df: pd.DataFrame, top_n: int = 20):
    pairs = top_vendor_pairs(cooc_df, top_n=top_n)
    pairs["pair"] = pairs["vendor_a"] + " ↔ " + pairs["vendor_b"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(pairs["pair"], pairs["shared_cves"], color="steelblue")
    ax.set_xlabel("Shared CVE count")
    ax.set_title(f"Top {top_n} Vendor Pairs by Shared CVE Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("eda_top_vendor_pairs.png", dpi=150)
```

**Typical findings on NVD data:**
- Microsoft ↔ Adobe pairs are among the largest (shared platform exploitation chains on Windows)
- Oracle ↔ IBM (Java SE and enterprise middleware share many classes of injection vulnerability)
- Linux kernel ↔ Red Hat / Ubuntu / SUSE (same codebase, multiple vendor CPEs per CVE)
- Browser vendors (Google Chrome ↔ Microsoft Edge ↔ Mozilla Firefox) share Chromium-ancestry vulnerabilities

The Linux kernel case illustrates a key data artefact: the same patch advisory generates multiple CPE entries for every Linux distribution. Normalise by deduplicating on `(vendor, product)` before computing co-occurrence to avoid inflated counts from multi-CPE CVEs.

---

### Which Products Cluster by CWE Profile?

Each product can be represented as a vector of CWE frequencies — a CWE histogram. Products with similar weakness profiles likely share code, development practices, or architectural patterns:

```python
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def build_product_cwe_matrix(edges: pd.DataFrame, min_cve_count: int = 15) -> pd.DataFrame:
    """
    Returns a product × CWE frequency matrix.
    Each row is a normalised histogram of CWE types for that product.
    """
    prod_counts = edges.groupby("product_fqn")["cve_id"].nunique()
    active = prod_counts[prod_counts >= min_cve_count].index

    subset = edges[edges["product_fqn"].isin(active)]
    pivot = (
        subset.groupby(["product_fqn", "primary_cwe"])
        .size()
        .unstack(fill_value=0)
    )
    # L1-normalise each row so it becomes a probability distribution
    matrix = pd.DataFrame(
        normalize(pivot.values, norm="l1"),
        index=pivot.index,
        columns=pivot.columns,
    )
    return matrix


def cluster_products_by_cwe(matrix: pd.DataFrame, n_clusters: int = 8):
    """KMeans clustering of products based on CWE profile vectors."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    # Visualise with PCA reduction to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.7, s=40)

    # Annotate a subset of notable products
    products = matrix.index.tolist()
    notable = {p for p in products if any(
        kw in p for kw in ["openssl", "linux_kernel", "firefox", "chrome", "apache", "nginx", "windows"]
    )}
    for i, prod in enumerate(products):
        if prod in notable:
            ax.annotate(prod.split("/")[-1], (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.9)

    ax.set_title("Product Clusters by CWE Profile (PCA 2D projection)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    plt.savefig("eda_product_cwe_clusters.png", dpi=150)

    return pd.Series(labels, index=matrix.index, name="cluster")


def describe_clusters(matrix: pd.DataFrame, cluster_labels: pd.Series) -> pd.DataFrame:
    """For each cluster, show the dominant CWE types."""
    df = matrix.copy()
    df["cluster"] = cluster_labels
    return (
        df.groupby("cluster")
        .mean()
        .T
        .apply(lambda col: col.nlargest(5))
        .T
    )
```

---

### How Correlated Are Vendor Disclosure Timelines?

Vendors sometimes coordinate disclosures (joint patches, embargo periods). This produces temporal clustering in their CVE publication dates:

```python
def build_vendor_monthly_series(edges: pd.DataFrame, top_n_vendors: int = 20) -> pd.DataFrame:
    """
    Returns a pivot table: month × vendor, values = CVE count per month.
    """
    top_vendors = (
        edges.groupby("vendor")["cve_id"].nunique()
        .nlargest(top_n_vendors)
        .index.tolist()
    )
    subset = edges[edges["vendor"].isin(top_vendors)].copy()
    subset["month"] = subset["published"].dt.to_period("M")

    monthly = (
        subset.groupby(["month", "vendor"])["cve_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    monthly.index = monthly.index.to_timestamp()
    return monthly


def compute_vendor_timeline_correlations(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation between vendor monthly CVE publication counts.
    High correlation = vendors disclose at the same time (shared platform, coordinated).
    """
    return monthly.corr(method="pearson")


def plot_vendor_timeline_heatmap(monthly: pd.DataFrame):
    corr = compute_vendor_timeline_correlations(monthly)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Vendor CVE Disclosure Timeline Correlation (Pearson, monthly counts)")
    plt.tight_layout()
    plt.savefig("eda_vendor_timeline_correlation.png", dpi=150)
```

**Interpretation keys:**
- Microsoft ↔ Adobe: historically high correlation (~0.55–0.70) due to coordinated Patch Tuesday releases
- Google Chrome ↔ Mozilla Firefox: moderate correlation (~0.35–0.50) — share some vulnerability classes but disclosure cadences differ
- Linux kernel ↔ any distro: high correlation because distros derive CVEs from the same upstream

---

### Which Products Are Highest-Centrality Nodes?

```python
def compute_product_centrality(edges: pd.DataFrame, min_cve: int = 10) -> pd.DataFrame:
    """
    Approximate degree centrality: number of unique CVEs × number of unique products
    co-occurring on those CVEs. Identifies products that connect many others.
    """
    prod_counts = edges.groupby("product_fqn")["cve_id"].nunique()
    active = prod_counts[prod_counts >= min_cve].index

    subset = edges[edges["product_fqn"].isin(active)]

    # For each product, count how many distinct other products share its CVEs
    cve_products = subset.groupby("cve_id")["product_fqn"].apply(set)

    product_neighbors = {p: set() for p in active}
    for products in cve_products:
        products_list = list(products)
        for i, p in enumerate(products_list):
            if p in product_neighbors:
                product_neighbors[p].update(
                    q for q in products_list if q != p and q in product_neighbors
                )

    centrality = pd.DataFrame({
        "product_fqn": list(product_neighbors.keys()),
        "cve_count": [prod_counts.get(p, 0) for p in product_neighbors],
        "neighbor_count": [len(v) for v in product_neighbors.values()],
    })
    centrality["centrality_score"] = (
        centrality["cve_count"] * centrality["neighbor_count"]
    )
    return centrality.sort_values("centrality_score", ascending=False)
```

---

## Graph Construction and Analysis with NetworkX

### Bipartite Graph: CVE ↔ Product

The natural representation for this domain is a bipartite graph where one node set is CVEs and the other is products, with edges representing "product X is affected by CVE Y":

```python
import networkx as nx
from networkx.algorithms import bipartite

def build_bipartite_graph(edges: pd.DataFrame) -> nx.Graph:
    """
    Nodes: CVE IDs (bipartite=0) and product FQNs (bipartite=1).
    Edges: (cve_id, product_fqn) with weight = base_score (float).
    """
    B = nx.Graph()

    cves = edges["cve_id"].unique()
    products = edges["product_fqn"].unique()

    B.add_nodes_from(cves, bipartite=0, node_type="cve")
    B.add_nodes_from(products, bipartite=1, node_type="product")

    for _, row in edges.iterrows():
        B.add_edge(
            row["cve_id"],
            row["product_fqn"],
            weight=float(row["base_score"]) if pd.notna(row["base_score"]) else 5.0,
            cwe=row.get("primary_cwe", ""),
            epss=float(row.get("epss_score", 0.0)),
        )

    print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"  CVE nodes: {len(cves)}, Product nodes: {len(products)}")
    print(f"  Is bipartite: {bipartite.is_bipartite(B)}")
    return B
```

---

### Projected Product-Product Graph

Project the bipartite graph onto the product layer. The resulting unipartite graph has products as nodes and edges weighted by the number of shared CVEs. This is the core object for centrality and community analysis:

```python
def project_to_products(
    B: nx.Graph,
    min_shared_cves: int = 2,
    weight_by_score: bool = True,
) -> nx.Graph:
    """
    Project the CVE-product bipartite graph onto the product layer.

    Edge weight options:
      - weight_by_score=False: raw count of shared CVEs
      - weight_by_score=True:  sum of max(base_score) for shared CVEs (severity-weighted)
    """
    product_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}
    cve_nodes     = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}

    P = nx.Graph()
    P.add_nodes_from(product_nodes)

    # For each CVE, connect all product pairs that share it
    for cve in cve_nodes:
        neighbors = list(B.neighbors(cve))
        if len(neighbors) < 2:
            continue
        score = B.nodes[cve].get("base_score", 5.0) if not weight_by_score else \
                max(B[cve][p].get("weight", 5.0) for p in neighbors)

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                pa, pb = neighbors[i], neighbors[j]
                if P.has_edge(pa, pb):
                    P[pa][pb]["weight"] += score if weight_by_score else 1
                    P[pa][pb]["shared_cves"] += 1
                else:
                    P.add_edge(pa, pb, weight=score, shared_cves=1)

    # Filter out weak edges
    weak = [(u, v) for u, v, d in P.edges(data=True) if d["shared_cves"] < min_shared_cves]
    P.remove_edges_from(weak)

    # Remove isolated nodes
    P.remove_nodes_from(list(nx.isolates(P)))

    print(f"Product graph: {P.number_of_nodes()} nodes, {P.number_of_edges()} edges")
    return P
```

---

### Centrality Measures

```python
def compute_graph_centrality(P: nx.Graph) -> pd.DataFrame:
    """
    Compute multiple centrality measures for all products.

    Degree centrality:    fraction of products this product shares CVEs with
    Betweenness:          how often this product lies on shortest paths (bridging role)
    PageRank:             iterative importance — products connected to important products score higher
    Eigenvector:          influence — connected to many well-connected products
    """
    print("Computing degree centrality...")
    degree_c = nx.degree_centrality(P)

    print("Computing betweenness centrality (may be slow on large graphs)...")
    # For large graphs use approximate: nx.betweenness_centrality(P, k=200)
    between_c = nx.betweenness_centrality(P, weight="weight", normalized=True)

    print("Computing PageRank...")
    pagerank = nx.pagerank(P, weight="weight", alpha=0.85, max_iter=200)

    print("Computing eigenvector centrality...")
    try:
        eigen_c = nx.eigenvector_centrality(P, weight="weight", max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigen_c = {n: 0.0 for n in P.nodes()}

    df = pd.DataFrame({
        "product_fqn": list(P.nodes()),
        "degree_centrality": [degree_c[n] for n in P.nodes()],
        "betweenness_centrality": [between_c[n] for n in P.nodes()],
        "pagerank": [pagerank[n] for n in P.nodes()],
        "eigenvector_centrality": [eigen_c[n] for n in P.nodes()],
        "degree": [P.degree(n) for n in P.nodes()],
        "weighted_degree": [sum(d["weight"] for _, _, d in P.edges(n, data=True)) for n in P.nodes()],
    })
    return df.sort_values("pagerank", ascending=False)


def plot_centrality_comparison(centrality_df: pd.DataFrame, top_n: int = 20):
    top = centrality_df.head(top_n)
    labels = [p.split("/")[-1] for p in top["product_fqn"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, col, title in zip(axes, ["pagerank", "betweenness_centrality", "degree_centrality"],
                               ["PageRank", "Betweenness Centrality", "Degree Centrality"]):
        ax.barh(labels[::-1], top[col].iloc[::-1])
        ax.set_title(title)
        ax.set_xlabel("Score")

    plt.suptitle(f"Top {top_n} Products by Centrality Measure")
    plt.tight_layout()
    plt.savefig("eda_centrality_measures.png", dpi=150)
```

**What each centrality measure reveals:**
- **Degree centrality** — which products appear alongside the most others (breadth of exposure)
- **Betweenness centrality** — which products are bridges between otherwise-disconnected clusters (architectural linchpins like `libssl`, `libc`)
- **PageRank** — recursive importance: a product is important if it co-occurs with other important products (useful for risk propagation scoring)
- **Eigenvector centrality** — similar to PageRank but without the teleportation term; amplifies cluster structure

---

### Community Detection (Louvain)

Communities in the product graph correspond to software ecosystems — groups of products that consistently share the same CVEs (because they share code, a common platform, or a vendor):

```python
# pip install python-louvain  (community package)
import community as community_louvain
import matplotlib.cm as cm

def detect_communities(P: nx.Graph) -> dict[str, int]:
    """
    Louvain community detection. Returns dict: product_fqn → community_id.
    Louvain maximises modularity — the degree to which edges within communities
    exceed what is expected by chance. O(n log n) for large sparse graphs.
    """
    partition = community_louvain.best_partition(P, weight="weight", random_state=42)
    modularity = community_louvain.modularity(partition, P, weight="weight")
    n_communities = len(set(partition.values()))
    print(f"Louvain: {n_communities} communities, modularity={modularity:.4f}")
    return partition


def plot_community_graph(P: nx.Graph, partition: dict[str, int], max_nodes: int = 300):
    """Spring-layout visualisation coloured by community."""
    # Subsample for readability
    if P.number_of_nodes() > max_nodes:
        top_nodes = sorted(P.nodes(), key=lambda n: P.degree(n), reverse=True)[:max_nodes]
        subgraph = P.subgraph(top_nodes)
    else:
        subgraph = P

    pos = nx.spring_layout(subgraph, weight="weight", k=0.3, seed=42)
    communities = [partition.get(n, 0) for n in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(18, 16))
    nx.draw_networkx_nodes(subgraph, pos, node_color=communities,
                           cmap=cm.tab20, node_size=30, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.1, width=0.5, ax=ax)

    # Label high-degree nodes
    high_degree = {n: n.split("/")[-1]
                   for n in subgraph.nodes() if subgraph.degree(n) >= 10}
    nx.draw_networkx_labels(subgraph, pos, labels=high_degree, font_size=6, ax=ax)

    ax.set_title("Product Vulnerability Graph — Community Structure (Louvain)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("eda_product_communities.png", dpi=150, bbox_inches="tight")


def describe_communities(
    P: nx.Graph,
    partition: dict[str, int],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """For each community, report the dominant vendor and top CWE types."""
    df = pd.DataFrame({
        "product_fqn": list(partition.keys()),
        "community": list(partition.values()),
    })
    df["vendor"] = df["product_fqn"].str.split("/").str[0]

    # Join back to edges for CWE profile per community
    merged = df.merge(
        edges[["product_fqn", "primary_cwe", "base_score"]].drop_duplicates(),
        on="product_fqn", how="left",
    )
    summary = (
        merged.groupby("community")
        .agg(
            product_count=("product_fqn", "nunique"),
            top_vendor=("vendor", lambda x: x.mode().iloc[0] if len(x) else ""),
            mean_score=("base_score", "mean"),
            top_cwe=("primary_cwe", lambda x: x.mode().iloc[0] if len(x) else ""),
        )
        .sort_values("product_count", ascending=False)
    )
    return summary
```

---

## Dependency Propagation Analysis

### SBOM Parsing: CycloneDX and SPDX

A Software Bill of Materials encodes the complete transitive dependency tree. Correlating SBOMs with the CVE product graph reveals real-world exposure paths:

```python
import json
from pathlib import Path

def parse_cyclonedx_sbom(sbom_path: str) -> dict[str, list[str]]:
    """
    Parse a CycloneDX JSON SBOM.
    Returns: dict mapping component purl → list of dependency purls.
    CycloneDX format: https://cyclonedx.org/specification/overview/
    """
    with open(sbom_path) as f:
        sbom = json.load(f)

    # Build purl → component map
    components = {}
    for comp in sbom.get("components", []):
        purl = comp.get("purl", "")
        if purl:
            components[purl] = {
                "name": comp.get("name", ""),
                "version": comp.get("version", ""),
                "type": comp.get("type", "library"),
            }

    # Build dependency adjacency
    dep_graph: dict[str, list[str]] = {purl: [] for purl in components}
    for dep in sbom.get("dependencies", []):
        ref = dep.get("ref", "")
        if ref in dep_graph:
            dep_graph[ref] = dep.get("dependsOn", [])

    return dep_graph, components


def purl_to_osv_query(purl: str) -> dict | None:
    """
    Convert a Package URL (purl) to an OSV API query body.
    purl format: pkg:<type>/<namespace>/<name>@<version>
    """
    # e.g., pkg:pypi/requests@2.28.0  or  pkg:cargo/serde@1.0.193
    import re
    m = re.match(r"pkg:(\w+)/(?:([^/]+)/)?([^@]+)@([^?]+)", purl)
    if not m:
        return None
    ecosystem_map = {
        "pypi": "PyPI", "cargo": "crates.io", "npm": "npm",
        "maven": "Maven", "golang": "Go", "nuget": "NuGet",
        "gem": "RubyGems",
    }
    pkg_type, namespace, name, version = m.groups()
    ecosystem = ecosystem_map.get(pkg_type.lower())
    if not ecosystem:
        return None
    full_name = f"{namespace}/{name}" if namespace else name
    return {
        "version": version,
        "package": {"name": full_name, "ecosystem": ecosystem},
    }
```

---

### Transitive CVE Exposure Scoring

Building on the SBOM dependency graph, compute per-package propagated risk:

```python
import networkx as nx
from collections import defaultdict

def build_dependency_graph(dep_graph: dict[str, list[str]]) -> nx.DiGraph:
    """Build a directed dependency graph from SBOM dep_graph dict."""
    G = nx.DiGraph()
    for src, deps in dep_graph.items():
        G.add_node(src)
        for dep in deps:
            G.add_edge(src, dep)   # src depends on dep
    return G


def compute_transitive_exposure(
    dep_graph_nx: nx.DiGraph,
    direct_cves: dict[str, list[dict]],   # purl → list of CVE dicts from OSV
    epss_lookup: dict[str, float],         # cve_id → epss_score
) -> pd.DataFrame:
    """
    For each component, compute:
      - direct_cve_count: CVEs directly affecting this component
      - transitive_cve_count: CVEs affecting any dependency (direct + indirect)
      - max_direct_score: highest CVSS score among direct CVEs
      - max_transitive_epss: highest EPSS score across all transitive CVEs
      - propagation_depth: max dependency chain depth to a vulnerable component
    """
    rows = []
    for node in dep_graph_nx.nodes():
        # All nodes reachable FROM this node (its dependencies, recursively)
        descendants = nx.descendants(dep_graph_nx, node)
        all_affected = {node} | descendants

        direct_cve_list = direct_cves.get(node, [])
        transitive_cves = []
        for dep in all_affected:
            transitive_cves.extend(direct_cves.get(dep, []))

        direct_scores = [c.get("base_score", 0) for c in direct_cve_list if c.get("base_score")]
        all_epss = [
            epss_lookup.get(c.get("id", ""), 0)
            for c in transitive_cves
        ]

        # Propagation depth: longest path to a vulnerable dependency
        depth = 0
        for dep in descendants:
            if direct_cves.get(dep):
                try:
                    paths = nx.all_shortest_paths(dep_graph_nx, node, dep)
                    d = max(len(list(p)) - 1 for p in paths)
                    depth = max(depth, d)
                except nx.NetworkXNoPath:
                    pass

        rows.append({
            "purl": node,
            "direct_cve_count": len(direct_cve_list),
            "transitive_cve_count": len(transitive_cves),
            "max_direct_score": max(direct_scores, default=0.0),
            "max_transitive_epss": max(all_epss, default=0.0),
            "propagation_depth": depth,
            "dependency_count": len(descendants),
        })

    df = pd.DataFrame(rows)
    # VPSS: Vulnerability Propagation Severity Score (composite)
    df["vpss"] = (
        df["max_direct_score"] * 0.5
        + df["max_transitive_epss"] * 10 * 0.3
        + np.log1p(df["transitive_cve_count"]) * 0.2
    )
    return df.sort_values("vpss", ascending=False)
```

---

### Cargo Ecosystem (Rust): rustsec + cargo-audit

The Rust ecosystem has first-class tooling for vulnerability scanning through the RustSec advisory database:

```bash
# Install cargo-audit
cargo install cargo-audit

# Scan current project's Cargo.lock
cargo audit

# Output as JSON for programmatic processing
cargo audit --json 2>/dev/null | jq '.vulnerabilities.list[]'
```

```python
import subprocess, json

def run_cargo_audit(manifest_dir: str) -> list[dict]:
    """
    Run cargo-audit on a Rust project and return structured vulnerability results.
    Requires: cargo-audit installed in PATH.
    """
    result = subprocess.run(
        ["cargo", "audit", "--json"],
        cwd=manifest_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # cargo audit exits 1 if vulnerabilities found — capture output regardless
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    vulns = report.get("vulnerabilities", {}).get("list", [])
    rows = []
    for v in vulns:
        adv = v.get("advisory", {})
        pkg = v.get("package", {})
        rows.append({
            "rustsec_id": adv.get("id", ""),
            "cve_id": adv.get("aliases", [None])[0],
            "crate": pkg.get("name", ""),
            "installed_version": pkg.get("version", ""),
            "patched_versions": adv.get("patched", []),
            "severity": adv.get("cvss", {}).get("score") if adv.get("cvss") else None,
            "title": adv.get("title", ""),
            "url": adv.get("url", ""),
        })
    return rows
```

```python
def build_cargo_dependency_graph(lock_file_path: str) -> nx.DiGraph:
    """
    Parse Cargo.lock to build a dependency graph of crate versions.
    Each node is 'crate_name@version', edges point from dependent to dependency.
    """
    import tomllib   # Python 3.11+ stdlib; pip install tomli for older versions

    with open(lock_file_path, "rb") as f:
        lock = tomllib.load(f)

    G = nx.DiGraph()
    for pkg in lock.get("package", []):
        node = f"{pkg['name']}@{pkg['version']}"
        G.add_node(node, name=pkg["name"], version=pkg["version"])
        for dep_str in pkg.get("dependencies", []):
            # dep_str format: "name version" or "name version (source)"
            dep_name = dep_str.split()[0]
            dep_ver  = dep_str.split()[1] if len(dep_str.split()) > 1 else "*"
            dep_node = f"{dep_name}@{dep_ver}"
            G.add_edge(node, dep_node)

    return G
```

---

## Temporal Correlation Analysis

### Cross-Correlation of Vendor Disclosure Timelines

```python
from scipy.signal import correlate
import numpy as np

def compute_cross_correlation(
    monthly: pd.DataFrame,
    vendor_a: str,
    vendor_b: str,
    max_lag_months: int = 12,
) -> pd.Series:
    """
    Cross-correlation between two vendor monthly CVE counts.
    Lag > 0: vendor A leads vendor B (B's peak follows A's peak by lag months).
    Lag < 0: vendor B leads vendor A.
    Useful for detecting coordinated disclosures or follow-on vulnerability patterns.
    """
    a = monthly[vendor_a].fillna(0).values.astype(float)
    b = monthly[vendor_b].fillna(0).values.astype(float)

    # Normalise
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    corr = correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(b))

    # Restrict to ±max_lag_months
    centre = len(a) - 1
    lo, hi = centre - max_lag_months, centre + max_lag_months + 1
    return pd.Series(
        corr[lo:hi] / len(a),
        index=lags[lo:hi],
        name=f"{vendor_a} → {vendor_b}",
    )


def find_leading_vendors(monthly: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    For each vendor pair (A, B): compute Pearson correlation between A's count
    at month t and B's count at month t+lag.
    High positive correlation = A leads B by `lag` months.
    """
    vendors = monthly.columns.tolist()
    rows = []
    for i, a in enumerate(vendors):
        for j, b in enumerate(vendors):
            if i == j:
                continue
            a_series = monthly[a].iloc[:-lag].values.astype(float)
            b_lagged  = monthly[b].iloc[lag:].values.astype(float)
            if np.std(a_series) == 0 or np.std(b_lagged) == 0:
                continue
            r = np.corrcoef(a_series, b_lagged)[0, 1]
            if abs(r) > 0.2:
                rows.append({
                    "leading_vendor": a,
                    "following_vendor": b,
                    "lag_months": lag,
                    "correlation": r,
                })

    return pd.DataFrame(rows).sort_values("correlation", ascending=False)
```

---

### Coordinated Disclosure Detection

```python
def detect_coordinated_disclosures(edges: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """
    A CVE is 'coordinated' if multiple vendors publish it within `window_days` of each other.
    Returns CVEs with the number of vendors disclosing within the window.
    """
    cve_dates = (
        edges.groupby(["cve_id", "vendor"])["published"]
        .min()
        .reset_index()
    )

    results = []
    for cve_id, group in cve_dates.groupby("cve_id"):
        if len(group) < 2:
            continue
        dates = group["published"].sort_values()
        span = (dates.max() - dates.min()).days
        results.append({
            "cve_id": cve_id,
            "vendor_count": len(group),
            "disclosure_span_days": span,
            "is_coordinated": int(span <= window_days),
            "vendors": list(group["vendor"]),
        })

    df = pd.DataFrame(results)
    pct_coordinated = df["is_coordinated"].mean() * 100
    print(f"Coordinated disclosures (≤{window_days} days): {pct_coordinated:.1f}% of multi-vendor CVEs")
    return df.sort_values("vendor_count", ascending=False)
```

---

## Node Embeddings for Product Similarity

### Node2Vec on the Product-CVE Graph

Node2Vec performs biased random walks on the graph and trains a Word2Vec skip-gram model on the walk sequences. The resulting embeddings capture both local neighbourhood structure (BFS) and global community structure (DFS) depending on the `p` and `q` parameters:

```python
# pip install node2vec  (wraps gensim Word2Vec)
from node2vec import Node2Vec
import numpy as np

def train_node2vec_embeddings(
    P: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    p: float = 1.0,       # return parameter: low p = BFS-like (local structure)
    q: float = 0.5,       # in-out parameter: low q = DFS-like (global communities)
    workers: int = 4,
) -> dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings on the product-product graph.

    Parameter intuition for vulnerability correlation:
      p=1, q=0.5 → DFS-biased: captures community membership (product ecosystems)
      p=0.5, q=2 → BFS-biased: captures local clique structure (shared-library groups)
    """
    node2vec = Node2Vec(
        P,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        weight_key="weight",
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {
        node: model.wv[node]
        for node in P.nodes()
        if node in model.wv
    }
    model.save("product_node2vec.model")
    return embeddings
```

---

### Product Nearest-Neighbour Search

```python
from sklearn.neighbors import NearestNeighbors

def build_product_similarity_index(
    embeddings: dict[str, np.ndarray],
) -> tuple[NearestNeighbors, list[str]]:
    """
    Build a k-NN index over product embeddings for fast similarity lookup.
    Use cosine distance — embedding norms carry magnitude noise.
    """
    products = list(embeddings.keys())
    matrix = np.stack([embeddings[p] for p in products])

    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(matrix)
    return index, products


def find_similar_products(
    query_product: str,
    index: NearestNeighbors,
    product_list: list[str],
    embeddings: dict[str, np.ndarray],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Return the k most similar products to a query product.
    Similarity is defined by shared position in the CVE co-occurrence graph.
    """
    if query_product not in embeddings:
        raise KeyError(f"Product '{query_product}' not in embedding vocabulary")

    vec = embeddings[query_product].reshape(1, -1)
    distances, indices = index.kneighbors(vec, n_neighbors=top_k + 1)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        neighbor = product_list[idx]
        if neighbor == query_product:
            continue
        results.append({
            "product_fqn": neighbor,
            "cosine_distance": dist,
            "similarity": 1 - dist,
        })

    return pd.DataFrame(results).head(top_k)
```

---

### Link Prediction: Will Product X Be Affected Next?

Given a newly disclosed CVE and its current set of affected products, predict which other products are likely to be affected:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def build_link_prediction_features(
    product_a: str,
    product_b: str,
    P: nx.Graph,
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute a feature vector for a (product_a, product_b) pair for link prediction.
    The target label is 1 if they share >= 1 CVE, 0 otherwise.
    """
    # Graph-structural features (Adamic-Adar, Jaccard, common neighbours)
    aa = next(nx.adamic_adar_index(P, [(product_a, product_b)]), (None, None, 0))[2]
    jc = next(nx.jaccard_coefficient(P, [(product_a, product_b)]), (None, None, 0))[2]
    cn = len(list(nx.common_neighbors(P, product_a, product_b)))

    # Embedding-based features
    if product_a in embeddings and product_b in embeddings:
        va, vb = embeddings[product_a], embeddings[product_b]
        cosine  = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
        hadamard = va * vb
    else:
        cosine = 0.0
        hadamard = np.zeros(64)

    structural = np.array([aa, jc, cn, cosine])
    return np.concatenate([structural, hadamard])


def train_link_predictor(
    P: nx.Graph,
    embeddings: dict[str, np.ndarray],
    neg_ratio: int = 3,
) -> LogisticRegression:
    """
    Train a link prediction model.
    Positive examples: existing edges in P.
    Negative examples: random non-edges (sampled at neg_ratio × positive count).
    """
    import random
    nodes = list(P.nodes())
    positive_edges = list(P.edges())
    all_non_edges = list(nx.non_edges(P))
    negative_edges = random.sample(all_non_edges, min(len(positive_edges) * neg_ratio, len(all_non_edges)))

    X, y = [], []
    for a, b in positive_edges:
        X.append(build_link_prediction_features(a, b, P, embeddings))
        y.append(1)
    for a, b in negative_edges:
        X.append(build_link_prediction_features(a, b, P, embeddings))
        y.append(0)

    X, y = np.array(X), np.array(y)

    clf = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0)
    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"Link prediction ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    clf.fit(X, y)
    return clf


def predict_affected_products(
    known_affected: list[str],       # products already confirmed in a new CVE
    candidate_products: list[str],   # products to score
    P: nx.Graph,
    embeddings: dict[str, np.ndarray],
    clf: LogisticRegression,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Given a new CVE affecting `known_affected`, rank `candidate_products` by
    probability of also being affected.
    """
    rows = []
    for candidate in candidate_products:
        if candidate in known_affected:
            continue
        scores = []
        for known in known_affected:
            feats = build_link_prediction_features(known, candidate, P, embeddings)
            prob = clf.predict_proba([feats])[0][1]
            scores.append(prob)
        # Use max probability across all known-affected products
        rows.append({"product_fqn": candidate, "affected_probability": max(scores, default=0.0)})

    return pd.DataFrame(rows).sort_values("affected_probability", ascending=False).head(top_k)
```

---

## Knowledge Graph Approaches

### CWE–CVE–CPE Unified Graph

A unified knowledge graph treats CVEs, CWEs, products, and vendors as entities connected by typed relationships. This enables complex traversal queries impossible in a flat table:

```
Entity types:
  :CVE   {id, base_score, published, epss_score}
  :CWE   {id, name, category}
  :Product {fqn, name}
  :Vendor  {name}

Relationship types:
  (:CVE)-[:HAS_WEAKNESS]→(:CWE)
  (:CVE)-[:AFFECTS]→(:Product)
  (:Product)-[:MADE_BY]→(:Vendor)
  (:Product)-[:DEPENDS_ON]→(:Product)      # from SBOM
  (:CWE)-[:PARENT_OF]→(:CWE)              # CWE hierarchy
```

```python
# pip install py2neo  (Neo4j Python driver)
from py2neo import Graph, Node, Relationship

def load_knowledge_graph(
    edges: pd.DataFrame,
    neo4j_uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
):
    graph = Graph(neo4j_uri, auth=(user, password))
    graph.run("MATCH (n) DETACH DELETE n")   # clear existing (dev only)

    tx = graph.begin()
    cve_nodes, product_nodes, vendor_nodes, cwe_nodes = {}, {}, {}, {}

    for _, row in edges.iterrows():
        cve_id = row["cve_id"]
        if cve_id not in cve_nodes:
            n = Node("CVE", id=cve_id,
                     base_score=float(row["base_score"]) if pd.notna(row["base_score"]) else None,
                     epss_score=float(row.get("epss_score", 0)),
                     pub_year=int(row.get("pub_year", 0)))
            cve_nodes[cve_id] = n
            tx.create(n)

        vendor = row["vendor"]
        if vendor not in vendor_nodes:
            vn = Node("Vendor", name=vendor)
            vendor_nodes[vendor] = vn
            tx.create(vn)

        prod_fqn = row["product_fqn"]
        if prod_fqn not in product_nodes:
            pn = Node("Product", fqn=prod_fqn, name=row["product"])
            product_nodes[prod_fqn] = pn
            tx.create(pn)
            tx.create(Relationship(pn, "MADE_BY", vendor_nodes[vendor]))

        cwe = row.get("primary_cwe", "CWE-UNKNOWN")
        if cwe not in cwe_nodes:
            cn = Node("CWE", id=cwe)
            cwe_nodes[cwe] = cn
            tx.create(cn)

        tx.create(Relationship(cve_nodes[cve_id], "AFFECTS", product_nodes[prod_fqn]))
        tx.create(Relationship(cve_nodes[cve_id], "HAS_WEAKNESS", cwe_nodes[cwe]))

    graph.commit(tx)
    print(f"Loaded {len(cve_nodes)} CVEs, {len(product_nodes)} products, "
          f"{len(vendor_nodes)} vendors, {len(cwe_nodes)} CWEs")
    return graph
```

---

### Neo4j Property Graph and Cypher Queries

Once the graph is loaded, Cypher enables correlation queries that would be awkward in SQL or pandas:

```cypher
// 1. Which products are most connected (highest CVE degree)?
MATCH (p:Product)<-[:AFFECTS]-(c:CVE)
RETURN p.fqn AS product, count(c) AS cve_count
ORDER BY cve_count DESC
LIMIT 20;

// 2. Which product PAIRS share the most high-severity CVEs?
MATCH (pa:Product)<-[:AFFECTS]-(c:CVE)-[:AFFECTS]->(pb:Product)
WHERE pa.fqn < pb.fqn AND c.base_score >= 9.0
RETURN pa.fqn, pb.fqn, count(c) AS shared_critical_cves
ORDER BY shared_critical_cves DESC
LIMIT 20;

// 3. Community of products sharing a specific CWE type
MATCH (p:Product)<-[:AFFECTS]-(c:CVE)-[:HAS_WEAKNESS]->(w:CWE {id: "CWE-787"})
RETURN p.fqn, count(c) AS oom_write_cves, avg(c.base_score) AS avg_score
ORDER BY oom_write_cves DESC;

// 4. Transitive exposure: which products share vulnerabilities with OpenSSL
//    through shared upstream dependencies?
MATCH path = (openssl:Product {fqn: "openssl/openssl"})
             <-[:AFFECTS]-(c:CVE)-[:AFFECTS]->(other:Product)
WHERE other.fqn <> "openssl/openssl"
RETURN other.fqn, count(c) AS shared_cves, max(c.base_score) AS max_score
ORDER BY shared_cves DESC;

// 5. Vendor co-occurrence weighted by EPSS (exploitation risk)
MATCH (va:Vendor)<-[:MADE_BY]-(pa:Product)<-[:AFFECTS]-(c:CVE)-[:AFFECTS]->(pb:Product)
      -[:MADE_BY]->(vb:Vendor)
WHERE va.name < vb.name AND c.epss_score > 0.1
RETURN va.name, vb.name, count(c) AS exploited_shared_cves, sum(c.epss_score) AS total_epss
ORDER BY total_epss DESC
LIMIT 20;

// 6. Graph Data Science: run Louvain directly in Neo4j
//    (requires Neo4j GDS plugin)
CALL gds.graph.project(
  'productGraph',
  'Product',
  {AFFECTS_SAME: {type: '*', orientation: 'UNDIRECTED'}}
);
CALL gds.louvain.write('productGraph', {writeProperty: 'community'})
YIELD communityCount, modularity;
```

---

## Rust Backend: petgraph + Axum

### Building and Querying the Product Graph in Rust

`petgraph` is the canonical graph library for Rust. It supports directed/undirected graphs with arbitrary node/edge data, and ships Dijkstra, Bellman-Ford, Floyd-Warshall, topological sort, SCC detection (Tarjan/Kosaraju), minimum spanning tree, and isomorphism algorithms.

[![Current Crates.io Version](https://img.shields.io/crates/v/petgraph.svg)](https://crates.io/crates/petgraph)

```toml
[dependencies]
petgraph = { version = "0.7", features = ["serde-1"] }
serde = { version = "1", features = ["derive"] }
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

```rust
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::{dijkstra, page_rank};
use petgraph::Undirected;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Node payload: a product in the vulnerability graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductNode {
    pub fqn: String,          // "vendor/product"
    pub cve_count: u32,
    pub mean_score: f32,
    pub community: Option<u32>,
}

/// Edge payload: the relationship between two products
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedVulnEdge {
    pub shared_cves: u32,
    pub max_score: f32,
    pub weight: f32,          // used by weighted algorithms
}

pub type ProductGraph = Graph<ProductNode, SharedVulnEdge, Undirected>;

/// Build the product-product co-occurrence graph from a flat edge list
pub fn build_product_graph(
    edges: &[(String, String, u32, f32)],   // (product_a, product_b, shared_cves, max_score)
    min_shared: u32,
) -> (ProductGraph, HashMap<String, NodeIndex>) {
    let mut graph = ProductGraph::new_undirected();
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

    // Collect per-product CVE stats
    let mut cve_counts: HashMap<String, u32> = HashMap::new();
    let mut score_sums: HashMap<String, f32> = HashMap::new();

    for (a, b, count, score) in edges {
        *cve_counts.entry(a.clone()).or_insert(0) += count;
        *cve_counts.entry(b.clone()).or_insert(0) += count;
        *score_sums.entry(a.clone()).or_insert(0.0) += score;
        *score_sums.entry(b.clone()).or_insert(0.0) += score;
    }

    // Add nodes
    for (fqn, &cve_count) in &cve_counts {
        let mean_score = score_sums[fqn] / cve_count as f32;
        let idx = graph.add_node(ProductNode {
            fqn: fqn.clone(),
            cve_count,
            mean_score,
            community: None,
        });
        node_map.insert(fqn.clone(), idx);
    }

    // Add edges
    for (a, b, shared_cves, max_score) in edges {
        if *shared_cves < min_shared {
            continue;
        }
        if let (Some(&idx_a), Some(&idx_b)) = (node_map.get(a), node_map.get(b)) {
            graph.add_edge(idx_a, idx_b, SharedVulnEdge {
                shared_cves: *shared_cves,
                max_score: *max_score,
                // Inverse weight for shortest-path algorithms:
                // products with MANY shared CVEs are "closer" in risk space
                weight: 1.0 / (*shared_cves as f32 + 1.0),
            });
        }
    }

    (graph, node_map)
}
```

```rust
use petgraph::algo::page_rank;
use petgraph::visit::EdgeRef;

/// Compute PageRank for all product nodes
pub fn compute_pagerank(graph: &ProductGraph, damping: f32, iterations: u32) -> HashMap<NodeIndex, f32> {
    let n = graph.node_count();
    if n == 0 { return HashMap::new(); }

    let mut ranks: HashMap<NodeIndex, f32> = graph.node_indices()
        .map(|i| (i, 1.0 / n as f32))
        .collect();

    let out_weights: HashMap<NodeIndex, f32> = graph.node_indices()
        .map(|i| {
            let total: f32 = graph.edges(i)
                .map(|e| e.weight().shared_cves as f32)
                .sum();
            (i, if total > 0.0 { total } else { 1.0 })
        })
        .collect();

    for _ in 0..iterations {
        let mut new_ranks: HashMap<NodeIndex, f32> = graph.node_indices()
            .map(|i| (i, (1.0 - damping) / n as f32))
            .collect();

        for node in graph.node_indices() {
            for edge in graph.edges(node) {
                let neighbor = edge.target();
                let edge_weight = edge.weight().shared_cves as f32;
                let contribution = damping * ranks[&node] * (edge_weight / out_weights[&node]);
                *new_ranks.entry(neighbor).or_insert(0.0) += contribution;
            }
        }
        ranks = new_ranks;
    }
    ranks
}

/// Find the k products most similar to a query product (by shortest weighted path)
pub fn k_nearest_products(
    query_fqn: &str,
    graph: &ProductGraph,
    node_map: &HashMap<String, NodeIndex>,
    k: usize,
) -> Vec<(String, f32)> {
    let query_idx = match node_map.get(query_fqn) {
        Some(&idx) => idx,
        None => return vec![],
    };

    // Dijkstra returns shortest path distances from query_idx
    let distances: HashMap<NodeIndex, f32> = dijkstra(
        graph,
        query_idx,
        None,
        |e| e.weight().weight,   // inverse of shared_cves
    );

    let mut results: Vec<(String, f32)> = distances
        .into_iter()
        .filter(|(idx, _)| *idx != query_idx)
        .map(|(idx, dist)| (graph[idx].fqn.clone(), dist))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(k);
    results
}
```

---

### Axum API for Correlation Queries

```rust
use axum::{extract::{State, Path, Query}, Json, Router, routing::get};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct GraphState {
    pub graph: Arc<RwLock<ProductGraph>>,
    pub node_map: Arc<RwLock<HashMap<String, NodeIndex>>>,
    pub pagerank: Arc<RwLock<HashMap<NodeIndex, f32>>>,
}

#[derive(Serialize)]
struct SimilarProductsResponse {
    query: String,
    similar: Vec<SimilarProduct>,
}

#[derive(Serialize)]
struct SimilarProduct {
    product_fqn: String,
    distance: f32,
    shared_cves: u32,
    pagerank_score: f32,
}

#[derive(Deserialize)]
struct SimilarQuery {
    k: Option<usize>,
}

async fn similar_products_handler(
    State(state): State<GraphState>,
    Path(product_fqn): Path<String>,
    Query(params): Query<SimilarQuery>,
) -> Json<SimilarProductsResponse> {
    let k = params.k.unwrap_or(10);
    let graph = state.graph.read().await;
    let node_map = state.node_map.read().await;
    let pagerank = state.pagerank.read().await;

    let neighbors = k_nearest_products(&product_fqn, &graph, &node_map, k);

    let similar: Vec<SimilarProduct> = neighbors
        .into_iter()
        .map(|(fqn, dist)| {
            let idx = node_map.get(&fqn).copied();
            let shared = idx
                .and_then(|i| node_map.get(&product_fqn).copied().map(|j| (i, j)))
                .and_then(|(i, j)| graph.find_edge(i, j))
                .map(|e| graph[e].shared_cves)
                .unwrap_or(0);
            let pr = idx.and_then(|i| pagerank.get(&i)).copied().unwrap_or(0.0);

            SimilarProduct { product_fqn: fqn, distance: dist, shared_cves: shared, pagerank_score: pr }
        })
        .collect();

    Json(SimilarProductsResponse { query: product_fqn, similar })
}

#[derive(Serialize)]
struct TopProductsResponse {
    products: Vec<TopProduct>,
}

#[derive(Serialize)]
struct TopProduct {
    product_fqn: String,
    cve_count: u32,
    pagerank_score: f32,
    degree: usize,
}

async fn top_products_handler(State(state): State<GraphState>) -> Json<TopProductsResponse> {
    let graph = state.graph.read().await;
    let pagerank = state.pagerank.read().await;

    let mut products: Vec<TopProduct> = graph.node_indices()
        .map(|i| TopProduct {
            product_fqn: graph[i].fqn.clone(),
            cve_count: graph[i].cve_count,
            pagerank_score: *pagerank.get(&i).unwrap_or(&0.0),
            degree: graph.neighbors(i).count(),
        })
        .collect();

    products.sort_by(|a, b| b.pagerank_score.partial_cmp(&a.pagerank_score).unwrap());
    products.truncate(50);

    Json(TopProductsResponse { products })
}

pub fn build_router(state: GraphState) -> Router {
    Router::new()
        .route("/products/top", get(top_products_handler))
        .route("/products/:fqn/similar", get(similar_products_handler))
        .with_state(state)
}
```

### cargo-audit Integration for Live Scanning

```rust
use std::process::Command;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct AuditReport {
    vulnerabilities: AuditVulnerabilities,
}

#[derive(Deserialize, Debug)]
struct AuditVulnerabilities {
    list: Vec<AuditVulnerability>,
    count: u32,
}

#[derive(Deserialize, Debug)]
struct AuditVulnerability {
    advisory: Advisory,
    package: PackageInfo,
}

#[derive(Deserialize, Debug)]
struct Advisory {
    id: String,
    title: String,
    url: String,
    #[serde(default)]
    aliases: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct PackageInfo {
    name: String,
    version: String,
}

pub fn run_cargo_audit(manifest_dir: &str) -> Result<Vec<AuditVulnerability>, String> {
    let output = Command::new("cargo")
        .args(["audit", "--json"])
        .current_dir(manifest_dir)
        .output()
        .map_err(|e| format!("Failed to run cargo audit: {e}"))?;

    // cargo audit exits 1 when vulns found — parse stdout regardless
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Ok(vec![]);
    }

    let report: AuditReport = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse cargo audit output: {e}"))?;

    Ok(report.vulnerabilities.list)
}
```

---

## Decision Matrix

| Goal | Primary Technique | Python Libraries | Rust Component |
|---|---|---|---|
| **Which vendors share the most CVEs?** | Co-occurrence matrix + Jaccard | `pandas`, `scipy.sparse`, `sklearn.preprocessing` | Serialised to JSON; served from Axum cache |
| **Statistically significant product pairs** | Chi-squared / Fisher's exact test | `scipy.stats.chi2_contingency`, `fisher_exact` | — (offline Python analysis) |
| **Product clusters by weakness profile** | CWE histogram → KMeans / UMAP | `sklearn.cluster`, `umap-learn`, `seaborn` | — |
| **Highest-centrality products** | PageRank, betweenness in product graph | `networkx` | `petgraph` (PageRank implemented above) |
| **Ecosystem communities** | Louvain modularity maximisation | `python-louvain`, `networkx`, `neo4j-gds` | — (Python offline; results stored in graph nodes) |
| **Product similarity search** | Node2Vec embeddings + k-NN index | `node2vec`, `gensim`, `sklearn.neighbors` | `ort` (serve embeddings as ONNX) + in-memory HashMap |
| **Predict products affected by new CVE** | Link prediction on embedding + structural features | `sklearn.linear_model`, `node2vec` | Export classifier to ONNX; serve via `ort` |
| **Transitive CVE exposure from SBOM** | Dependency graph BFS | `networkx`, `cyclonedx-python`, `requests` (OSV API) | `petgraph` BFS/DFS; `cargo-audit` for Rust projects |
| **Temporal vendor co-disclosure** | Cross-correlation of monthly CVE counts | `scipy.signal.correlate`, `pandas`, `matplotlib` | — (offline analysis) |
| **Knowledge graph traversal** | Property graph + Cypher | `py2neo`, `neo4j` Python driver | Neo4j Bolt driver for Rust (`neo4rs`) |
| **Live Rust dependency scan** | cargo-audit + RustSec advisory DB | — | `cargo-audit` subprocess + JSON parsing in Axum |
| **Full unified knowledge graph** | CWE–CVE–CPE graph in Neo4j | `py2neo`, `networkx` | `neo4rs` for read queries from Axum |
