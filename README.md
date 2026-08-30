# imblearn-resc

**Re-SC (Resampling based on Sample Concatenation)** algorithms for imbalanced learning. 

This package integrates Re-SC resampling with the `scikit-learn` and `imbalanced-learn` ecosystems. It maps strictly imbalanced binary data into a higher-dimensional ($2d$) concatenated feature space. `ReSC` uses density-weighted random sampling; `KMeansReSC` filters majority inputs and summarizes the retained input collection with KMeans centroids.

## 📦 Installation

You can install `imblearn-resc` directly from PyPI using pip:

```bash
pip install imblearn-resc
```

*Requires Python >=3.11, scikit-learn >=1.4.0, and imbalanced-learn >=0.12.0*

---

## 🚀 Quick Start & Usage

Because Re-SC algorithms map your original features ($d$) into a concatenated feature space ($2d$), **you must always pair the Sampler with the `ReSCTransformer` inside an `imblearn` Pipeline.**

* **The Sampler** (`ReSC` or `KMeansReSC`) transforms the training data during `.fit_resample()`.
* **The Transformer** (`ReSCTransformer`) bypasses the training data, but safely duplicates the test data features ($x \rightarrow [x, x]$) during `.predict()` so your classifier receives the correct dimensions.

### Example: Complete Pipeline

Here is a full, runnable example of how to use `ReSC` and `KMeansReSC` with a standard machine learning classifier.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Import the pipeline from imbalanced-learn (NOT standard sklearn!)
from imblearn.pipeline import Pipeline

# 2. Import the Re-SC Samplers and Transformer
from imblearn_resc.oversampling import ReSC, KMeansReSC
from imblearn_resc.preprocessing import ReSCTransformer

# Generate a highly imbalanced dummy dataset (10% minority, 90% majority)
X, y = make_classification(
    n_classes=2, class_sep=2, weights=[0.1, 0.9], 
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=5, n_clusters_per_class=1, 
    n_samples=1000, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Option A: Standard ReSC Pipeline
# ==========================================
pipeline_resc = Pipeline([
    ('sampler', ReSC(M=1.5, k=5, random_state=42)),
    ('transformer', ReSCTransformer()),             # <--- Mandatory!
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and Predict
pipeline_resc.fit(X_train, y_train)
y_pred_resc = pipeline_resc.predict(X_test)

print("ReSC Classification Report:")
print(classification_report(y_test, y_pred_resc))


# ==========================================
# Option B: KMeansReSC Pipeline
# ==========================================
pipeline_kmeans = Pipeline([
    ('sampler', KMeansReSC(
        M=1.5,
        n_neighbors=5,
        safe_threshold=0.9,
        random_state=42,
        max_k_candidates=None,
        kmeans_params={'n_init': 10},
    )),
    ('transformer', ReSCTransformer()),             # <--- Mandatory!
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and Predict
pipeline_kmeans.fit(X_train, y_train)
y_pred_kmeans = pipeline_kmeans.predict(X_test)

print("KMeansReSC Classification Report:")
print(classification_report(y_test, y_pred_kmeans))
```

## 🧠 Key Parameters

### `ReSC`
* **`M`** *(float, default=1.5)*: The maximum acceptable imbalance ratio threshold for the resulting dataset.
* **`k`** *(int, default=5)*: Number of nearest neighbors used to calculate majority sample weights.
* **`alpha`** *(float, default=0.05)*: Significance level for the Z-test used to compute the required statistical sample size.

### `KMeansReSC`

* **`M`** *(float, default=1.5)*: Multiplier used to define the upper cluster-count candidate bound. It is not an unconditional balance guarantee.
* **`n_neighbors`** *(int, default=5)*: Neighborhood size for the majority-input safety score. The queried training observation is included.
* **`safe_threshold`** *(float, default=0.9)*: Inclusive uniform-vote threshold used to retain majority inputs.
* **`random_state`** *(default=None)*: Random state used to derive one integer seed shared by candidate and final KMeans fits.
* **`max_k_candidates`** *(int or None, default=None)*: Maximum size of a deterministic, evenly spaced grid over the Silhouette-feasible part of the theoretical interval. `None` evaluates every feasible integer.
* **`knn_params`** *(dict or None)*: Nearest-neighbor search options. `n_neighbors`, `metric`, and `weights` are owned by KMeans-ReSC and rejected here.
* **`kmeans_params`** *(dict or None)*: Additional KMeans options, including `n_init`. `n_clusters` and `random_state` are owned by KMeans-ReSC and rejected here.

KMeans-ReSC defines $n_1=\lfloor|\mathcal{P}|^2/|\mathcal{N}|\rfloor$, $K_{\min}=\max(1,n_1)$, and $K_{\max}=\max(1,\lfloor M n_1\rfloor)$. By default it evaluates every Silhouette-feasible integer in $[K_{\min},K_{\max}]$. Setting `max_k_candidates` retains at most that many deterministically spaced feasible values, always including the feasible endpoints. The highest observed score is selected and ties use the smallest cluster count. If no value satisfies $2\leq K<|\mathcal{N}_{\mathrm{safe}}|$, the final fit uses $\min(K_{\min},|\mathcal{N}_{\mathrm{safe}}|)$ centroids without Silhouette selection.

The safety filter applies only to original majority input samples. Generated centroids are retained without a second safety filter and are not guaranteed to satisfy the input-safety threshold. With one-hot encoded inputs, centroid coordinates may be fractional numerical prototypes rather than decodable categorical records.

After fitting, `fallback_used_` reports whether an empty filtered collection forced restoration of the complete majority input collection. `selection_fallback_used_` reports whether no candidate produced a Silhouette score. Selection diagnostics are available as `n1_`, `k_min_`, `k_max_`, `safe_majority_count_`, `candidate_ks_`, and `selected_k_`. `fit_resample` always keeps the standard two-value return contract.

### Container behavior

For `KMeansReSC`, NumPy input produces NumPy output. A pandas `DataFrame` input produces a `DataFrame` with generated doubled feature names such as `age_1` and `age_2`; a pandas target `Series` remains a `Series` aligned to the new output index. Original row indices and source dtypes are not preserved because output rows include generated centroids and concatenated pairs.

### Repeatability

Repeatability requires identical ordered input values, preprocessing, package versions, platform, integer `random_state`, and integer `kmeans_params['n_init']`. Other accepted configurations, such as `random_state=None` or `n_init='auto'`, remain outside this guarantee. Centroid order and cross-platform bitwise identity are not guaranteed.

### Windows and MKL troubleshooting

On affected Windows environments using Microsoft OpenMP and Intel MKL, scikit-learn may warn about a KMeans memory leak. Configure the process before Python starts:

```powershell
$env:OMP_NUM_THREADS = "1"
python your_script.py
```

KMeans-ReSC never changes `OMP_NUM_THREADS` and never suppresses the upstream warning. The value above is an operational workaround, not an estimator parameter or universal package default.
