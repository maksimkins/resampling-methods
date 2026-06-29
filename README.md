# imblearn-resc

**Re-SC (Resampling based on Sample Concatenation)** algorithms for imbalanced learning. 

This package is fully compatible with the `scikit-learn` and `imbalanced-learn` ecosystems. It addresses class imbalance by mapping data into a higher-dimensional (2d) concatenated feature space, utilizing either density-weighted random sampling (`ReSC`) or K-Means clustering (`KMeansReSC`) to safely resample the majority and minority classes.

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
    ('sampler', KMeansReSC(M=1.5, num_candidates_to_test=5, random_state=42)),
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
* **`M`** *(float, default=1.5)*: The maximum acceptable imbalance ratio threshold for the resulting dataset.
* **`num_candidates_to_test`** *(int, default=5)*: How many 'k' values (clusters) to test during geometric tuning using the Silhouette Score.