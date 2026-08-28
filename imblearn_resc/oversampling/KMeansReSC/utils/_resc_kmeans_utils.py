from typing import Tuple, List, Union, Any, Optional

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score


_RESERVED_KNN_PARAMS = frozenset({"metric", "n_neighbors", "weights"})
_RESERVED_KMEANS_PARAMS = frozenset({"n_clusters", "random_state"})


def _validated_parameter_copy(
    params: Optional[dict],
    *,
    reserved: frozenset[str],
    parameter_name: str,
) -> dict:
    """Copy a parameter mapping and reject keys owned by KMeans-ReSC."""
    copied = dict(params) if params is not None else {}
    conflicts = sorted(reserved.intersection(copied))
    if conflicts:
        quoted = ", ".join(repr(key) for key in conflicts)
        raise ValueError(
            f"{parameter_name} contains reserved parameter(s): {quoted}. "
            "Configure these values through KMeansReSC instead."
        )
    return copied


def validate_knn_params(knn_params: Optional[dict]) -> dict:
    """Return a safe copy of user KNN search parameters."""
    return _validated_parameter_copy(
        knn_params,
        reserved=_RESERVED_KNN_PARAMS,
        parameter_name="knn_params",
    )


def validate_kmeans_params(kmeans_params: Optional[dict]) -> dict:
    """Return a safe copy of user KMeans parameters."""
    return _validated_parameter_copy(
        kmeans_params,
        reserved=_RESERVED_KMEANS_PARAMS,
        parameter_name="kmeans_params",
    )


def _split_kmeans_params(kmeans_params: Optional[dict]) -> tuple[dict, Any]:
    """Separate n_init while preserving the caller-owned mapping."""
    kmeans_kwargs = validate_kmeans_params(kmeans_params)
    n_init = kmeans_kwargs.pop("n_init", "auto")
    return kmeans_kwargs, n_init


def find_best_k_geometric(
    X_maj: NDArray[np.float64], 
    k_candidates: List[int],
    random_state: int,
    kmeans_params: Optional[dict] = None
) -> int:
    """
    Selects a cluster count using the highest observed Silhouette score.

    Evaluates a list of candidate values for k by applying K-Means clustering
    and selecting the value that maximizes the observed Silhouette Score. Keys
    owned by KMeans-ReSC are rejected in user-provided dictionaries. A single
    feasible candidate is evaluated normally. If no candidate can be scored,
    the method returns 1.

    Args:
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the majority class.
        k_candidates (List[int]): A list of integer candidate values for the number of clusters (k) to test.
        random_state (int): Seed used by the random number generator for K-Means initialization.
        kmeans_params (dict, optional): Additional keyword arguments to pass safely to KMeans.

    Returns:
        int: The selected number of clusters.
    """
    best_k = None
    best_score = -np.inf
    kmeans_kwargs, n_init_val = _split_kmeans_params(kmeans_params)

    for k in k_candidates:
        if k < 2 or k >= len(X_maj):
            continue

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init_val,
            **kmeans_kwargs
        )
        cluster_labels = kmeans.fit_predict(X_maj)

        if np.unique(cluster_labels).size < 2:
            continue

        score = silhouette_score(X_maj, cluster_labels)

        if score > best_score:
            best_score = score
            best_k = k

    return 1 if best_k is None else best_k


def get_safe_majority_samples_knn(
    X: NDArray[np.float64],
    y: NDArray[Any],
    X_maj: NDArray[np.float64],
    maj_label: Union[int, str, float],
    n_neighbors: int = 5,
    threshold: float = 0.9,
    knn_params: Optional[dict] = None
) -> Tuple[NDArray[np.float64], bool]:
    """
    Identifies majority inputs that pass the configured local-score threshold.

    The K-Nearest Neighbors model is fitted on the complete training dataset.
    Each queried majority observation is included in its own exact Euclidean
    neighborhood, and uniform voting defines its local majority score. The term
    "safe" is only an operational threshold label, not a non-overlap guarantee.

    If the threshold is too strict and filters out every single majority sample, 
    the function restores the original, unfiltered majority collection.

    Args:
        X (numpy.typing.NDArray[np.float64]): 2D NumPy array of the entire training dataset's features.
        y (numpy.typing.NDArray[Any]): 1D NumPy array of the entire training dataset's labels.
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing only the features of the majority class.
        maj_label (Union[int, str, float]): The target label assigned to the majority class.
        n_neighbors (int, optional): Number of neighbors to use for the KNN density estimation. Defaults to 5.
        threshold (float, optional): The minimum probability required (0.0 to 1.0) for a sample to be considered safe. Defaults to 0.9.
        knn_params (dict, optional): Additional keyword arguments to pass safely to KNeighborsClassifier.

    Returns:
        tuple: The filtered majority array and whether empty-filter fallback was used.
    """
    if n_neighbors > len(X):
        raise ValueError(
            f"n_neighbors={n_neighbors} must not exceed n_samples={len(X)}."
        )

    knn_kwargs = validate_knn_params(knn_params)
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="euclidean",
        weights="uniform",
        **knn_kwargs,
    )
    knn.fit(X, y)

    neighbor_indices = knn.kneighbors(X_maj, return_distance=False).copy()
    majority_indices = np.flatnonzero(y == maj_label)

    # Explicitly include each queried training observation. This matters when
    # duplicate points tie at distance zero and the neighbor search returns a
    # different duplicate instead of the queried index.
    for row, sample_index in enumerate(majority_indices):
        if sample_index not in neighbor_indices[row]:
            neighbor_indices[row, -1] = sample_index

    majority_scores = np.mean(y[neighbor_indices] == maj_label, axis=1)
    safe_mask = majority_scores >= threshold
    X_maj_safe = X_maj[safe_mask]

    fallback_used = len(X_maj_safe) == 0
    if len(X_maj_safe) == 0:
        X_maj_safe = X_maj

    return X_maj_safe, fallback_used


def get_set_n_kmeans_re_sc(
    X: NDArray[np.float64],
    y: NDArray[Any],
    min_label: Union[int, str, float],
    maj_label: Union[int, str, float],
    M: float = 1.5,
    random_state: int = 42,
    n_neighbors: int = 5,
    safe_threshold: float = 0.9,
    knn_params: Optional[dict] = None,
    kmeans_params: Optional[dict] = None
) -> Tuple[NDArray[np.float64], bool]:
    """
    Generates centroid representatives from threshold-filtered majority inputs.

    This function filters majority inputs, constructs the complete bounded cluster
    count grid, selects the highest observed feasible Silhouette score, and returns
    every center from the final KMeans fit. The centroids are not filtered again.

    Args:
        X (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the entire training dataset's features.
        y (numpy.typing.NDArray[Any]): 1D NumPy array containing the target labels (supports categorical strings).
        min_label (Union[int, str, float]): The target label assigned to the minority class.
        maj_label (Union[int, str, float]): The target label assigned to the majority class.
        M (float, optional): Multiplier used to define the upper candidate bound. Defaults to 1.5.
        random_state (int, optional): Seed used for reproducibility in K-Means initialization. Defaults to 42.
        n_neighbors (int, optional): Number of neighbors for the KNN safety check. Defaults to 5.
        safe_threshold (float, optional): Minimum required KNN probability for a sample to be "safe". Defaults to 0.9.
        knn_params (dict, optional): Additional keyword arguments to pass to KNeighborsClassifier.
        kmeans_params (dict, optional): Additional keyword arguments to pass to KMeans.

    Returns:
        tuple: The selected KMeans centers and safety-filter fallback diagnostic.

    Raises:
        ValueError: If either the minority or majority class contains zero samples.
    """
    X_min = X[y == min_label]
    X_maj = X[y == maj_label]
    
    n_maj = len(X_maj)
    n_min = len(X_min)
    
    if n_maj == 0 or n_min == 0:
        raise ValueError("Both minority and majority classes must have at least one sample.")
    
    n1 = int((n_min ** 2) / n_maj)
    k_max = max(1, int(M * n1))

    X_maj_safe, fallback_used = get_safe_majority_samples_knn(
        X=X, 
        y=y, 
        X_maj=X_maj, 
        maj_label=maj_label,
        n_neighbors=n_neighbors,
        threshold=safe_threshold,
        knn_params=knn_params
    )

    candidates = list(range(1, k_max + 1))

    best_k = find_best_k_geometric(
        X_maj_safe,
        candidates,
        random_state,
        kmeans_params=kmeans_params
    )
    best_k = min(best_k, len(X_maj_safe))

    kmeans_kwargs, n_init_val = _split_kmeans_params(kmeans_params)

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=n_init_val,
        **kmeans_kwargs
    )
    kmeans.fit(X_maj_safe)

    X_set_n = kmeans.cluster_centers_

    return X_set_n, fallback_used


def kmeans_re_sc_concatenation(
    X_min: NDArray[np.float64], 
    X_maj: NDArray[np.float64], 
    X_set_n: NDArray[np.float64],
    min_label: Union[int, str, float] = 1,
    maj_label: Union[int, str, float] = -1
) -> Tuple[NDArray[np.float64], NDArray[Any]]:
    """
    Concatenates pairs of samples from the same class to map the dataset into a 2d dimensional space.

    The minority class is augmented by horizontally stacking repeated and tiled
    permutations of itself. The majority class is paired directly with the KMeans
    centroid collection generated previously.

    Args:
        X_min (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the minority class.
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the original majority class.
        X_set_n (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the calculated cluster centers.
        min_label (Union[int, str, float], optional): The target label assigned to the minority class. Defaults to 1.
        maj_label (Union[int, str, float], optional): The target label assigned to the majority class. Defaults to -1.

    Returns:
        Tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[Any]]: 
            A tuple containing:
            - X_resampled: The concatenated 2D NumPy array with 2*d features.
            - y_resampled: The 1D NumPy array containing the target labels (supports categorical strings).

    Raises:
        ValueError: If X_min is empty, as the minority class must contain at least one sample to concatenate.
    """
    m = len(X_min)
    if m == 0:
        raise ValueError("X_min cannot be empty. Minority class must have at least one sample.")
    
    P_repeat = np.repeat(X_min, m, axis=0)
    P_tile = np.tile(X_min, (m, 1))
    P_c = np.hstack([P_repeat, P_tile]) 
    
    y_p_c = np.full(len(P_c), min_label)

    M = len(X_maj)
    k = len(X_set_n)
    
    if k > 0:
        N_repeat = np.repeat(X_maj, k, axis=0)  
        C_tile = np.tile(X_set_n, (M, 1))
        N_c = np.hstack([N_repeat, C_tile])
        
        y_n_c = np.full(len(N_c), maj_label)
        
        X_c_array = np.vstack([P_c, N_c])
        y_c_array = np.hstack([y_p_c, y_n_c])
    else:
        X_c_array = P_c
        y_c_array = y_p_c

    return X_c_array, y_c_array
