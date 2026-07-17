from typing import Tuple, List, Union, Any, Optional

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score


def find_best_k_geometric(
    X_maj: NDArray[np.float64], 
    k_candidates: List[int],
    random_state: int,
    kmeans_params: Optional[dict] = None
) -> int:
    """
    Finds the optimal number of clusters (k) using the Silhouette Score.

    Evaluates a list of candidate values for k by applying K-Means clustering 
    and selecting the value that maximizes the Silhouette Score. It automatically 
    safeguards against overriding the core algorithmic parameters by popping 
    colliding keys from user-provided dictionaries. If only one valid candidate 
    is provided or all candidates are invalid, it provides a safe fallback.

    Args:
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the majority class.
        k_candidates (List[int]): A list of integer candidate values for the number of clusters (k) to test.
        random_state (int): Seed used by the random number generator for K-Means initialization.
        kmeans_params (dict, optional): Additional keyword arguments to pass safely to KMeans.

    Returns:
        int: The optimal number of clusters (k) selected from the candidates.
    """
    if len(k_candidates) == 1:
        single_k = k_candidates[0] if k_candidates[0] != 0 else 1  
        return single_k

    best_k = None
    best_score = -2.0 
    
    kmeans_kwargs = dict(kmeans_params) if kmeans_params is not None else {}
    n_init_val = kmeans_kwargs.pop('n_init', 'auto')
    
    kmeans_kwargs.pop('n_clusters', None)
    
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
        
        score = silhouette_score(X_maj, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            
    if best_k is None:
        best_k = max(1, k_candidates[0])
        
    return best_k


def get_safe_majority_samples_knn(
    X: NDArray[np.float64],
    y: NDArray[Any],
    X_maj: NDArray[np.float64],
    maj_label: Union[int, str, float],
    n_neighbors: int = 5,
    threshold: float = 0.9,
    knn_params: Optional[dict] = None
) -> NDArray[np.float64]:
     """
    Identifies 'safe' majority samples using a KNN classifier.

    A sample is considered safe if the KNN classifier predicts its probability 
    of belonging to the majority class is greater than or equal to the threshold.
    If no samples meet this criteria, the original majority set is returned.

    Args:
        X (numpy.typing.NDArray[np.float64]): Features of the entire training dataset.
        y (numpy.typing.NDArray[Any]): Target labels of the entire training dataset.
        X_maj (numpy.typing.NDArray[np.float64]): Features of the majority class subset.
        maj_label (Union[int, str, float]): The target label assigned to the majority class.
        n_neighbors (int, optional): Number of neighbors to use for KNN. Defaults to 5.
        threshold (float, optional): The minimum probability required to be considered safe. Defaults to 0.9.

    Returns:
        numpy.typing.NDArray[np.float64]: A 2D NumPy array containing the 'safe' majority samples.
    """
    knn_kwargs = dict(knn_params) if knn_params is not None else {}
    knn_kwargs.pop('n_neighbors', None)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, **knn_kwargs)
    knn.fit(X, y)
    
    maj_class_idx = np.where(knn.classes_ == maj_label)[0][0]
    
    probs = knn.predict_proba(X_maj)
    prob_majority = probs[:, maj_class_idx]
    
    safe_mask = prob_majority >= threshold
    X_maj_safe = X_maj[safe_mask]
    
    if len(X_maj_safe) == 0:
        X_maj_safe = X_maj 
        
    return X_maj_safe


def get_set_n_kmeans_re_sc(
    X: NDArray[np.float64],
    y: NDArray[Any],
    min_label: Union[int, str, float],
    maj_label: Union[int, str, float],
    M: float = 1.5,
    num_candidates_to_test: int = 5,
    random_state: int = 42,
    n_neighbors: int = 5,
    safe_threshold: float = 0.9,
    knn_params: Optional[dict] = None,
    kmeans_params: Optional[dict] = None
) -> NDArray[np.float64]:
    """
    Generates the Set_N subset using K-Means clustering on 'safe' majority samples.

    This function identifies 'safe' majority samples using a KNN classifier (dropping 
    samples near the minority boundary). It dynamically calculates candidate values 
    for the number of clusters based on acceptable imbalance ratio bounds, finds 
    the optimal k via Silhouette Score, and returns the K-Means cluster centers.

    Args:
        X (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the entire training dataset's features.
        y (numpy.typing.NDArray[Any]): 1D NumPy array containing the target labels (supports categorical strings).
        min_label (Union[int, str, float]): The target label assigned to the minority class.
        maj_label (Union[int, str, float]): The target label assigned to the majority class.
        M (float, optional): Maximum acceptable imbalance ratio threshold. Defaults to 1.5.
        num_candidates_to_test (int, optional): Number of k candidates to evaluate during geometric tuning. Defaults to 5.
        random_state (int, optional): Seed used for reproducibility in K-Means initialization. Defaults to 42.
        n_neighbors (int, optional): Number of neighbors for the KNN safety check. Defaults to 5.
        safe_threshold (float, optional): Minimum required KNN probability for a sample to be "safe". Defaults to 0.9.
        knn_params (dict, optional): Additional keyword arguments to pass to KNeighborsClassifier.
        kmeans_params (dict, optional): Additional keyword arguments to pass to KMeans.

    Returns:
        numpy.typing.NDArray[np.float64]: A 2D NumPy array containing the features of the selected K-Means cluster centers.

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
    upper_bound = int(M * n1)
    
    X_maj_safe = get_safe_majority_samples_knn(
        X=X, 
        y=y, 
        X_maj=X_maj, 
        maj_label=maj_label,
        n_neighbors=n_neighbors,
        threshold=safe_threshold,
        knn_params=knn_params
    )
        
    step = max(1, (upper_bound - n1) // max(1, (num_candidates_to_test - 1)))
    candidates = list(range(n1, upper_bound + 1, step))
    
    best_k = find_best_k_geometric(
        X_maj_safe, 
        candidates, 
        random_state,
        kmeans_params=kmeans_params
    )
    best_k = min(best_k, len(X_maj_safe))
    
    kmeans_kwargs = dict(kmeans_params) if kmeans_params is not None else {}
    n_init_val = kmeans_kwargs.pop('n_init', 'auto')
    
    kmeans_kwargs.pop('n_clusters', None)
    
    kmeans = KMeans(
        n_clusters=best_k, 
        random_state=random_state, 
        n_init=n_init_val, 
        **kmeans_kwargs
    )
    kmeans.fit(X_maj_safe)
    
    X_set_n = kmeans.cluster_centers_
    
    return X_set_n


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
    permutations of itself. The majority class is paired directly with the K-Means 
    cluster centers (Set_N) generated previously.

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
        Set_N_tile = np.tile(X_set_n, (M, 1)) 
        N_c = np.hstack([N_repeat, Set_N_tile])
        
        y_n_c = np.full(len(N_c), maj_label)
        
        X_c_array = np.vstack([P_c, N_c])
        y_c_array = np.hstack([y_p_c, y_n_c])
    else:
        X_c_array = P_c
        y_c_array = y_p_c

    return X_c_array, y_c_array