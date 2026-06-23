from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

def find_best_k_geometric(
    X_maj: NDArray[np.float64], 
    k_candidates: List[int],
    random_state: int
) -> int:
    """
    Finds the optimal number of clusters (k) using the Silhouette Score.

    Evaluates a list of candidate values for k by applying K-Means clustering 
    and selecting the value that maximizes the Silhouette Score. If only one 
    valid candidate is provided or all candidates are invalid, it provides a 
    safe fallback.

    Args:
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the majority class.
        k_candidates (List[int]): A list of integer candidate values for the number of clusters (k) to test.
        random_state (int): Seed used by the random number generator for K-Means initialization to ensure reproducibility.

    Returns:
        int: The optimal number of clusters (k) selected from the candidates.
    """
    if len(k_candidates) == 1:
        single_k = k_candidates[0] if k_candidates[0] != 0 else 1  
        return single_k

    best_k = None
    best_score = -2.0 
    
    for k in k_candidates:
        if k < 2 or k >= len(X_maj):
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_maj)
        
        score = silhouette_score(X_maj, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            
    if best_k is None:
        best_k = max(1, k_candidates[0])
        
    return best_k


def get_set_n_kmeans_re_sc(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    min_label: int,
    maj_label: int,
    M: float = 1.5,
    num_candidates_to_test: int = 5,
    random_state: int = 42
) -> NDArray[np.float64]:
    """
    Generates the Set_N subset using K-Means clustering on 'safe' majority samples.

    This function identifies 'safe' majority samples using a KNN classifier (samples 
    with >= 90% probability of belonging to the majority class). It then dynamically 
    generates a list of candidate values for the number of clusters based on the 
    imbalance ratio bounds, finds the optimal k using the Silhouette Score, and 
    returns the resulting K-Means cluster centers to be used as Set_N.

    Args:
        X (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the entire training dataset.
        y (numpy.typing.NDArray[np.int_]): 1D NumPy array containing the target labels.
        min_label (int): The target label assigned to the minority class.
        maj_label (int): The target label assigned to the majority class.
        M (float, optional): The maximum acceptable imbalance ratio threshold. Defaults to 1.5.
        num_candidates_to_test (int, optional): The number of candidate values for k to evaluate during geometric tuning. Defaults to 5.
        random_state (int, optional): Seed used by the random number generator for reproducibility. Defaults to 42.

    Returns:
        numpy.typing.NDArray[np.float64]: A 2D NumPy array containing the features of the generated majority subset (the K-Means cluster centers).

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
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    maj_class_idx = np.where(knn.classes_ == maj_label)[0][0]
    
    probs = knn.predict_proba(X_maj)
    prob_majority = probs[:, maj_class_idx]
    
    safe_mask = prob_majority >= 0.9
    X_maj_safe = X_maj[safe_mask]
    
    if len(X_maj_safe) == 0:
        X_maj_safe = X_maj 
        
    step = max(1, (upper_bound - n1) // max(1, (num_candidates_to_test - 1)))
    candidates = list(range(n1, upper_bound + 1, step))
    
    best_k = find_best_k_geometric(X_maj_safe, candidates, random_state)
    best_k = min(best_k, len(X_maj_safe))
    
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init='auto')
    kmeans.fit(X_maj_safe)
    
    X_set_n = kmeans.cluster_centers_
    
    return X_set_n

def kmeans_re_sc_concatenation(
    X_min: NDArray[np.float64], 
    X_maj: NDArray[np.float64], 
    X_set_n: NDArray[np.float64],
    min_label: int = 1,
    maj_label: int = -1
) -> Tuple[NDArray[np.float64], NDArray[np.int_]]:
    """
    Concatenates pairs of samples from the same class to map the dataset into a 2d dimensional space.

    Args:
        X_min (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the minority class.
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the original majority class.
        X_set_n (numpy.typing.NDArray[np.float64]): 2D NumPy array containing the features of the selected majority subset.
        min_label (int, optional): The target label assigned to the minority class.
        maj_label (int, optional): The target label assigned to the majority class.

    Returns:
        Tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[np.int_]]: 
            A tuple containing:
            - X_resampled: The concatenated 2D NumPy array with 2 * d features.
            - y_resampled: The 1D NumPy array containing the target labels for the new samples.

    Raises:
        ValueError: If X_min is empty, as the minority class must contain at least one sample.
    """
    m = len(X_min)
    if m == 0:
        raise ValueError("X_min cannot be empty. Minority class must have at least one sample.")
    
    P_repeat = np.repeat(X_min, m, axis=0)
    P_tile = np.tile(X_min, (m, 1))
    P_c = np.hstack([P_repeat, P_tile]) 
    
    y_p_c = np.full(len(P_c), min_label, dtype=np.int_)

    M = len(X_maj)
    k = len(X_set_n)
    
    if k > 0:
        N_repeat = np.repeat(X_maj, k, axis=0)  
        Set_N_tile = np.tile(X_set_n, (M, 1)) 
        N_c = np.hstack([N_repeat, Set_N_tile])
        y_n_c = np.full(len(N_c), maj_label, dtype=np.int_)
        
        X_c_array = np.vstack([P_c, N_c])
        y_c_array = np.hstack([y_p_c, y_n_c])
    else:
        X_c_array = P_c
        y_c_array = y_p_c

    return X_c_array, y_c_array