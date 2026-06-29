from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import scipy.stats as stats

from sklearn.neighbors import NearestNeighbors

def calculate_set_n_size_re_sc(
    X_maj: NDArray[np.float64], 
    P: int, 
    alpha: float = 0.05, 
    epsilon: float = 0.05, 
    M: float = 1.5
) -> int:
    """
    Calculates the target size for the majority subset (Set_N) in the Re-SC algorithm.

    Args:
        X_maj (numpy.typing.NDArray[np.float64]): 2D NumPy array containing features of the majority class.
        P (int): number of samples in the minority class.
        alpha (float): significance level for the Z-test used to compute statistical sample size.
        epsilon (float): acceptable tolerance error for representing the majority class distribution
        M (float): acceptable imbalance ratio threshold.

    Returns:
        int: The calculated number of majority samples for Set_N.

    Raises:
        ValueError: If X_maj is an empty array. 
    """ 
    n_maj: int = len(X_maj)
    if n_maj == 0:
        raise ValueError("X_maj cannot be empty.")
    
    n1 = (P ** 2) / n_maj
    z_score = stats.norm.ppf(1 - alpha / 2)
    sigma = np.std(X_maj, ddof=1) 
    z_sq = z_score ** 2
    sigma_sq = sigma ** 2
    epsilon_sq = epsilon ** 2
    
    numerator = n_maj * z_sq * sigma_sq 
    denominator = (n_maj * epsilon_sq) + (z_sq * sigma_sq) 
    n2 = numerator / denominator

    if n1 == 0: 
        return int(n2)
    
    Pr = n2 / n1

    if Pr < 1:
        set_n_size = n1
    elif 1 <= Pr <= M:
        set_n_size = n2
    else: 
        set_n_size = M * n1
        
    return int(np.ceil(set_n_size))

def get_set_n_random_weighted_re_sc(
    X: NDArray[np.float64], 
    y: NDArray[np.int_], 
    n_size: int, 
    k: int = 5
) -> NDArray[np.float64]:
    """
    Selects a subset of majority class samples (Set_N) using a density-weighted random sampling strategy.

    Args:
        X (numpy.typing.NDArray[np.float64]): 2D NumPy array containing entire training dataset.
        y (numpy.typing.NDArray[np.int_]): 1D NumPy array containing labels.
        n_size (int): number of majority samples to select.
        k (int, optional): number of nearest neighbors to evaluate for the weighting mechanism. Defaults to 5.

    Returns:
        numpy.typing.NDArray[np.float64]: A 2D NumPy array containing selected majority subset.

    Raises:
        ValueError: If there are no majority samples present in the dataset.
        ValueError: If no valid majority samples are found.
    """
    labels, counts = np.unique(y, return_counts=True)
    maj_label = labels[np.argmax(counts)]

    maj_mask = (y == maj_label)
    maj_indices = np.where(maj_mask)[0]
    
    if len(maj_indices) == 0:
        raise ValueError("No majority samples found in the dataset.")
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0 
    X_norm = (X - X_mean) / X_std
    
    knn = NearestNeighbors(n_neighbors=k + 1).fit(X_norm)
    
    X_maj_norm = X_norm[maj_indices]
    distances, neighbor_idxs = knn.kneighbors(X_maj_norm)
    
    weights = []
    valid_indices = []
    
    for i, n_idxs in enumerate(neighbor_idxs):
        actual_neighbors = n_idxs[1:]  
        maj_neighbor_count = np.sum(y[actual_neighbors] == maj_label)
        weight = maj_neighbor_count / k

        if weight > 0:
            weights.append(weight)
            valid_indices.append(maj_indices[i])
            
    weights_arr = np.array(weights, dtype=np.float64)
    valid_indices_arr = np.array(valid_indices, dtype=np.int_)
    
    if len(weights_arr) == 0:
        raise ValueError("No valid majority samples found (all are surrounded by minority class).")
        
    probs = weights_arr / np.sum(weights_arr)
    actual_n_size = min(n_size, len(valid_indices_arr))

    selected_indices = np.random.choice(
        valid_indices_arr, 
        size=actual_n_size, 
        replace=False, 
        p=probs
    )
    
    return X[selected_indices]

def re_sc_concatenation(
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