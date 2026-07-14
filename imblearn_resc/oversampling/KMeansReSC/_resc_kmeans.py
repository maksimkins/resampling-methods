from typing import Optional, Union, List, Tuple, Any
from numbers import Real, Integral

import numpy as np
from numpy.typing import NDArray

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from imblearn.base import BaseSampler

from .utils._resc_kmeans_utils import (
    get_set_n_kmeans_re_sc, 
    kmeans_re_sc_concatenation
)

class KMeansReSC(BaseSampler):
    """
    Resampling based on Sample Concatenation (Re-SC) using K-Means clustering.
    
    This algorithm addresses class imbalance by mapping the data into a higher-dimensional 
    (2d) concatenated feature space. It identifies "safe" majority samples using KNN, 
    determines the optimal number of clusters (k) via Silhouette Score, and uses the 
    resulting K-Means cluster centers as the majority subset (Set_N).

    Attributes:
        M (float): The maximum acceptable imbalance ratio threshold for the resulting dataset.
        num_candidates_to_test (int): How many 'k' values to test during geometric tuning.
        random_state (int, RandomState instance, default=None): Controls the randomization.
        n_neighbors (int): Number of neighbors to use for the KNN safety check.
        safe_threshold (float): Minimum probability required for a majority sample to be safe.
        knn_params (dict, optional): Additional keyword arguments to pass to KNeighborsClassifier.
        kmeans_params (dict, optional): Additional keyword arguments to pass to KMeans.

    Methods:
        _fit_resample(X, y): Core resampling logic that executes KMeansReSC and returns concatenated arrays.
        get_feature_names_out(input_features): Generates output feature names for the 2d concatenated space.
    """
    _sampling_type = 'over-sampling'
    
    _parameter_constraints = {
        "M": [Interval(Real, 0, None, closed="left")],
        "num_candidates_to_test": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "safe_threshold": [Interval(Real, 0.0, 1.0, closed="both")],
        "knn_params": [dict, None],
        "kmeans_params": [dict, None]
    }
    
    def __init__(
        self, 
        M=1.5, 
        num_candidates_to_test=5, 
        random_state=None,
        n_neighbors=5,
        safe_threshold=0.9,
        knn_params=None,
        kmeans_params=None
    ):
        super().__init__()
        self.M = M
        self.num_candidates_to_test = num_candidates_to_test
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.safe_threshold = safe_threshold
        self.knn_params = knn_params
        self.kmeans_params = kmeans_params

    def _fit_resample(
        self, 
        X: NDArray[np.float64], 
        y: NDArray[Any]  
    ) -> Tuple[NDArray[np.float64], NDArray[Any]]:
        """
        Executes resampling logic for KMeansReSC.

        Args:
            X (numpy.typing.NDArray[np.float64]): 2D matrix containing the features of the original training dataset.
            y (numpy.typing.NDArray[Any]): 1D array containing the target labels.

        Returns:
            Tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[Any]]: 
                A tuple containing the resampled feature matrix (mapped to a 2d space) 
                and the corresponding label array.

        Raises:
            ValueError: If the dataset does not contain at least two distinct classes.
        """

        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise ValueError("The target 'y' needs to have at least two classes.")
            
        random_state_obj = check_random_state(self.random_state)
        seed = random_state_obj.randint(0, 2**31 - 1)
        
        min_label = labels[np.argmin(counts)]
        maj_label = labels[np.argmax(counts)]

        X_set_n = get_set_n_kmeans_re_sc(
            X=X,
            y=y,
            min_label=min_label,
            maj_label=maj_label,
            M=self.M,
            num_candidates_to_test=self.num_candidates_to_test,
            random_state=seed,
            n_neighbors=self.n_neighbors,
            safe_threshold=self.safe_threshold,
            knn_params=self.knn_params,
            kmeans_params=self.kmeans_params
        )

        X_resampled, y_resampled = kmeans_re_sc_concatenation(
            X_min=X[y == min_label], 
            X_maj=X[y == maj_label], 
            X_set_n=X_set_n,
            min_label=min_label,
            maj_label=maj_label
        )

        return X_resampled, y_resampled

    def get_feature_names_out(
        self, 
        input_features: Optional[Union[List[str], NDArray[np.object_]]] = None
    ) -> NDArray[np.object_]:
        """
        Get output feature names for transformation. 

        Args:
            input_features (Optional[Union[List[str], numpy.typing.NDArray[np.object_]]]): 
                Original input feature names. If None, generic names are generated.

        Returns:
            numpy.typing.NDArray[np.object_]: An array of strings containing the new feature 
                names for the 2d concatenated space.
        """
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        out_features = [f"{name}_1" for name in input_features] + [f"{name}_2" for name in input_features]

        return np.asarray(out_features, dtype=object)