from typing import Optional, Union, List, Tuple, Any
from numbers import Real, Integral

import numpy as np
from numpy.typing import NDArray

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from imblearn.base import BaseSampler

from .utils._resc_kmeans_utils import (
    get_set_n_kmeans_re_sc,
    kmeans_re_sc_concatenation,
    validate_kmeans_params,
    validate_knn_params,
)

class KMeansReSC(BaseSampler):
    """
    Resampling based on Sample Concatenation (Re-SC) using K-Means clustering.
    
    This algorithm addresses class imbalance by mapping the data into a higher-dimensional 
    (2d) concatenated feature space. It threshold-filters majority inputs using KNN,
    selects a number of clusters from a bounded complete grid using the highest
    observed Silhouette Score, and uses the resulting K-Means centers as majority
    representatives.

    Attributes:
        M (float): Multiplier used to define the upper candidate bound.
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
        "M": [Interval(Real, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "safe_threshold": [Interval(Real, 0.0, 1.0, closed="both")],
        "knn_params": [dict, None],
        "kmeans_params": [dict, None]
    }
    
    def __init__(
        self, 
        M=1.5,
        random_state=None,
        n_neighbors=5,
        safe_threshold=0.9,
        knn_params=None,
        kmeans_params=None
    ):
        super().__init__()
        self.M = M
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
            ValueError: If the dataset is not strictly imbalanced binary data.
        """

        labels, counts = self._validate_target_domain(y)
        if not np.isfinite(self.M):
            raise ValueError("M must be finite.")
        if self.n_neighbors > len(X):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must not exceed "
                f"n_samples={len(X)}."
            )

        # Validate ownership before running the safety filter or any KMeans fit.
        validate_knn_params(self.knn_params)
        validate_kmeans_params(self.kmeans_params)

        random_state_obj = check_random_state(self.random_state)
        seed = random_state_obj.randint(0, 2**31 - 1)

        min_label = labels[np.argmin(counts)]
        maj_label = labels[np.argmax(counts)]

        X_set_n, fallback_used = get_set_n_kmeans_re_sc(
            X=X,
            y=y,
            min_label=min_label,
            maj_label=maj_label,
            M=self.M,
            random_state=seed,
            n_neighbors=self.n_neighbors,
            safe_threshold=self.safe_threshold,
            knn_params=self.knn_params,
            kmeans_params=self.kmeans_params
        )
        self.fallback_used_ = bool(fallback_used)

        X_resampled, y_resampled = kmeans_re_sc_concatenation(
            X_min=X[y == min_label],
            X_maj=X[y == maj_label],
            X_set_n=X_set_n,
            min_label=min_label,
            maj_label=maj_label
        )

        return X_resampled, y_resampled

    @staticmethod
    def _validate_target_domain(y):
        """Validate the strictly imbalanced binary domain used by the paper."""
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        if len(labels) != 2:
            raise ValueError(
                "KMeansReSC requires exactly two target classes; "
                f"got {len(labels)}."
            )
        if counts[0] == counts[1]:
            raise ValueError(
                "KMeansReSC requires strictly unequal class counts so minority "
                "and majority classes are unambiguous."
            )
        return labels, counts

    def fit_resample(self, X, y, **params):
        """Resample while preserving pandas containers across the d-to-2d map."""
        self._validate_target_domain(y)
        is_dataframe = (
            X.__class__.__name__ == "DataFrame"
            and X.__class__.__module__.startswith("pandas")
        )
        if not is_dataframe:
            return super().fit_resample(X, y, **params)

        import pandas as pd

        input_features = np.asarray(X.columns, dtype=object)
        y_array = np.asarray(y)
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.ravel()

        X_resampled, y_resampled = super().fit_resample(
            X.to_numpy(),
            y_array,
            **params,
        )

        if all(isinstance(name, str) for name in input_features):
            self.feature_names_in_ = input_features

        output_columns = self.get_feature_names_out(input_features)
        X_output = pd.DataFrame(X_resampled, columns=output_columns)

        if (
            y.__class__.__name__ == "Series"
            and y.__class__.__module__.startswith("pandas")
        ):
            y_output = pd.Series(
                y_resampled,
                index=X_output.index,
                name=getattr(y, "name", None),
            )
        elif (
            y.__class__.__name__ == "DataFrame"
            and y.__class__.__module__.startswith("pandas")
        ):
            y_output = pd.DataFrame(
                np.asarray(y_resampled).reshape(-1, 1),
                index=X_output.index,
                columns=y.columns,
            )
        elif isinstance(y, list):
            y_output = np.asarray(y_resampled).tolist()
        else:
            y_output = y_resampled

        return X_output, y_output

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
            input_features = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)],
            )

        out_features = [f"{name}_1" for name in input_features] + [f"{name}_2" for name in input_features]

        return np.asarray(out_features, dtype=object)
