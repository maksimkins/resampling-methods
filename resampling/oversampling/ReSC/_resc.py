from typing import Optional, Union, List, Tuple
from numbers import Real, Integral

import numpy as np
from numpy.typing import NDArray

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from imblearn.base import BaseSampler

from .utils._resc_utils import (
    calculate_set_n_size_re_sc,
    get_set_n_random_weighted_re_sc,
    re_sc_concatenation
)


class ReSC(BaseSampler):
    """
    Resampling based on Sample Concatenation (Re-SC) using density-weighted random sampling.
    
    This algorithm addresses class imbalance by mapping the data into a higher-dimensional 
    (2d) concatenated feature space. It over-samples the minority class by concatenating 
    minority samples with themselves, and under-samples the majority class by pairing 
    original majority samples with a statistically determined subset (Set_N).

    Attributes:
        M (float): The maximum acceptable imbalance ratio threshold for the resulting dataset.
        k (int): Number of nearest neighbors used to calculate majority sample weights.
        alpha (float): Significance level for the Z-test used to compute the required statistical sample size.
        epsilon (float): Acceptable tolerance error for representing the majority class distribution.
        random_state (int, RandomState instance, default=None): Controls the randomization of the algorithm.

    Methods:
        _fit_resample(X, y): Core resampling logic that executes Re-SC1 and returns concatenated arrays.
        get_feature_names_out(input_features): Generates output feature names for the 2d concatenated space.
    """
    _sampling_type = 'over-sampling'
    _parameter_constraints = {
        "M": [Interval(Real, 0, None, closed="left")],          
        "k": [Interval(Integral, 1, None, closed="left")],      
        "alpha": [Interval(Real, 0, 1, closed="both")],         
        "epsilon": [Interval(Real, 0, None, closed="neither")], 
        "random_state": ["random_state"]                        
    }
    def __init__(self, M=1.5, k=5, alpha=0.05, epsilon=0.05, random_state=None):
        super().__init__()
        self.M = M
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.random_state = random_state

    def _fit_resample(
        self, 
        X: NDArray[np.float64], 
        y: NDArray[np.int_]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int_]]:
        """
        Executes resampling logic for Re-SC.

        Args:
            X (numpy.typing.NDArray[np.float64]): 2D matrix containing the features of the original training dataset.
            y (numpy.typing.NDArray[np.int_]): 1D array containing the target labels.

        Returns:
            Tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[np.int_]]: 
                A tuple containing the resampled feature matrix (mapped to a 2d space) 
                and the corresponding label array.

        Raises:
            ValueError: If the dataset does not contain at least two distinct classes.
        """ 
        random_state = check_random_state(self.random_state)
        np.random.seed(random_state.randint(0, 2**32 - 1))
        
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise ValueError("The target 'y' needs to have at least two classes.")
            
        min_label = labels[np.argmin(counts)]
        maj_label = labels[np.argmax(counts)]

        X_min = X[y == min_label]
        X_maj = X[y == maj_label]

        target_size = calculate_set_n_size_re_sc(
            X_maj=X_maj, 
            P=len(X_min), 
            alpha=self.alpha, 
            epsilon=self.epsilon, 
            M=self.M
        )
        X_set_n = get_set_n_random_weighted_re_sc(
            X=X, 
            y=y, 
            n_size=target_size, 
            k=self.k
        )
        X_resampled, y_resampled = re_sc_concatenation(
            X_min=X_min, 
            X_maj=X_maj, 
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