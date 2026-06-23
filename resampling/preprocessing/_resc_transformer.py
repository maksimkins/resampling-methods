from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class ReSCTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms test data into a 2d concatenated space for prediction with Re-SC models.
    
    This transformer must be placed immediately after a Re-SC sampler in a Pipeline.
    - During training (fit_transform), it receives data that the Sampler has already 
      mapped to 2d space, so it simply passes it through.
    - During testing (predict), the Sampler is bypassed, so this Transformer receives 
      data in the original d space. It duplicates the features (x -> [x, x]) to 
      match the 2d space the classifier expects.
    """

    def __init__(self):
        pass

    def fit(self, X: NDArray[np.float64], y: Optional[NDArray] = None) -> 'ReSCTransformer':
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.original_d_ = self.n_features_in_ // 2
        self.is_fitted_ = True 

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)

        if X.shape[1] == self.n_features_in_:
            return X

        if X.shape[1] == self.original_d_:
            return np.hstack([X, X])

        raise ValueError(
            f"X has {X.shape[1]} features. ReSCTransformer expects either "
            f"{self.original_d_} (raw test data) or {self.n_features_in_} (resampled train data)."
        )

    def get_feature_names_out(
        self, 
        input_features: Optional[Union[List[str], NDArray[np.object_]]] = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.original_d_)]

        out_features = [f"{name}_1" for name in input_features] + \
                       [f"{name}_2" for name in input_features]

        return np.asarray(out_features, dtype=object)