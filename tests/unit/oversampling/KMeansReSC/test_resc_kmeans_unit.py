import pytest
import numpy as np
from unittest.mock import patch

from imblearn_resc.oversampling import KMeansReSC

@pytest.fixture
def dummy_data():
    """Provides a basic imbalanced dataset for testing."""
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    y = np.array([0, 0, 0, 1])
    return X, y


def test_kmeans_resc_init():
    """Test that parameters are assigned correctly upon initialization."""
    sampler = KMeansReSC(M=2.0, num_candidates_to_test=10, random_state=42)
    assert sampler.M == 2.0
    assert sampler.num_candidates_to_test == 10
    assert sampler.random_state == 42


def test_kmeans_resc_single_class_error():
    """Test that the sampler catches datasets with only 1 class."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 0])  

    sampler = KMeansReSC()
    with pytest.raises(ValueError, match="needs to have at least two classes"):
        sampler._fit_resample(X, y)


@patch("imblearn_resc.oversampling.KMeansReSC._resc_kmeans.kmeans_re_sc_concatenation")
@patch("imblearn_resc.oversampling.KMeansReSC._resc_kmeans.get_set_n_kmeans_re_sc")
def test_kmeans_resc_fit_resample(mock_get_set_n, mock_concat, dummy_data):
    """Test the core logic flow without running the actual heavy math."""
    X, y = dummy_data
    
    mock_get_set_n.return_value = np.array([[9.0, 10.0]])
    mock_concat.return_value = (np.array([[1, 2, 1, 2]]), np.array([1]))

    sampler = KMeansReSC(M=1.5, num_candidates_to_test=3, random_state=42)
    X_res, y_res = sampler._fit_resample(X, y)

    mock_get_set_n.assert_called_once()
    mock_concat.assert_called_once()

    assert X_res.shape == (1, 4)
    assert y_res[0] == 1


def test_kmeans_resc_feature_names(dummy_data):
    """Test that it duplicates feature names properly for the 2d space."""
    X, y = dummy_data
    sampler = KMeansReSC()
    sampler.n_features_in_ = X.shape[1] 
    
    names = sampler.get_feature_names_out()
    expected_default = np.array(["x0_1", "x1_1", "x0_2", "x1_2"], dtype=object)
    np.testing.assert_array_equal(names, expected_default)
    
    names_custom = sampler.get_feature_names_out(["age", "income"])
    expected_custom = np.array(["age_1", "income_1", "age_2", "income_2"], dtype=object)
    np.testing.assert_array_equal(names_custom, expected_custom)


