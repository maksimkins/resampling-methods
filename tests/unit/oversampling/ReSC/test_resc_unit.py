import pytest
import numpy as np
from unittest.mock import patch

from imblearn_resc.oversampling import ReSC

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

def test_resc_init():
    """Test that parameters are assigned correctly upon initialization."""
    sampler = ReSC(M=2.0, k=3, alpha=0.01, epsilon=0.1, random_state=42)
    assert sampler.M == 2.0
    assert sampler.k == 3
    assert sampler.alpha == 0.01
    assert sampler.epsilon == 0.1
    assert sampler.random_state == 42


def test_resc_single_class_error():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1, 1])

    sampler = ReSC()
    with pytest.raises(ValueError, match="needs to have at least two classes"):
        sampler._fit_resample(X, y)

@patch("imblearn_resc.oversampling.ReSC._resc.re_sc_concatenation")
@patch("imblearn_resc.oversampling.ReSC._resc.get_set_n_random_weighted_re_sc")
@patch("imblearn_resc.oversampling.ReSC._resc.calculate_set_n_size_re_sc")
def test_resc_fit_resample(mock_calc_size, mock_get_set_n, mock_concat, dummy_data):
    """Test the core logic flow without running the actual heavy math."""
    X, y = dummy_data
    
    mock_calc_size.return_value = 2
    mock_get_set_n.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_concat.return_value = (np.array([[1, 2, 1, 2]]), np.array([1]))

    sampler = ReSC(M=1.5, k=5, alpha=0.05, epsilon=0.05, random_state=0)
    X_res, y_res = sampler._fit_resample(X, y)

    mock_calc_size.assert_called_once()
    mock_get_set_n.assert_called_once()
    mock_concat.assert_called_once()

    kwargs = mock_calc_size.call_args.kwargs
    assert kwargs['P'] == 1  
    assert kwargs['M'] == 1.5

    assert X_res.shape == (1, 4)
    assert y_res[0] == 1


def test_resc_feature_names():
    """Test feature names for standard ReSC."""
    sampler = ReSC()
    sampler.n_features_in_ = 3 
    
    names = sampler.get_feature_names_out(["A", "B", "C"])
    expected = np.array(["A_1", "B_1", "C_1", "A_2", "B_2", "C_2"], dtype=object)
    np.testing.assert_array_equal(names, expected)