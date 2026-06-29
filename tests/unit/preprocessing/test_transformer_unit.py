import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

from imblearn_resc.preprocessing import ReSCTransformer


def test_fit_calculates_dimensions():
    """Test that fit correctly deduces the 2d and d feature counts."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0]])
    
    transformer = ReSCTransformer()
    transformer.fit(X_train)

    assert transformer.n_features_in_ == 4
    assert transformer.original_d_ == 2
    assert transformer.is_fitted_ is True


def test_transform_duplicates_test_data():
    """Test Scenario 1: Pipeline passes raw test data (d features) and it gets duplicated."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0]])
    transformer = ReSCTransformer()
    transformer.fit(X_train)
    
    X_test = np.array([[10.0, 20.0], 
                       [30.0, 40.0]])
    X_transformed = transformer.transform(X_test)
    expected = np.array([[10.0, 20.0, 10.0, 20.0], 
                         [30.0, 40.0, 30.0, 40.0]])
    
    assert X_transformed.shape == (2, 4)
    np.testing.assert_array_equal(X_transformed, expected)


def test_transform_passes_through_train_data():
    """Test Scenario 2: Pipeline passes already-resampled training data (2d features)."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0]])
    
    transformer = ReSCTransformer()
    transformer.fit(X_train)
    X_transformed = transformer.transform(X_train)
    
    assert X_transformed.shape == (2, 4)
    np.testing.assert_array_equal(X_transformed, X_train)


def test_transform_invalid_dimensions():
    """Test Scenario 3: Passing data with completely wrong dimensions raises an error."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0]])
    transformer = ReSCTransformer()
    transformer.fit(X_train)
    
    X_invalid = np.array([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValueError):
        transformer.transform(X_invalid)


def test_not_fitted_error():
    """Ensure transform and feature name methods fail if fit hasn't been called."""
    transformer = ReSCTransformer()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(NotFittedError):
        transformer.transform(X)

    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


def test_get_feature_names_out_default():
    """Test feature name generation when no input names are provided."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0]])
    transformer = ReSCTransformer()
    transformer.fit(X_train) 

    names = transformer.get_feature_names_out()
    expected = np.array(["x0_1", "x1_1", "x0_2", "x1_2"], dtype=object)
    
    np.testing.assert_array_equal(names, expected)


def test_get_feature_names_out_custom():
    """Test feature name generation with custom input names."""
    X_train = np.array([[1.0, 2.0, 3.0, 4.0]])
    transformer = ReSCTransformer()
    transformer.fit(X_train)

    custom_features = ["age", "income"]
    names = transformer.get_feature_names_out(custom_features)
    expected = np.array(["age_1", "income_1", "age_2", "income_2"], dtype=object)
    
    np.testing.assert_array_equal(names, expected)