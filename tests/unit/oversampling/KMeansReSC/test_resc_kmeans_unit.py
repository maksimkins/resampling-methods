import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from imblearn_resc.oversampling import KMeansReSC
from imblearn_resc.preprocessing import ReSCTransformer


@pytest.fixture
def dummy_data():
    """Provide a small strictly imbalanced binary dataset."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [3.0, 4.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1])
    return X, y


def test_kmeans_resc_init():
    sampler = KMeansReSC(M=2.0, random_state=42)
    assert sampler.M == 2.0
    assert sampler.random_state == 42


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.array([0, 0, 0]), "exactly two"),
        (np.array([0, 1, 2]), "exactly two"),
        (np.array([0, 0, 1, 1]), "strictly unequal"),
    ],
)
def test_kmeans_resc_rejects_targets_outside_paper_domain(y, message):
    X = np.arange(len(y) * 2, dtype=float).reshape(len(y), 2)
    sampler = KMeansReSC(n_neighbors=1)
    with pytest.raises(ValueError, match=message):
        sampler.fit_resample(X, y)


def test_kmeans_resc_rejects_too_many_neighbors(dummy_data):
    X, y = dummy_data
    sampler = KMeansReSC(n_neighbors=len(X) + 1)
    with pytest.raises(ValueError, match="must not exceed"):
        sampler.fit_resample(X, y)


def test_kmeans_resc_rejects_nonfinite_M(dummy_data):
    X, y = dummy_data
    sampler = KMeansReSC(M=np.inf, n_neighbors=1)
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y)


@patch("imblearn_resc.oversampling.KMeansReSC._resc_kmeans.kmeans_re_sc_concatenation")
@patch("imblearn_resc.oversampling.KMeansReSC._resc_kmeans.get_set_n_kmeans_re_sc")
def test_kmeans_resc_fit_resample_preserves_two_value_return(
    mock_get_set_n,
    mock_concat,
    dummy_data,
):
    X, y = dummy_data
    mock_get_set_n.return_value = (np.array([[9.0, 10.0]]), True)
    mock_concat.return_value = (np.array([[1.0, 2.0, 1.0, 2.0]]), np.array([1]))

    sampler = KMeansReSC(M=1.5, random_state=42)
    result = sampler.fit_resample(X, y)

    assert len(result) == 2
    assert result[0].shape == (1, 4)
    assert result[1][0] == 1
    assert sampler.fallback_used_ is True
    mock_get_set_n.assert_called_once()
    mock_concat.assert_called_once()


def test_kmeans_resc_feature_names(dummy_data):
    X, _ = dummy_data
    sampler = KMeansReSC()
    sampler.n_features_in_ = X.shape[1]

    names = sampler.get_feature_names_out()
    expected_default = np.array(["x0_1", "x1_1", "x0_2", "x1_2"], dtype=object)
    np.testing.assert_array_equal(names, expected_default)

    names_custom = sampler.get_feature_names_out(["age", "income"])
    expected_custom = np.array(
        ["age_1", "income_1", "age_2", "income_2"],
        dtype=object,
    )
    np.testing.assert_array_equal(names_custom, expected_custom)


def test_ndarray_input_returns_ndarrays(dummy_data):
    X, y = dummy_data
    sampler = KMeansReSC(
        M=1.0,
        n_neighbors=1,
        safe_threshold=0.0,
        random_state=7,
        kmeans_params={"n_init": 2},
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    assert isinstance(X_resampled, np.ndarray)
    assert isinstance(y_resampled, np.ndarray)
    assert X_resampled.shape[1] == 2 * X.shape[1]


def test_dataframe_input_returns_generated_dataframe_and_aligned_series(dummy_data):
    pd = pytest.importorskip("pandas")
    X, y = dummy_data
    X_df = pd.DataFrame(X, columns=["age", "income"], index=np.arange(10, 16))
    y_series = pd.Series(y, name="target", index=X_df.index)
    sampler = KMeansReSC(
        M=1.0,
        n_neighbors=1,
        safe_threshold=0.0,
        random_state=7,
        kmeans_params={"n_init": 2},
    )

    X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)

    assert isinstance(X_resampled, pd.DataFrame)
    assert isinstance(y_resampled, pd.Series)
    assert X_resampled.columns.tolist() == [
        "age_1",
        "income_1",
        "age_2",
        "income_2",
    ]
    assert X_resampled.index.equals(pd.RangeIndex(len(X_resampled)))
    assert y_resampled.index.equals(X_resampled.index)
    assert y_resampled.name == "target"
    assert list(sampler.feature_names_in_) == ["age", "income"]
    assert sampler.get_feature_names_out().tolist() == X_resampled.columns.tolist()


def test_fixed_integer_configuration_is_repeatable_up_to_row_order(dummy_data):
    X, y = dummy_data
    parameters = dict(
        M=2.0,
        n_neighbors=1,
        safe_threshold=0.0,
        random_state=19,
        kmeans_params={"n_init": 3},
    )

    first_X, first_y = KMeansReSC(**parameters).fit_resample(X, y)
    second_X, second_y = KMeansReSC(**parameters).fit_resample(X, y)

    first_rows = sorted(map(tuple, np.column_stack([first_X, first_y])))
    second_rows = sorted(map(tuple, np.column_stack([second_X, second_y])))
    np.testing.assert_allclose(first_rows, second_rows)


def test_sampler_transformer_pipeline_handles_training_and_prediction(dummy_data):
    X, y = dummy_data
    pipeline = Pipeline(
        [
            (
                "sampler",
                KMeansReSC(
                    M=1.0,
                    n_neighbors=1,
                    safe_threshold=0.0,
                    random_state=7,
                    kmeans_params={"n_init": 2},
                ),
            ),
            ("transformer", ReSCTransformer()),
            ("classifier", LogisticRegression()),
        ]
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == y.shape


@pytest.mark.parametrize("thread_value", [None, "7"])
def test_fit_resample_does_not_mutate_omp_num_threads(thread_value):
    project_root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    if thread_value is None:
        environment.pop("OMP_NUM_THREADS", None)
        expected = "<missing>"
    else:
        environment["OMP_NUM_THREADS"] = thread_value
        expected = thread_value

    script = """
import os
import numpy as np
from imblearn_resc.oversampling import KMeansReSC

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [3., 3.], [3., 4.]])
y = np.array([0, 0, 0, 0, 1, 1])
KMeansReSC(
    M=1.0,
    n_neighbors=1,
    safe_threshold=0.0,
    random_state=7,
    kmeans_params={"n_init": 1},
).fit_resample(X, y)
print(os.environ.get("OMP_NUM_THREADS", "<missing>"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
