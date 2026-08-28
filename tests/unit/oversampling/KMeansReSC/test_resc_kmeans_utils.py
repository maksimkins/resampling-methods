from unittest.mock import patch
import warnings

import numpy as np
import pytest

from imblearn_resc.oversampling.KMeansReSC.utils import _resc_kmeans_utils as utils


class _CandidateKMeans:
    calls = []
    collapse = False
    emit_warning = False

    def __init__(self, n_clusters, random_state, n_init, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.kwargs = kwargs
        type(self).calls.append(self)

    def fit_predict(self, X):
        if self.emit_warning:
            warnings.warn("upstream KMeans warning", UserWarning)
        if self.collapse:
            return np.zeros(len(X), dtype=int)
        return np.arange(len(X), dtype=int) % self.n_clusters

    def fit(self, X):
        self.cluster_centers_ = np.asarray(X[: self.n_clusters], dtype=float)
        return self


@pytest.fixture(autouse=True)
def reset_fake_kmeans():
    _CandidateKMeans.calls = []
    _CandidateKMeans.collapse = False
    _CandidateKMeans.emit_warning = False


def test_single_feasible_candidate_is_scored():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with (
        patch.object(utils, "KMeans", _CandidateKMeans),
        patch.object(utils, "silhouette_score", return_value=0.5) as score,
    ):
        selected = utils.find_best_k_geometric(X, [2], 11, {"n_init": 3})

    assert selected == 2
    score.assert_called_once()


def test_equal_silhouette_scores_select_smallest_k():
    X = np.arange(12, dtype=float).reshape(6, 2)
    with (
        patch.object(utils, "KMeans", _CandidateKMeans),
        patch.object(utils, "silhouette_score", return_value=0.5),
    ):
        selected = utils.find_best_k_geometric(X, [1, 2, 3], 11)

    assert selected == 2


def test_collapsed_partitions_and_no_feasible_candidates_fall_back_to_one():
    X = np.arange(8, dtype=float).reshape(4, 2)
    _CandidateKMeans.collapse = True
    with patch.object(utils, "KMeans", _CandidateKMeans):
        assert utils.find_best_k_geometric(X, [2, 3], 11) == 1
        assert utils.find_best_k_geometric(X, [1], 11) == 1


def test_upstream_kmeans_warning_is_not_suppressed():
    X = np.arange(8, dtype=float).reshape(4, 2)
    _CandidateKMeans.emit_warning = True
    with (
        patch.object(utils, "KMeans", _CandidateKMeans),
        patch.object(utils, "silhouette_score", return_value=0.5),
        pytest.warns(UserWarning, match="upstream KMeans warning"),
    ):
        utils.find_best_k_geometric(X, [2], 11)


@pytest.mark.parametrize("reserved_key", ["n_clusters", "random_state"])
def test_kmeans_params_reject_reserved_keys(reserved_key):
    with pytest.raises(ValueError, match=reserved_key):
        utils.validate_kmeans_params({reserved_key: 4})


@pytest.mark.parametrize("reserved_key", ["n_neighbors", "metric", "weights"])
def test_knn_params_reject_reserved_keys(reserved_key):
    with pytest.raises(ValueError, match=reserved_key):
        utils.validate_knn_params({reserved_key: "conflict"})


def test_parameter_validation_does_not_mutate_caller_mappings():
    kmeans_params = {"n_init": 4, "max_iter": 20}
    knn_params = {"algorithm": "brute", "leaf_size": 10}
    expected_kmeans = kmeans_params.copy()
    expected_knn = knn_params.copy()

    utils.validate_kmeans_params(kmeans_params)
    utils.validate_knn_params(knn_params)

    assert kmeans_params == expected_kmeans
    assert knn_params == expected_knn


def test_complete_candidate_grid_and_same_seed_reach_all_kmeans_fits():
    X_min = np.arange(8, dtype=float).reshape(4, 2)
    X_maj = np.arange(16, dtype=float).reshape(8, 2) + 100
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * len(X_min) + [0] * len(X_maj))
    seen_candidates = []

    def record_and_choose(X_safe, candidates, random_state, kmeans_params=None):
        seen_candidates.extend(candidates)
        kwargs, n_init = utils._split_kmeans_params(kmeans_params)
        for candidate in candidates:
            if 2 <= candidate < len(X_safe):
                model = _CandidateKMeans(candidate, random_state, n_init, **kwargs)
                model.fit_predict(X_safe)
        return 2

    parameters = {"n_init": 3, "max_iter": 20}
    expected_parameters = parameters.copy()
    with (
        patch.object(
            utils,
            "get_safe_majority_samples_knn",
            return_value=(X_maj, False),
        ),
        patch.object(utils, "find_best_k_geometric", side_effect=record_and_choose),
        patch.object(utils, "KMeans", _CandidateKMeans),
    ):
        centers, fallback_used = utils.get_set_n_kmeans_re_sc(
            X=X,
            y=y,
            min_label=1,
            maj_label=0,
            M=1.5,
            random_state=17,
            n_neighbors=2,
            kmeans_params=parameters,
        )

    assert seen_candidates == [1, 2, 3]
    assert fallback_used is False
    assert centers.shape == (2, 2)
    assert {model.random_state for model in _CandidateKMeans.calls} == {17}
    assert {model.n_init for model in _CandidateKMeans.calls} == {3}
    assert parameters == expected_parameters


def test_n1_zero_still_produces_candidate_one():
    X_min = np.array([[0.0, 0.0]])
    X_maj = np.arange(8, dtype=float).reshape(4, 2) + 10
    X = np.vstack([X_min, X_maj])
    y = np.array([1, 0, 0, 0, 0])
    seen_candidates = []

    def select_one(X_safe, candidates, random_state, kmeans_params=None):
        seen_candidates.extend(candidates)
        return 1

    with (
        patch.object(
            utils,
            "get_safe_majority_samples_knn",
            return_value=(X_maj, False),
        ),
        patch.object(utils, "find_best_k_geometric", side_effect=select_one),
        patch.object(utils, "KMeans", _CandidateKMeans),
    ):
        utils.get_set_n_kmeans_re_sc(
            X,
            y,
            min_label=1,
            maj_label=0,
            M=1.5,
            random_state=3,
            n_neighbors=1,
        )

    assert seen_candidates == [1]


def test_safety_filter_enforces_euclidean_uniform_vote_and_self_inclusion():
    X = np.array([[0.0], [0.0], [2.0]])
    y = np.array([0, 1, 0])
    X_maj = X[y == 0]
    received_kwargs = {}

    class FakeKNN:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

        def fit(self, X_fit, y_fit):
            return self

        def kneighbors(self, X_query, return_distance=False):
            return np.array([[1], [1]])

    with patch.object(utils, "KNeighborsClassifier", FakeKNN):
        safe, fallback_used = utils.get_safe_majority_samples_knn(
            X,
            y,
            X_maj,
            maj_label=0,
            n_neighbors=1,
            threshold=1.0,
            knn_params={"algorithm": "brute"},
        )

    np.testing.assert_array_equal(safe, X_maj)
    assert fallback_used is False
    assert received_kwargs["metric"] == "euclidean"
    assert received_kwargs["weights"] == "uniform"
    assert received_kwargs["n_neighbors"] == 1


def test_empty_safety_filter_restores_complete_majority_collection():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    X_maj = X[y == 0]

    class FakeKNN:
        def __init__(self, **kwargs):
            pass

        def fit(self, X_fit, y_fit):
            return self

        def kneighbors(self, X_query, return_distance=False):
            return np.array([[0, 1], [2, 3]])

    with patch.object(utils, "KNeighborsClassifier", FakeKNN):
        safe, fallback_used = utils.get_safe_majority_samples_knn(
            X,
            y,
            X_maj,
            maj_label=0,
            n_neighbors=2,
            threshold=1.0,
        )

    np.testing.assert_array_equal(safe, X_maj)
    assert fallback_used is True


def test_concatenation_preserves_all_ordered_pairs_and_class_counts():
    X_min = np.array([[1.0], [2.0]])
    X_maj = np.array([[10.0], [20.0], [30.0]])
    centers = np.array([[100.0], [200.0]])

    X_resampled, y_resampled = utils.kmeans_re_sc_concatenation(
        X_min,
        X_maj,
        centers,
        min_label=1,
        maj_label=0,
    )

    expected_minority = np.array(
        [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]
    )
    expected_majority = np.array(
        [
            [10.0, 100.0],
            [10.0, 200.0],
            [20.0, 100.0],
            [20.0, 200.0],
            [30.0, 100.0],
            [30.0, 200.0],
        ]
    )
    np.testing.assert_array_equal(X_resampled[:4], expected_minority)
    np.testing.assert_array_equal(X_resampled[4:], expected_majority)
    assert np.count_nonzero(y_resampled == 1) == len(X_min) ** 2
    assert np.count_nonzero(y_resampled == 0) == len(X_maj) * len(centers)
