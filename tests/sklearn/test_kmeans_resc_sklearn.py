from sklearn.utils.estimator_checks import check_estimator
from resampling.methods import KMeansReSC

def test_resc_complies_sklearn_api():
    check_estimator(KMeansReSC())