from sklearn.utils.estimator_checks import check_estimator
from resampling.methods import ReSC

def test_resc_complies_sklearn_api():
    check_estimator(ReSC())