from sklearn.utils.estimator_checks import check_estimator
from imblearn_resc.oversampling import KMeansReSC

def test_resc_complies_sklearn_api():
    check_estimator(KMeansReSC())