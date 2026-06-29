from sklearn.utils.estimator_checks import check_estimator
from imblearn_resc.oversampling import ReSC

def test_resc_complies_sklearn_api():
    check_estimator(ReSC())