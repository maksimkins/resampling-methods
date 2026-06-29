from sklearn.utils.estimator_checks import check_estimator
from imblearn_resc.preprocessing import ReSCTransformer

print("Testing ReSCTransformer...")
check_estimator(ReSCTransformer())