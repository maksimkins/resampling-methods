from sklearn.utils.estimator_checks import check_estimator
from resampling.preprocessing import ReSCTransformer

print("Testing ReSCTransformer...")
check_estimator(ReSCTransformer())