import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



def drop_categorical_columns(X):
    X_numeric = X.select_dtypes(include=['number'])
    
    dropped_count = X.shape[1] - X_numeric.shape[1]
    if dropped_count > 0:
        print(f"-> Dropped {dropped_count} categorical/object columns.")
    
    return X_numeric



def preprocess_and_normalize(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.columns.difference(num_cols)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_num = imputer.fit_transform(X[num_cols])
    X_num = scaler.fit_transform(X_num)

    X_num_df = pd.DataFrame(X_num, columns=num_cols, index=X.index)

    X_cat_df = X[cat_cols].copy()

    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

    X_processed = X_processed[X.columns]

    return X_processed