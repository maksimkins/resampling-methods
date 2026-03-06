import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def drop_categorical_columns(X):
    X_numeric = X.select_dtypes(include=['number'])    
    return X_numeric



def standardize_all(df):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_out[num_cols] = scaler.fit_transform(df_out[num_cols])
    return df_out

    

def standardize_column(df, col_name):
    df_out = df.copy()
    scaler = StandardScaler()
    df_out[col_name] = scaler.fit_transform(df_out[[col_name]])
    return df_out



def minmax_all(df):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        df_out[num_cols] = scaler.fit_transform(df_out[num_cols])
    return df_out



def minmax_column(df, col_name):
    df_out = df.copy()
    scaler = MinMaxScaler()
    df_out[col_name] = scaler.fit_transform(df_out[[col_name]])
    return df_out



def center_all(df):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df_out[num_cols] = df_out[num_cols] - df_out[num_cols].mean()
    return df_out



def center_column(df, col_name):
    df_out = df.copy()
    df_out[col_name] = df_out[col_name] - df_out[col_name].mean()
    return df_out



def log_transform_all(df):
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include="number").columns
    for col in num_cols:
        min_val = df_out[col].min()
        if min_val < 0: 
            col_log = np.log1p(df_out[col] - min_val)
        else: 
            col_log = np.log1p(df_out[col])
        df_out[col] = col_log
    return df_out



def log_transform_column(df, col_name):
    df_out = df.copy()
    min_val = df_out[col_name].min()
        if min_val < 0: 
            col_log = np.log1p(df_out[col_name] - min_val)
        else: 
            col_log = np.log1p(df_out[col_name])
    df_out[col_name] = col_log
    return df_out