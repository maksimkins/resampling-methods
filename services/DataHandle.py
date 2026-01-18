import os
import glob

from tqdm import tqdm

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import fetch_openml

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler





def download_uci_datasets(name, uci_id, min_c, maj_c):
    print(f"Fetching {name} (UCI ID: {uci_id})...")
    try:
        dataset = fetch_ucirepo(id=uci_id)
        X = dataset.data.features
        y = dataset.data.targets
        
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        if maj_c == "All other":
            mask = pd.Series([True] * len(y), index=y.index)
        else:
            all_classes = (min_c + maj_c)
            mask = y.astype(str).isin([str(c) for c in all_classes])

        X_sub = X[mask].copy()
        y_sub = y[mask].copy()
        min_vals = [str(c) for c in (min_c if isinstance(min_c, list) else [min_c])]
        binary_target = np.where(y_sub.astype(str).isin(min_vals), 1, -1)
        
        return (X_sub, pd.Series(binary_target, name='target'))
    except Exception as e:
        print(f"Error fetching {name}: {e}")



def download_openml_datasets(name, data_id, min_c, maj_c):
    print(f"Fetching {name} (OpenML ID: {data_id})...")
    try:
        X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)

        if maj_c == "All other":
            mask = pd.Series([True] * len(y), index=y.index)
        else:
            all_classes = (min_c + maj_c)
            mask = y.astype(str).isin([str(c) for c in all_classes])
        X_sub = X[mask].copy()
        y_sub = y[mask].copy()
        min_vals = [str(c) for c in (min_c if isinstance(min_c, list) else [min_c])]
        binary_target = np.where(y_sub.astype(str).isin(min_vals), 1, -1)

        
        return (X_sub, pd.Series(binary_target, name='target'))
    except Exception as e:
        print(f"Error fetching {name}: {e}")



def create_synthetic_overlap(n_samples=1000, n_features=1, weights=(0.9, 0.1), density_type='high'):
    rng = np.random.RandomState(42)
    n_min = int(n_samples * weights[1])
    n_maj = n_samples - n_min
    
    # Generate Minority Class (centered at 0)
    X_min = rng.normal(loc=0.0, scale=1.0, size=(n_min, n_features))
    y_min = np.ones(n_min)
    
    # Generate Majority Class based on density type
    if density_type == 'high':
        # Type 1: High density in overlap
        X_maj = rng.normal(loc=1.0, scale=1.0, size=(n_maj, n_features))
        
    elif density_type == 'sparse':
        # Type 2: Overlap exists but majority is sparse there
        n_maj_overlap = int(n_maj * 0.1) # Only 10% in overlap
        n_maj_far = n_maj - n_maj_overlap
        
        X_maj_overlap = rng.normal(loc=0.0, scale=1.0, size=(n_maj_overlap, n_features))
        X_maj_far = rng.normal(loc=5.0, scale=1.0, size=(n_maj_far, n_features))
        X_maj = np.vstack([X_maj_overlap, X_maj_far])
        
    y_maj = -1 * np.ones(n_maj)
    
    # Stack the arrays
    X_array = np.vstack([X_min, X_maj])
    y_array = np.hstack([y_min, y_maj])
    
    # Create feature names
    feat_names = [f"feat_{i}" for i in range(n_features)]

    X_df = pd.DataFrame(X_array, columns=feat_names)
    
    return (X_df, pd.Series(y_array, name='target'))



def save_dataset_csv(X, y, name, folder_path="../data"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    df_combined = X.copy()
    target_col_name = 'target' 
    df_combined[target_col_name] = y.values
 
    file_path = os.path.join(folder_path, f"{name}.csv")
    df_combined.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")



def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    target_col_name = 'target'
    
    if target_col_name not in df.columns:
        raise ValueError(f"Column '{target_col_name}' not found in {file_path}")
    
    y = df[target_col_name]
    X = df.drop(columns=[target_col_name])
    y = y.astype(int)
    
    return X, y



def load_datasets_csv(folder_path="../data"):
    datasets_dict = {}
    
    search_path = os.path.join(folder_path, "*.csv")
    files = glob.glob(search_path)
    
    print(f"--- Loading datasets from '{folder_path}/' ---")
    for file_path in tqdm(files):
        filename = os.path.basename(file_path)
        name = os.path.splitext(filename)[0]
        
        try:
            X, y = load_dataset_csv(file_path)
            datasets_dict[name] = (X, y)
            print(f"Loaded: {name} | Shape: {X.shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    return datasets_dict



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