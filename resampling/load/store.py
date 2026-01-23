import os
import glob

from tqdm import tqdm

import numpy as np
import pandas as pd



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