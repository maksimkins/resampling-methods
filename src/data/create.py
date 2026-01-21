import numpy as np
import pandas as pd



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