import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import fetch_openml



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



def download_datasets():
    datasets = {}

    # --- 1. Abalone (UCI) ---
    # Min: 18 | Maj: 9
    datasets['abalone9-18'] = download_uci_datasets(
        name='abalone9-18', 
        uci_id=1, 
        min_c=[18], 
        maj_c=[9]
    )

    ## --- 2. Breast Tissue (OpenML) ---
    ## Min: car, fad | Maj: All other
    #datasets['breast'] = download_openml_datasets(
    #    name='breast', 
    #    data_id=1479, 
    #    min_c=['car', 'fad'], 
    #    maj_c="All other"
    #)

    # --- 3-10. Ecoli (UCI) ---
    datasets['ecoli-01vs235'] = download_uci_datasets(
        name='ecoli-01vs235', 
        uci_id=39, 
        min_c= ['imS', 'imL', 'om'], # in paper below
        maj_c=['cp', 'im'],          # in paper above
    )
    
    datasets['ecoli-01vs5'] = download_uci_datasets(
        name='ecoli-01vs5', 
        uci_id=39, 
        min_c=['om'],       # in paper below
        maj_c=['cp', 'im'], # in paper above
    )
    
    datasets['ecoli-0147vs56'] = download_uci_datasets(
        name='ecoli-0147vs56', 
        uci_id=39, 
        min_c=['om', 'omL'],                 # in paper below
        maj_c=['cp', 'im', 'imU', 'pp'],     # in paper above
    )
    
    datasets['ecoli-0234vs5'] = download_uci_datasets(
        name='ecoli-0234vs5', 
        uci_id=39, 
        min_c=['om'],                        # in paper below
        maj_c=['cp', 'imS', 'imL', 'imU'],   # in paper above
    )
    
    datasets['ecoli-046vs5'] = download_uci_datasets(
        name='ecoli-046vs5', 
        uci_id=39, 
        min_c=['om'],               # in paper below
        maj_c=['cp', 'imU', 'omL'], # in paper above
    )
    
    datasets['ecoli-067vs5'] = download_uci_datasets(
        name='ecoli-067vs5', 
        uci_id=39, 
        min_c=['om'],              # in paper below
        maj_c=['cp', 'omL', 'pp']  # in paper above
    )
    
    datasets['ecoli2'] = download_uci_datasets(
        name='ecoli2', 
        uci_id=39, 
        min_c=['pp'], 
        maj_c="All other"
    )
    
    datasets['ecoli3'] = download_uci_datasets(
        name='ecoli3', 
        uci_id=39, 
        min_c=['imU'], 
        maj_c="All other"
    )

    # --- 11-14. Glass (UCI) ---
    #datasets['glass0123vs456'] = download_uci_datasets(
    #    name='glass0123vs456', 
    #    uci_id=42, 
    #    min_c=[0, 1, 2, 3], 
    #    maj_c="All other"
    #)
    #
    #datasets['glass0'] = download_uci_datasets(
    #    name='glass0', 
    #    uci_id=42, 
    #    min_c=[0], 
    #    maj_c="All other"
    #)
    #
    #datasets['glass1'] = download_uci_datasets(
    #    name='glass1', 
    #    uci_id=42, 
    #    min_c=[1], 
    #    maj_c="All other"
    #)
    
    datasets['glass6'] = download_uci_datasets(
        name='glass6', 
        uci_id=42, 
        min_c=[6], 
        maj_c="All other"
    )

    # --- 15. Haberman (UCI) ---
    datasets['haberman'] = download_uci_datasets(
        name='haberman', 
        uci_id=43, 
        min_c=[2], 
        maj_c=[1]
    )

    # --- 16. Iris (UCI) ---
    datasets['iris0'] = download_uci_datasets(
        name='iris0', 
        uci_id=53, 
        min_c=['Iris-setosa'], 
        maj_c="All other"
    )

    ## --- 17. Leaf (OpenML) ---
    ## Min: 1 | Maj: All other (Using ID 1491)
    #datasets['leaf'] = download_openml_datasets(
    #    name='leaf', 
    #    data_id=1491, 
    #    min_c=[1,2,3,4,5,6,7], 
    #    maj_c="All other"
    #)
#
    ## --- 18-19. Thyroid (OpenML) ---
    ## Using ID 1515
    #datasets['new-thyroid1'] = download_openml_datasets(
    #    name='new-thyroid1', 
    #    data_id=1515, 
    #    min_c=[2], 
    #    maj_c="All other"
    #)
    #
    #datasets['new-thyroid2'] = download_openml_datasets(
    #    name='new-thyroid2', 
    #    data_id=1515, 
    #    min_c=[3], 
    #    maj_c="All other"
    #)

    # --- 20. Page Blocks (UCI) ---
    # Min: 3 | Maj: 2, 5 (Matches count 28 vs 444)
    datasets['page-blocks-13vs4'] = download_uci_datasets(
        name='page-blocks-13vs4', 
        uci_id=78, 
        min_c=[4], 
        maj_c=[1, 3]
    )

    ## --- 21. Parkinsons (UCI) ---
    #datasets['parkinsons'] = download_uci_datasets(
    #    name='parkinsons', 
    #    uci_id=174, 
    #    min_c=[0], 
    #    maj_c=[1]
    #)

    ## --- 22. Seeds (OpenML) ---
    ## Min: 1 | Maj: All other (Using ID 1499)
    #datasets['seeds'] = download_openml_datasets(
    #    name='seeds', 
    #    data_id=1499, 
    #    min_c=['Kama'], 
    #    maj_c=['Rosa', 'Canadian']
    #)
#
    ## --- 23. Shuttle (UCI) ---
    ## Min: 1 | Maj: 4
    #datasets['shuttle-0vs4'] = download_uci_datasets(
    #    name='shuttle-0vs4', 
    #    uci_id=148, 
    #    min_c=['positive'], 
    #    maj_c=['negative']
    #)

    # --- 24. SPECT (UCI) ---
    datasets['spect'] = download_uci_datasets(
        name='spect', 
        uci_id=95, 
        min_c=[0], # in paper below
        maj_c=[1]  # in paper above
    )

   ## --- 25. Vertebral (UCI) ---
   #datasets['vertebral'] = download_uci_datasets(
   #    name='vertebral', 
   #    uci_id=212, 
   #    min_c=['NO'], 
   #    maj_c=['AB']
   #)
#
   ## --- 26. WPBC (UCI) ---
   #datasets['wpbc'] = download_uci_datasets(
   #    name='wpbc', 
   #    uci_id=17, 
   #    min_c=['N'], 
   #    maj_c=['R']
   #)

    # --- 27-28. Yeast (UCI) ---
    datasets['yeast-1vs7'] = download_uci_datasets(
        name='yeast-1vs7', 
        uci_id=110, 
        min_c=['VAC'], 
        maj_c=['NUC']
    )
    
    datasets['yeast-2vs4'] = download_uci_datasets(
        name='yeast-2vs4', 
        uci_id=110, 
        min_c=['ME2'], # in paper below
        maj_c=['CYT']  # in paper above
    )

    return datasets