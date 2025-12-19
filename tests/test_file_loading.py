
import pandas as pd
import numpy as np
import os

def create_sample_files():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.5, 20.0, 30.5, 40.0, 50.5],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux']
    })
    
    os.makedirs('temp_test_data', exist_ok=True)
    
    # CSV
    df.to_csv('temp_test_data/test.csv', index=False)
    # TSV
    df.to_csv('temp_test_data/test.tsv', sep='\t', index=False)
    # JSON
    df.to_json('temp_test_data/test.json')
    # Excel
    df.to_excel('temp_test_data/test.xlsx', index=False)
    # Parquet
    df.to_parquet('temp_test_data/test.parquet')
    # Feather
    df.to_feather('temp_test_data/test.feather')
    # HDF5
    try:
        df.to_hdf('temp_test_data/test.h5', key='df', mode='w')
    except Exception as e:
        print(f"HDF5 error (likely library issue): {e}")
    # Stata
    df.to_stata('temp_test_data/test.dta', write_index=False)
    
    print("Sample files created in temp_test_data/")

def verify_loading():
    print("Verifying loading logic...")
    
    files = [
        'test.csv', 'test.tsv', 'test.json', 'test.xlsx', 
        'test.parquet', 'test.feather', 'test.h5', 'test.dta'
    ]
    
    results = {}
    
    for f in files:
        path = os.path.join('temp_test_data', f)
        if not os.path.exists(path):
            print(f"Skipping {f} (not found)")
            continue
            
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext == '.tsv':
                df = pd.read_csv(path, sep='\t')
            elif ext == '.json':
                df = pd.read_json(path)
            elif ext == '.xlsx':
                df = pd.read_excel(path)
            elif ext == '.parquet':
                df = pd.read_parquet(path)
            elif ext == '.feather':
                df = pd.read_feather(path)
            elif ext == '.h5':
                df = pd.read_hdf(path)
            elif ext == '.dta':
                df = pd.read_stata(path)
            else:
                continue
                
            print(f"Loaded {f}: OK (shape={df.shape})")
            results[f] = "Success"
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            results[f] = f"Error: {e}"
            
    return results

if __name__ == "__main__":
    create_sample_files()
    verify_loading()
