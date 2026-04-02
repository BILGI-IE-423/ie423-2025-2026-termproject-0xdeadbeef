import pandas as pd
import os

# Define file paths using relative paths for reproducibility
DATA_PATH = os.path.join("data", "raw", "Sarcasm_Headlines_Dataset.json")

def load_data():
    print("="*60)
    print(" MODULE 01: RAW DATA LOADING & INITIAL INSPECTION ")
    print("="*60)
    
    # 1. File Path Verification
    print("\n[INFO] Step 1: Verifying file path...")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Critical failure! File not found at: {DATA_PATH}")
        print("Please ensure the JSON dataset is correctly placed in 'data/raw/' folder.")
        return None
    
    try:
        # 2. Loading the Dataset
        print(f"[INFO] Loading data from '{DATA_PATH}'...")
        df = pd.read_json(DATA_PATH, lines=True)
        print("[SUCCESS] Data loaded successfully into Pandas DataFrame.")
        
        # 3. Structural Information (Shape & Types)
        print("\n--- Step 2: Structural Data Inspection ---")
        print(f"Total Rows (Observations): {df.shape[0]}")
        print(f"Total Columns (Variables): {df.shape[1]}")
        print(f"Columns List: {list(df.columns)}")
        
        print("\n--- Step 3: Data Types & Exact Memory Usage ---")
        # Using memory_usage='deep' for accurate memory footprint of object types
        df.info(memory_usage='deep')
        
        # 4. Data Quality Checks
        print("\n--- Step 4: Data Quality Checks ---")
        print("Missing Values (NaN/Null) per column:")
        print(df.isnull().sum())
        
        duplicate_count = df.duplicated().sum()
        print(f"\nExact Duplicate Rows found in raw data: {duplicate_count}")
        if duplicate_count > 0:
            print("  -> Note: These duplicates will be handled in the preprocessing pipeline (02).")
        
        # 5. Target Variable Distribution Preview
        print("\n--- Step 5: Target Variable (Class) Distribution ---")
        if 'is_sarcastic' in df.columns:
            class_counts = df['is_sarcastic'].value_counts()
            class_props = df['is_sarcastic'].value_counts(normalize=True) * 100
            for cls, count in class_counts.items():
                label = "Sarcastic" if cls == 1 else "Normal"
                prop = class_props[cls]  # İŞTE EKSİK OLAN SATIR BURASIYDI!
                print(f"Class {cls} ({label}): {count} samples ({prop:.2f}%)")
        
        # 6. Previewing the actual data
        print("\n--- Step 6: Raw Data Preview (First 5 Rows) ---")
        # Setting pandas option to show full text without truncation
        pd.set_option('display.max_colwidth', None)
        print(df.head())
        
        print("\n" + "="*60)
        print(" MODULE 01 EXECUTION COMPLETED SUCCESSFULLY ")
        print("="*60 + "\n")
        
        return df

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while processing the data: {e}")

if __name__ == "__main__":
    load_data()