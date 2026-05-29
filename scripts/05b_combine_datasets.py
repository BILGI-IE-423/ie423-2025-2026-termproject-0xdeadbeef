import pandas as pd
import os
 
# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
NEWS_PATH    = os.path.join("data", "processed", "cleaned_sarcasm_data.csv")
GEN_PATH     = os.path.join("data", "processed", "cleaned_gen_data.csv")
OUTPUT_PATH  = os.path.join("data", "processed", "combined_data.csv")
 
RANDOM_STATE = 42
 
def combine_datasets():
    print("="*60)
    print(" MODULE 05b: COMBINING NEWS + GEN DATASETS ")
    print("="*60)
 
    # ----------------------------------------------------------------
    # STEP 1: Load both processed datasets
    # ----------------------------------------------------------------
    print("\n--- Step 1: Loading Processed Datasets ---")
 
    if not os.path.exists(NEWS_PATH):
        print(f"[ERROR] News dataset not found: {NEWS_PATH}")
        print("Run 02_preprocess_data.py first.")
        return
 
    if not os.path.exists(GEN_PATH):
        print(f"[ERROR] GEN dataset not found: {GEN_PATH}")
        print("Run 05_preprocess_gen.py first.")
        return
 
    news_df = pd.read_csv(NEWS_PATH)
    gen_df  = pd.read_csv(GEN_PATH)
 
    print(f"News dataset : {news_df.shape[0]} rows x {news_df.shape[1]} columns")
    print(f"GEN dataset  : {gen_df.shape[0]} rows x {gen_df.shape[1]} columns")
 
    # ----------------------------------------------------------------
    # STEP 2: Verify column compatibility
    # Both datasets must have identical columns since 05_preprocess_gen.py
    # aligns GEN column names with the news dataset format.
    # ----------------------------------------------------------------
    print("\n--- Step 2: Verifying Column Compatibility ---")
 
    news_cols = set(news_df.columns)
    gen_cols  = set(gen_df.columns)
 
    if news_cols != gen_cols:
        print("[ERROR] Column mismatch between datasets!")
        print(f"  News columns : {sorted(news_cols)}")
        print(f"  GEN columns  : {sorted(gen_cols)}")
        print("  Check 05_preprocess_gen.py column alignment.")
        return
 
    print(f"[OK] Both datasets have identical columns: {sorted(news_cols)}")
 
    # ----------------------------------------------------------------
    # STEP 3: Add source column for traceability
    # This allows downstream analysis to distinguish the origin of each
    # row when needed (e.g. leakage investigation).
    # ----------------------------------------------------------------
    print("\n--- Step 3: Adding Source Column ---")
    news_df['source'] = 'news'
    gen_df['source']  = 'gen'
    print("Added 'source' column: 'news' or 'gen'")
 
    # ----------------------------------------------------------------
    # STEP 4: Combine and shuffle
    # ----------------------------------------------------------------
    print("\n--- Step 4: Combining and Shuffling ---")
    combined_df = pd.concat(
        [news_df, gen_df], ignore_index=True
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
 
    print(f"Combined dataset: {combined_df.shape[0]} rows x "
          f"{combined_df.shape[1]} columns")
 
    # ----------------------------------------------------------------
    # STEP 5: Class distribution check
    # ----------------------------------------------------------------
    print("\n--- Step 5: Class Distribution ---")
    class_counts = combined_df['is_sarcastic'].value_counts()
    for cls, count in class_counts.items():
        label = "Sarcastic" if cls == 1 else "Normal"
        pct   = count / len(combined_df) * 100
        print(f"  Class {cls} ({label}): {count} rows ({pct:.1f}%)")
 
    print("\n--- Step 5b: Source Distribution ---")
    source_counts = combined_df['source'].value_counts()
    for src, count in source_counts.items():
        pct = count / len(combined_df) * 100
        print(f"  {src}: {count} rows ({pct:.1f}%)")
 
    # ----------------------------------------------------------------
    # STEP 6: Save combined dataset
    # ----------------------------------------------------------------
    print("\n--- Step 6: Saving Combined Dataset ---")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Final shape: {combined_df.shape[0]} rows x "
          f"{combined_df.shape[1]} columns")
    print(f"Columns: {list(combined_df.columns)}")
 
    print("\n" + "="*60)
    print(" MODULE 05b EXECUTION COMPLETED SUCCESSFULLY ")
    print("="*60)
    print(f"\nOutput: data/processed/combined_data.csv")
    print("Next step: python scripts/06_modeling.py")
 
if __name__ == "__main__":
    combine_datasets()