import pandas as pd
import os
import re
 
# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
INPUT_PATH  = os.path.join("data", "raw", "GEN-sarc-notsarc.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cleaned_gen_data.csv")
 
# ----------------------------------------------------------------
# Contraction map — mirrors 02_preprocess_data.py
# ----------------------------------------------------------------
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he's": "he is", "how's": "how is",
    "I'm": "I am", "I've": "I have", "I'll": "I will", "I'd": "I would",
    "isn't": "is not", "it's": "it is", "let's": "let us",
    "she's": "she is", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would"
}
 
def expand_contractions(text):
    for word, expanded in CONTRACTION_MAP.items():
        text = re.sub(re.escape(word), expanded, text, flags=re.IGNORECASE)
    return text
 
def clean_text(text):
    """
    Mirrors the clean_text logic in 02_preprocess_data.py so that
    both datasets go through an identical text cleaning pipeline.
    """
    text = str(text)
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
def run_gen_preprocessing():
    print("="*60)
    print(" MODULE 05: GEN DATASET PREPROCESSING ")
    print("="*60)
 
    # ----------------------------------------------------------------
    # STEP 1: Load raw GEN dataset
    # ----------------------------------------------------------------
    print("\n--- Step 1: Loading Raw GEN Dataset ---")
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] File not found: {INPUT_PATH}")
        print("Place GEN-sarc-notsarc.csv in data/raw/")
        return
 
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
 
    # ----------------------------------------------------------------
    # STEP 2: Label conversion
    # Convert 'sarc'/'notsarc' to 1/0 to match the news dataset format.
    # ----------------------------------------------------------------
    print("\n--- Step 2: Label Conversion ---")
    df['is_sarcastic'] = df['class'].map({'sarc': 1, 'notsarc': 0})
    df = df.dropna(subset=['is_sarcastic'])
    df['is_sarcastic'] = df['is_sarcastic'].astype(int)
 
    class_counts = df['is_sarcastic'].value_counts()
    for cls, count in class_counts.items():
        label = "Sarcastic" if cls == 1 else "Normal"
        pct = count / len(df) * 100
        print(f"  Class {cls} ({label}): {count} samples ({pct:.1f}%)")
 
    # ----------------------------------------------------------------
    # STEP 3: Feature engineering (from raw text — before cleaning)
    # Same 6 features as 02_preprocess_data.py, extracted from the
    # original text to preserve punctuation signals.
    # Note: word_count and char_count will differ from the news dataset
    # because forum comments are longer. StandardScaler in 06_modeling.py
    # will normalize these differences before model training.
    # ----------------------------------------------------------------
    print("\n--- Step 3: Feature Engineering (from raw text) ---")
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['avg_word_length'] = df['text'].apply(
        lambda x: sum(len(w) for w in str(x).split()) / len(str(x).split())
        if len(str(x).split()) > 0 else 0)
    df['quote_count'] = df['text'].apply(
        lambda x: str(x).count('"') + str(x).count("'"))
    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
    df['question_mark_count'] = df['text'].apply(lambda x: str(x).count('?'))
 
    print("Feature engineering complete.")
    print(f"\nFeature means by class:")
    feature_cols = ['char_count', 'word_count', 'avg_word_length',
                    'quote_count', 'exclamation_count', 'question_mark_count']
    summary = df.groupby('is_sarcastic')[feature_cols].mean().round(2)
    summary.index = summary.index.map({0: 'Normal', 1: 'Sarcastic'})
    print(summary.to_string())
 
    # ----------------------------------------------------------------
    # STEP 4: Text cleaning (same pipeline as 02_preprocess_data.py)
    # ----------------------------------------------------------------
    print("\n--- Step 4: Text Cleaning ---")
    df['cleaned_text'] = df['text'].apply(clean_text)
 
    # ----------------------------------------------------------------
    # STEP 5: Integrity filtering (same rules as 02_preprocess_data.py)
    # ----------------------------------------------------------------
    print("\n--- Step 5: Integrity Filtering ---")
    initial_count = len(df)
    df = df[df['cleaned_text'].str.strip() != '']
    df = df[df['cleaned_text'].apply(lambda x: len(str(x).split()) > 2)]
    dropped = initial_count - len(df)
    print(f"Rows dropped by integrity filters: {dropped}")
    print(f"Remaining rows: {len(df)}")
 
    # ----------------------------------------------------------------
    # STEP 6: Select and rename columns to match news dataset format
    # ----------------------------------------------------------------
    print("\n--- Step 6: Aligning Column Names with News Dataset ---")
    df = df.rename(columns={'cleaned_text': 'cleaned_headline',
                             'text': 'headline'})
    cols = ['headline', 'cleaned_headline'] + feature_cols + ['is_sarcastic']
    df = df[cols]
    print(f"Final columns: {list(df.columns)}")
 
    # ----------------------------------------------------------------
    # STEP 7: Save processed GEN dataset
    # ----------------------------------------------------------------
    print("\n--- Step 7: Saving Processed GEN Dataset ---")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
 
    print("\n" + "="*60)
    print(" MODULE 05 EXECUTION COMPLETED SUCCESSFULLY ")
    print("="*60)
    print(f"\nOutput: data/processed/cleaned_gen_data.csv")
    print("Next step: python scripts/06_modeling.py")
 
if __name__ == "__main__":
    run_gen_preprocessing()