import pandas as pd
import os
import re

INPUT_PATH = os.path.join("data", "raw", "Sarcasm_Headlines_Dataset.json")
OUTPUT_PATH = os.path.join("data", "processed", "cleaned_sarcasm_data.csv")

CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he's": "he is", "how's": "how is",
    "I'm": "I am", "I've": "I have", "I'll": "I will", "I'd": "I would",
    "isn't": "is not", "it's": "it is", "let's": "let us", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is",
    "where's": "where is", "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would"
}

def expand_contractions(text):
    for word, expanded in CONTRACTION_MAP.items():
        text = re.sub(re.escape(word), expanded, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    text = str(text)
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing():
    print("--- Step 1: Loading Raw Data ---")
    df = pd.read_json(INPUT_PATH, lines=True)
    initial_shape = df.shape[0]

    print("--- Step 2: Dropping Duplicates and Unnecessary Columns ---")
    df = df.drop_duplicates(subset=['headline'])
    if 'article_link' in df.columns:
        df = df.drop(columns=['article_link'])

    print("--- Step 3: Advanced Feature Engineering ---")
    df['char_count'] = df['headline'].apply(lambda x: len(str(x)))
    df['word_count'] = df['headline'].apply(lambda x: len(str(x).split()))

    # FIXED: Correctly calculates average word length by summing
    # individual word lengths, excluding whitespace characters.
    # The old formula (char_count / word_count) incorrectly included
    # spaces in the character count, skewing the feature for RQ1.
    df['avg_word_length'] = df['headline'].apply(
        lambda x: sum(len(w) for w in str(x).split()) / len(str(x).split())
        if len(str(x).split()) > 0 else 0
    )

    df['quote_count'] = df['headline'].apply(lambda x: str(x).count('"') + str(x).count("'"))
    df['exclamation_count'] = df['headline'].apply(lambda x: str(x).count('!'))
    df['question_mark_count'] = df['headline'].apply(lambda x: str(x).count('?'))

    print("--- Step 4: Text Cleaning ---")
    df['cleaned_headline'] = df['headline'].apply(clean_text)

    print("--- Step 5: Academic Data Integrity Checks (Filtering) ---")
    df = df[df['cleaned_headline'].str.strip() != '']
    df = df[df['cleaned_headline'].apply(lambda x: len(str(x).split()) > 2)]
    print(f"Rows dropped due to integrity filters: {initial_shape - df.shape[0]}")

    print("--- Step 6: Saving Processed Data ---")
    feature_cols = ['char_count', 'word_count', 'avg_word_length', 'quote_count',
                    'exclamation_count', 'question_mark_count']
    cols = ['headline', 'cleaned_headline'] + feature_cols + ['is_sarcastic']
    df = df[cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS! Processed data saved to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    run_preprocessing()