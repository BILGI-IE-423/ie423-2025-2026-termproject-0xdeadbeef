# Data Directory

## Dataset Access Instructions

### News Headlines Dataset for Sarcasm Detection
- **Source:** https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
- **File:** `Sarcasm_Headlines_Dataset.json`
- **Format:** JSON Lines (one JSON object per line), inside a zip archive
- **Size:** 28,619 headlines, 2 columns (`is_sarcastic`, `headline`)
- **Placement:** Download and place the zip file inside `data/raw/`

### How to obtain the dataset
1. Go to the Kaggle link above
2. Download the dataset (zip file)
3. Place `Sarcasm_Headlines_Dataset.json` inside `data/raw/`
4. The scripts will automatically extract the JSON files when run

## Processed Data

After running `scripts/02_preprocess_data.py`, the cleaned dataset is saved to:
- `data/processed/cleaned_sarcasm_data.csv`
