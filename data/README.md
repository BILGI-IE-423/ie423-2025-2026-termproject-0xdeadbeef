## Dataset Access Instructions

### News Headlines Dataset for Sarcasm Detection
- **Source:** https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
- **File:** `Sarcasm_Headlines_Dataset.json`
- **Format:** JSON Lines (one JSON object per line), inside a zip archive
- **Size:** 28619 article link, 28619 headlines, 3 columns (`article_link`, `headline`, `is_sarcastic`)
- **Placement:** Download and place the zip file inside `data/raw/`

## Processed Data

After running `scripts/02_preprocess_data.py`, the cleaned dataset is saved to:
- `data/processed/cleaned_sarcasm_data.csv`
