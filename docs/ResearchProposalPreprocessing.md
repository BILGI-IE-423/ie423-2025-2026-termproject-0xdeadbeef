# IE 423 Term Project Proposal — Decoding Sarcasm in News Headlines: A Feature Engineering and Classification Approach

---

## Team Information

- Ada Güner Nohut
- Veli Erenay Açıl
- Onur Sarıdoğan

---

## Dataset Description

We use the **News Headlines Dataset for Sarcasm Detection**, obtained from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection).

This dataset contains **28,619 news headlines** collected from two sources:
- **The Onion** — a satirical news outlet known for sarcastic and ironic headlines
- **HuffPost (Huffington Post)** — a mainstream news outlet with straightforward headlines

Each headline is labeled as either **sarcastic (1)** or **not sarcastic (0)**. The dataset was curated by Rishabh Misra and is widely used in NLP research on sarcasm and irony detection.

We selected this dataset because sarcasm is one of the most challenging phenomena in natural language processing — it requires understanding the gap between literal meaning and intended meaning. Unlike datasets that rely on self-reported sarcasm tags (like Reddit's `/s`), this dataset derives its labels from the inherent nature of the source (The Onion is sarcastic by design), making the labels more reliable and consistent.

---

## Dataset Access

The dataset is stored in:

`data/raw/Sarcasm_Headlines_Dataset_v2.json`

If the dataset is not present, it can be downloaded from:

[https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

After downloading, place the file inside:

`data/raw/`

---

## Research Questions

### Research Question 1 (Statistical)
**Is there a statistically significant difference between sarcastic and non-sarcastic (literal) headlines in terms of word count, sentence length, and punctuation usage (e.g., exclamation marks)?**

**Explanation:**
Sarcasm detection literature often assumes that sarcastic and non-sarcastic texts differ stylistically, but this assumption is rarely tested with formal statistical rigor on headline-level data. This question directly measures whether the structural features we engineer — word count and punctuation marks (exclamation points, question marks) — show statistically significant differences between sarcastic and non-sarcastic headlines using hypothesis tests (e.g., Mann-Whitney U, t-test). Our initial EDA already hints at subtle differences: sarcastic headlines use fewer words on average (9.72 vs 10.36) and fewer question marks (0.014 vs 0.042). However, visual differences can be misleading without proper significance testing. By applying statistical tests with effect size calculations (Cohen's d), we will determine whether these differences are genuine structural signatures of sarcasm or merely noise. If significant differences emerge, these features become reliable inputs for classification models; if not, it would demonstrate that sarcasm hides within semantics rather than structure — a critical finding for feature engineering strategy.

### Research Question 2 (Linguistic Paradox)
**How do sarcastic texts deceive standard sentiment analysis libraries, and what is the frequency of positive words (e.g., "great," "amazing," "perfect") appearing within sarcastic headlines?**

**Explanation:**
Sarcasm is fundamentally a linguistic paradox: it uses positive words to convey negative meaning, or vice versa. A headline like "What a wonderful day to realize nothing matters" contains overtly positive vocabulary ("wonderful") but carries a deeply negative or ironic sentiment. Standard sentiment analysis tools (such as VADER, TextBlob, or NLTK's sentiment module) assign sentiment scores based on word-level polarity, making them inherently vulnerable to sarcasm. This question investigates the frequency and distribution of positively-valenced words (e.g., "great," "love," "amazing," "perfect," "best") within sarcastic headlines and measures how often standard sentiment analyzers misclassify sarcastic text as positive. By computing sentiment polarity scores for both sarcastic and non-sarcastic headlines and comparing the distributions, we can quantify the "deception rate" of sarcasm — the percentage of sarcastic headlines that fool sentiment tools into producing positive scores. This analysis reveals the fundamental limitation of lexicon-based sentiment analysis and motivates the need for context-aware approaches.

### Research Question 3 (Machine Learning)
**Can traditional machine learning models (e.g., Logistic Regression or SVM) detect sarcasm by relying solely on word frequencies (TF-IDF), or do they require word groups (N-grams) to capture contextual meaning?**

**Explanation:**
The core challenge of sarcasm detection lies in whether word-level information is sufficient or whether the model needs to capture word sequences and context. TF-IDF with unigrams treats each word independently — it can learn that "area" and "man" are individually associated with sarcasm, but it cannot learn that "area man" as a phrase is a strong sarcasm indicator. By systematically comparing the performance of Logistic Regression and SVM models trained on (a) unigram TF-IDF features only versus (b) bigram and trigram TF-IDF features (N-grams), we can directly measure how much contextual word ordering contributes to sarcasm detection. If N-gram models significantly outperform unigram models, it would demonstrate that sarcasm relies on specific multi-word patterns and phrases rather than individual keyword signals. Conversely, if unigrams perform comparably, it would suggest that the vocabulary alone — without word order — carries most of the discriminative signal. This comparison provides actionable guidance on feature representation choices for sarcasm detection systems and establishes a clear baseline before considering more complex deep learning approaches.

---

## Project Proposal

This project aims to investigate **sarcasm detection in news headlines** from three complementary angles: statistical analysis of structural features, linguistic analysis of how sarcasm deceives sentiment tools, and machine learning experiments comparing feature representation strategies.

**Phase 1 — Data Preprocessing:**
We will clean the raw dataset by removing duplicates, expanding English contractions (e.g., "won't" → "will not"), applying advanced text cleaning (lowercase, remove numbers, remove punctuation), and engineering text features including word count and punctuation counts. The cleaned dataset is saved to `data/processed/cleaned_sarcasm_data.csv`.

**Phase 2 — Exploratory Data Analysis (EDA):**
We will conduct thorough EDA to understand the distributions, patterns, and differences between sarcastic and non-sarcastic headlines. This includes visualizing label distributions, comparing text features across classes, analyzing vocabulary differences through word clouds and frequency analysis, and examining punctuation usage patterns.

**Phase 3 — Statistical Hypothesis Testing (RQ1):**
We will apply formal statistical tests (Mann-Whitney U, independent t-test) to determine whether word count and punctuation usage differ significantly between sarcastic and non-sarcastic headlines. Effect sizes (Cohen's d) will quantify the practical magnitude of any detected differences.

**Phase 4 — Sentiment Paradox Analysis (RQ2):**
We will apply standard sentiment analysis tools (VADER, TextBlob) to both sarcastic and non-sarcastic headlines and measure the "deception rate" — how often sarcastic headlines are misclassified as positive. We will also compute the frequency of positively-valenced words within sarcastic text to quantify the linguistic paradox.

**Phase 5 — TF-IDF vs N-gram Classification (RQ3):**
We will train Logistic Regression and SVM models using (a) unigram TF-IDF features and (b) bigram/trigram TF-IDF features, then compare accuracy, precision, recall, and F1-score to determine whether word context (N-grams) significantly improves sarcasm detection over individual word frequencies.

**Expected Outcomes:**
- Statistical evidence on whether structural features (word count, punctuation) are reliable sarcasm indicators
- A quantified "deception rate" showing how sarcasm fools standard sentiment analysis tools
- A direct comparison of unigram vs N-gram performance for sarcasm classification

**Possible Challenges:**
- The dataset is sourced from only two outlets (The Onion and HuffPost), so the model may learn source-specific style rather than general sarcasm patterns
- Sentiment analysis tools may behave differently on short headline text compared to full-length reviews
- Class imbalance (14,951 vs 13,552) may require stratified sampling in model evaluation

## Preprocessing Steps

### Step 1 — Loading the Data
The dataset was loaded using `scripts/01_load_data.py`. The script reads the JSON Lines format file, converts it to a pandas DataFrame, and prints basic information including shape, column names, data types, and missing values.

**Result:** 28,619 rows, 2 columns (`is_sarcastic`, `headline`), no missing values.

### Step 2 — Initial Inspection
We checked:
- Shape of the dataset: 28,619 rows x 2 columns
- Column names and data types: `is_sarcastic` (int64), `headline` (str)
- Missing values: **None** across all columns
- Label distribution: 14,985 not sarcastic, 13,634 sarcastic (slightly imbalanced)

### Step 3 — Cleaning and Feature Engineering
Using `scripts/02_preprocess_data.py`, we performed the following:

| Step | Description |
|------|-------------|
| Standardize column names | Lowercase, strip whitespace |
| Remove duplicates | Dropped 116 duplicate headlines |
| Expand contractions | e.g., "won't" → "will not", "it's" → "it is" |
| Advanced text cleaning | Lowercase, remove numbers, remove punctuation, normalize whitespace |
| word_count | Number of words in original headline |
| exclamation_count | Number of `!` characters (counted before cleaning) |
| question_mark_count | Number of `?` characters (counted before cleaning) |

### Step 4 — Saving Processed Data
The cleaned dataset was saved to:

`data/processed/cleaned_sarcasm_data.csv`

**Final shape:** 28,503 rows x 6 columns (after removing 116 duplicates)

Final columns: `headline`, `cleaned_headline`, `word_count`, `exclamation_count`, `question_mark_count`, `is_sarcastic`

---

## Initial Outputs

### Dataset Shape
After preprocessing, the dataset contains **28,503 rows** and **6 columns**.

### Missing Value Summary
No missing values exist in any column after preprocessing.

### Feature Statistics by Label

| Feature | Sarcastic (mean) | Not Sarcastic (mean) |
|---------|-------------------|----------------------|
| word_count | 9.72 | 10.36 |
| exclamation_count | 0.006 | 0.011 |
| question_mark_count | 0.014 | 0.042 |

### Visualizations

All figures below were generated by `scripts/03_basic_eda.py`.

#### Figure 1 — Sarcasm Label Distribution
![Label Distribution](../outputs/figures/01_label_distribution.png)

The dataset is relatively balanced with 14,951 non-sarcastic and 13,552 sarcastic headlines.

---

#### Figure 2 — Word Count Density (KDE)
![Word Count KDE](../outputs/figures/02_word_count_kde.png)

The KDE plot reveals that both classes have similar word count distributions peaking around 8–12 words, but non-sarcastic headlines exhibit a slightly heavier right tail, suggesting they tend to be wordier.

---

#### Figure 3 — Top 20 Most Common Words
![Top Words](../outputs/figures/03_top_words_comparison.png)

Sarcastic headlines frequently use words like "man," "area," "nation," "new," and "report" — reflecting The Onion's signature style of mimicking real news. Non-sarcastic headlines favor words like "trump," "new," "women," and "people."

---

#### Figure 4 — Punctuation Usage Comparison
![Punctuation Comparison](../outputs/figures/04_punctuation_comparison.png)

Non-sarcastic headlines use more question marks on average, while both classes show minimal use of exclamation marks.

---

#### Figure 5 — Word Cloud: Sarcastic vs Not Sarcastic
![Word Clouds](../outputs/figures/05_word_clouds.png)

Word clouds visualize the most frequent words in each class. Sarcastic headlines are dominated by words like "area," "man," "nation," and "report" — The Onion's signature vocabulary for mimicking real news. Non-sarcastic headlines center around "trump," "new," "people," and "women," reflecting actual news topics.

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone [your-repository-link]
cd [repository-name]
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) and place the file inside:
```
data/raw/
```

### 4. Run the scripts in order
```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

---

## Transparency and Traceability

All outputs presented in this markdown file are generated from the Python scripts in the `scripts/` folder.

| Output Type | Location | Generated By |
|-------------|----------|--------------|
| Figures (5 plots) | `outputs/figures/` | `scripts/03_basic_eda.py` |
| Summary Tables | `outputs/tables/` | `scripts/02_preprocess_data.py` |
| Cleaned Dataset | `data/processed/cleaned_sarcasm_data.csv` | `scripts/02_preprocess_data.py` |

The repository is designed so that another user can reproduce the same outputs by:
1. Installing the required packages (`requirements.txt`)
2. Placing the dataset correctly (`data/raw/`)
3. Running the scripts in order (`01` → `02` → `03`)

Every figure and table shown in this document is traceable to a specific script and can be reproduced independently.
