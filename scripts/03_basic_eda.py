import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import CountVectorizer
import glob

# Define file paths
INPUT_PATH = os.path.join("data", "processed", "cleaned_sarcasm_data.csv")
FIGURES_PATH = os.path.join("outputs", "figures")
TABLES_PATH = os.path.join("outputs", "tables")

COLOR_NORMAL = '#3182bd'
COLOR_SARCASM = '#e6550d'

def run_eda():
    print("--- Step 0: Cleaning old figures ---")
    old_files = glob.glob(os.path.join(FIGURES_PATH, '*'))
    for f in old_files:
        os.remove(f)

    print("--- Step 1: Loading Processed Data ---")
    df = pd.read_csv(INPUT_PATH)
    # Defensive programming: verify data integrity before EDA,
    # even though 02_preprocess_data.py already handles nulls.
    df = df.dropna(subset=['cleaned_headline'])

    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(TABLES_PATH, exist_ok=True)

    sns.set_theme(style="whitegrid")

    print("--- Step 1.5: Generating Statistical Summary Table ---")
    # Produces a per-class mean comparison table for all engineered features.
    # This directly supports RQ1 by showing whether stylistic features
    # (exclamation, quote counts) differ systematically between classes.
    numeric_features = ['char_count', 'word_count', 'avg_word_length',
                        'quote_count', 'exclamation_count', 'question_mark_count']
    summary_table = df.groupby('is_sarcastic')[numeric_features].mean().round(2)
    summary_table.index = summary_table.index.map({0: 'Normal', 1: 'Sarcastic'})
    summary_table.to_csv(os.path.join(TABLES_PATH, 'feature_summary_by_class.csv'))
    print("Saved: outputs/tables/feature_summary_by_class.csv")

    print("--- Step 2: Generating Target Distribution (Bar Chart) ---")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='is_sarcastic', data=df, palette=[COLOR_NORMAL, COLOR_SARCASM])
    plt.title('Class Balance: Normal vs Sarcastic', fontsize=14, weight='bold')
    plt.xlabel('Category (0: Normal, 1: Sarcastic)')
    plt.ylabel('Count')

    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + 200
        ax.annotate(percentage, (x, y), size=12, weight='bold', color='#252525')

    plt.savefig(os.path.join(FIGURES_PATH, 'target_distribution_bar.png'), bbox_inches='tight')
    plt.close()

    print("--- Step 3: Generating Length Distribution (Side-by-Side Histograms) ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.histplot(data=df[df['is_sarcastic'] == 0], x='word_count', bins=30,
                 color=COLOR_NORMAL, ax=axes[0])
    axes[0].set_title('Word Count: Normal Headlines', weight='bold')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')

    sns.histplot(data=df[df['is_sarcastic'] == 1], x='word_count', bins=30,
                 color=COLOR_SARCASM, ax=axes[1])
    axes[1].set_title('Word Count: Sarcastic Headlines', weight='bold')
    axes[1].set_xlabel('Word Count')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'length_distribution_hist.png'), bbox_inches='tight')
    plt.close()

    print("--- Step 4: Generating Outlier Check (Boxplot) ---")
    plt.figure(figsize=(10, 6))
    flierprops = dict(marker='o', markerfacecolor='#969696', markersize=4,
                      linestyle='none', markeredgecolor='none', alpha=0.4)
    sns.boxplot(x='is_sarcastic', y='char_count', data=df,
                palette=[COLOR_NORMAL, COLOR_SARCASM], flierprops=flierprops)
    plt.title('Character Count Spread & Outliers', fontsize=14, weight='bold')
    plt.xlabel('Is Sarcastic')
    plt.ylabel('Total Character Count')
    plt.savefig(os.path.join(FIGURES_PATH, 'character_count_boxplot.png'), bbox_inches='tight')
    plt.close()

    print("--- Step 5: Generating Correlation Heatmap ---")
    numeric_cols = ['is_sarcastic', 'char_count', 'word_count', 'avg_word_length',
                    'quote_count', 'exclamation_count', 'question_mark_count']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Engineered Features', fontsize=14, weight='bold')
    plt.savefig(os.path.join(FIGURES_PATH, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

    print("--- Step 6: Generating Top Bigrams Bar Charts ---")
    def get_top_ngrams(corpus, n=10, ngram_range=(2, 2)):
        vec = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

    sarcastic_text = df[df['is_sarcastic'] == 1]['cleaned_headline']
    df_sarcastic_bigram = pd.DataFrame(get_top_ngrams(sarcastic_text, 10),
                                       columns=['Bigram', 'Frequency'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Bigram', data=df_sarcastic_bigram, palette='Oranges_r')
    plt.title('Top 10 Bigrams in Sarcastic Headlines', weight='bold')
    plt.savefig(os.path.join(FIGURES_PATH, 'top_bigrams_sarcastic.png'), bbox_inches='tight')
    plt.close()

    normal_text = df[df['is_sarcastic'] == 0]['cleaned_headline']
    df_normal_bigram = pd.DataFrame(get_top_ngrams(normal_text, 10),
                                    columns=['Bigram', 'Frequency'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Bigram', data=df_normal_bigram, palette='Blues_r')
    plt.title('Top 10 Bigrams in Normal Headlines', weight='bold')
    plt.savefig(os.path.join(FIGURES_PATH, 'top_bigrams_normal.png'), bbox_inches='tight')
    plt.close()

    print(f"\nSUCCESS! All EDA outputs generated.")
    print(f"  Figures -> outputs/figures/ ({len(glob.glob(os.path.join(FIGURES_PATH, '*')))} files)")
    print(f"  Tables  -> outputs/tables/feature_summary_by_class.csv")

if __name__ == "__main__":
    run_eda()