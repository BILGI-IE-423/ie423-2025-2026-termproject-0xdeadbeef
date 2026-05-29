import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
 
# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
NEWS_PATH    = os.path.join("data", "processed", "cleaned_sarcasm_data.csv")
GEN_PATH     = os.path.join("data", "processed", "cleaned_gen_data.csv")
FIGURES_PATH = os.path.join("visuals", "figures")
TABLES_PATH  = os.path.join("visuals", "tables")
 
COLOR_NORMAL  = '#3182bd'
COLOR_SARCASM = '#e6550d'
COLOR_NEUTRAL = '#969696'
 
POSITIVE_WORDS = [
    'great', 'amazing', 'wonderful', 'fantastic', 'brilliant',
    'excellent', 'perfect', 'best', 'incredible', 'outstanding'
]
 
# ----------------------------------------------------------------
# Helper: apply VADER and TextBlob to a dataframe
# ----------------------------------------------------------------
def apply_sentiment(df, text_col='headline'):
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df['vader_compound'] = df[text_col].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound'])
    df['vader_label'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05
        else ('negative' if x <= -0.05 else 'neutral'))
    df['textblob_score'] = df[text_col].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity)
    df['textblob_label'] = df['textblob_score'].apply(
        lambda x: 'positive' if x > 0.05
        else ('negative' if x < -0.05 else 'neutral'))
    return df
 
def misleading_rate(df, label_col, target=1):
    sarc = df[df['is_sarcastic'] == target]
    return (sarc[label_col] == 'positive').mean() * 100
 
# ----------------------------------------------------------------
# Helper: clean axis style
# ----------------------------------------------------------------
def clean_ax(ax, remove_top=True, remove_right=True, remove_left=False):
    ax.spines['top'].set_visible(not remove_top)
    ax.spines['right'].set_visible(not remove_right)
    ax.spines['left'].set_visible(not remove_left)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
 
def add_bar_labels(ax, fmt='{:.1f}%', offset=0.5, fontsize=11):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x() + p.get_width() / 2,
                    h + offset,
                    fmt.format(h),
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold')
 
# ================================================================
# MAIN
# ================================================================
def run_sentiment_analysis():
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(TABLES_PATH, exist_ok=True)
    sns.set_theme(style="white")
 
    # ================================================================
    # PART 1 — News dataset sentiment analysis (RQ2)
    # ================================================================
    print("="*60)
    print(" PART 1: NEWS DATASET — SENTIMENT ANALYSIS (RQ2) ")
    print("="*60)
 
    print("\n--- Step 1: Loading News Dataset ---")
    news_df = pd.read_csv(NEWS_PATH)
    news_df = apply_sentiment(news_df, text_col='headline')
    print(f"Loaded: {news_df.shape[0]} rows")
 
    # ── Per-class summary ─────────────────────────────────────────
    print("\n--- Step 2: Per-Class Sentiment Summary ---")
    for tool, col in [('VADER', 'vader_compound'),
                      ('TextBlob', 'textblob_score')]:
        summary = news_df.groupby('is_sarcastic')[col].describe().round(3)
        summary.index = summary.index.map({0: 'Normal', 1: 'Sarcastic'})
        print(f"\n{tool} Summary (News):")
        print(summary)
        summary.to_csv(
            os.path.join(TABLES_PATH,
                         f'sentiment_summary_{tool.lower()}_news.csv'))
        print(f"Saved: visuals/tables/sentiment_summary_{tool.lower()}_news.csv")
 
    # ── Misleading rates ──────────────────────────────────────────
    print("\n--- Step 3: Misleading Rate (News) ---")
    vader_news    = misleading_rate(news_df, 'vader_label')
    textblob_news = misleading_rate(news_df, 'textblob_label')
    print(f"VADER    — sarcastic headlines scored POSITIVE: {vader_news:.1f}%")
    print(f"TextBlob — sarcastic headlines scored POSITIVE: {textblob_news:.1f}%")
 
    # ── Score distribution plot (News) ───────────────────────────
    print("\n--- Step 4: Score Distribution Plots (News) ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('VADER vs TextBlob — Sentiment Score Distribution\n'
                 '(News Headlines)',
                 fontsize=14, fontweight='bold', y=1.01)
 
    for col_idx, (tool, col) in enumerate([
        ('VADER', 'vader_compound'),
        ('TextBlob', 'textblob_score')
    ]):
        for row_idx, (label, cls) in enumerate([('Normal', 0), ('Sarcastic', 1)]):
            color = COLOR_NORMAL if cls == 0 else COLOR_SARCASM
            ax    = axes[row_idx][col_idx]
            sns.histplot(data=news_df[news_df['is_sarcastic'] == cls],
                         x=col, bins=40, color=color, ax=ax, alpha=0.85)
            ax.set_title(f'{tool}: {label} Headlines', fontweight='bold')
            ax.set_xlabel('Sentiment Score (−1 = very negative, +1 = very positive)')
            ax.set_ylabel('Number of Headlines')
            ax.axvline(x=0.05, color='red', linestyle='--',
                       linewidth=1.2, alpha=0.7, label='Positive threshold (0.05)')
            ax.axvline(x=-0.05, color='orange', linestyle='--',
                       linewidth=1.2, alpha=0.7, label='Negative threshold (−0.05)')
            ax.legend(fontsize=8)
            clean_ax(ax)
 
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'RQ2_Sentiment_Score_Distribution.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/sentiment_distribution_news.png")
 
 
 
    # ── Positive word frequency (news) ────────────────────────────
    print("\n--- Step 5: Positive Word Frequency (News) ---")
    rows_news = []
    sarc_news = news_df[news_df['is_sarcastic'] == 1]
    norm_news = news_df[news_df['is_sarcastic'] == 0]
    for word in POSITIVE_WORDS:
        s_pct = sarc_news['headline'].str.lower().str.contains(
            r'\b' + word + r'\b', regex=True).mean() * 100
        n_pct = norm_news['headline'].str.lower().str.contains(
            r'\b' + word + r'\b', regex=True).mean() * 100
        rows_news.append({'word': word,
                          'sarcastic_%': round(s_pct, 2),
                          'normal_%':    round(n_pct, 2)})
    word_news_df = pd.DataFrame(rows_news)
    print(word_news_df.to_string(index=False))
    word_news_df.to_csv(
        os.path.join(TABLES_PATH, 'positive_word_frequency_news.csv'),
        index=False)
    print("Saved: visuals/tables/positive_word_frequency_news.csv")
 
    x     = range(len(word_news_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    bars_n = ax.bar([i - width/2 for i in x], word_news_df['normal_%'],
                    width, label='Normal', color=COLOR_NORMAL, alpha=0.85,
                    edgecolor='white')
    bars_s = ax.bar([i + width/2 for i in x], word_news_df['sarcastic_%'],
                    width, label='Sarcastic', color=COLOR_SARCASM, alpha=0.85,
                    edgecolor='white')
    for bar in list(bars_n) + list(bars_s):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.02, f'{h:.2f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(word_news_df['word'], rotation=25, ha='right',
                       fontsize=11)
    ax.set_ylabel('Frequency in Headlines (%)', fontsize=11)
    ax.set_title('RQ2: How Often Do Positive Words Appear\n'
                 'in Sarcastic vs Normal Headlines?',
                 fontweight='bold', fontsize=13)
    ax.legend(title='Headline Type', fontsize=10)
    ax.set_ylim(0, max(word_news_df[['normal_%', 'sarcastic_%']].max()) * 1.3)
    clean_ax(ax, remove_left=True)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'RQ2_Positive_Word_Frequency.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/positive_word_frequency_news.png")
 
    # ── Most deceiving headlines (news) ───────────────────────────
    print("\n--- Step 6: Most Deceiving Headlines (News) ---")
    deceiving_news = (
        news_df[news_df['is_sarcastic'] == 1]
        .nlargest(10, 'vader_compound')
        [['headline', 'vader_compound', 'textblob_score']]
        .rename(columns={'headline':       'Headline',
                         'vader_compound': 'VADER Score',
                         'textblob_score': 'TextBlob Score'})
        .round(3)
    )
    print(deceiving_news.to_string(index=False))
    deceiving_news.to_csv(
        os.path.join(TABLES_PATH, 'most_deceiving_headlines_news.csv'),
        index=False)
    print("Saved: visuals/tables/most_deceiving_headlines_news.csv")
 
    # ================================================================
    # PART 2 — GEN dataset sentiment analysis (Cross-Domain RQ2)
    # ================================================================
    print("\n" + "="*60)
    print(" PART 2: GEN DATASET — SENTIMENT ANALYSIS (CROSS-DOMAIN RQ2) ")
    print("="*60)
 
    if not os.path.exists(GEN_PATH):
        print(f"[WARNING] {GEN_PATH} not found. Run 05_preprocess_gen.py first.")
    else:
        print("\n--- Step 7: Loading GEN Dataset ---")
        gen_df = pd.read_csv(GEN_PATH)
        gen_df = apply_sentiment(gen_df, text_col='headline')
        print(f"Loaded: {gen_df.shape[0]} rows")
 
        # ── Per-class summary (GEN) ───────────────────────────────
        print("\n--- Step 8: Per-Class Sentiment Summary (GEN) ---")
        for tool, col in [('VADER', 'vader_compound'),
                          ('TextBlob', 'textblob_score')]:
            summary = gen_df.groupby('is_sarcastic')[col].describe().round(3)
            summary.index = summary.index.map({0: 'Normal', 1: 'Sarcastic'})
            print(f"\n{tool} Summary (GEN):")
            print(summary)
            summary.to_csv(
                os.path.join(TABLES_PATH,
                             f'sentiment_summary_{tool.lower()}_gen.csv'))
            print(f"Saved: visuals/tables/sentiment_summary_{tool.lower()}_gen.csv")
 
        # ── Misleading rates (GEN) ────────────────────────────────
        print("\n--- Step 9: Misleading Rate (GEN) ---")
        vader_gen    = misleading_rate(gen_df, 'vader_label')
        textblob_gen = misleading_rate(gen_df, 'textblob_label')
        print(f"VADER    — sarcastic comments scored POSITIVE: {vader_gen:.1f}%")
        print(f"TextBlob — sarcastic comments scored POSITIVE: {textblob_gen:.1f}%")
 
        # ── Cross-domain misleading rate comparison ───────────────
        print("\n--- Step 10: Cross-Domain Misleading Rate Comparison ---")
        comparison_data = pd.DataFrame([
            {'Dataset': 'News Headlines', 'Tool': 'VADER',
             'Misleading Rate (%)': vader_news},
            {'Dataset': 'News Headlines', 'Tool': 'TextBlob',
             'Misleading Rate (%)': textblob_news},
            {'Dataset': 'GEN (Forum)',    'Tool': 'VADER',
             'Misleading Rate (%)': vader_gen},
            {'Dataset': 'GEN (Forum)',    'Tool': 'TextBlob',
             'Misleading Rate (%)': textblob_gen},
        ])
        print(comparison_data.to_string(index=False))
        comparison_data.to_csv(
            os.path.join(TABLES_PATH, 'misleading_rate_crossdomain.csv'),
            index=False)
        print("Saved: visuals/tables/misleading_rate_crossdomain.csv")
 
        # Side-by-side: VADER and TextBlob, News vs GEN
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            'RQ2: What Percentage of Sarcastic Texts Are Mistakenly Scored as Positive?\n'
            'News Headlines vs Forum Comments | VADER and TextBlob',
            fontsize=13, fontweight='bold')
 
        for ax, tool in zip(axes, ['VADER', 'TextBlob']):
            sub    = comparison_data[comparison_data['Tool'] == tool]
            colors = ['#e6550d', '#fd8d3c']  # orange palette (news=dark, forum=light)
            bars   = ax.bar(['News Headlines', 'Forum Comments'],
                            sub['Misleading Rate (%)'].values,
                            color=colors, alpha=0.88, width=0.45,
                            edgecolor='white')
            for bar, val in zip(bars, sub['Misleading Rate (%)'].values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.8, f'{val:.1f}%',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=13)
            ax.set_title(tool, fontweight='bold', fontsize=13)
            ax.set_ylabel('Sarcastic Texts Scored as Positive (%)', fontsize=11)
            ax.set_ylim(0, 60)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter())
            clean_ax(ax, remove_left=True)
            ax.set_yticks([])
 
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_PATH, 'RQ2_Misleading_Rate_News_vs_Forum.png'),
            bbox_inches='tight', dpi=150)
        plt.close()
        print("Saved: visuals/figures/misleading_rate_crossdomain.png")
 
        # ── Most deceiving comments (GEN) ─────────────────────────
        print("\n--- Step 11: Most Deceiving Forum Comments (GEN) ---")
        deceiving_gen = (
            gen_df[gen_df['is_sarcastic'] == 1]
            .nlargest(10, 'vader_compound')
            [['headline', 'vader_compound', 'textblob_score']]
            .rename(columns={'headline':       'Comment',
                             'vader_compound': 'VADER Score',
                             'textblob_score': 'TextBlob Score'})
            .round(3)
        )
        print(deceiving_gen.to_string(index=False))
        deceiving_gen.to_csv(
            os.path.join(TABLES_PATH, 'most_deceiving_comments_gen.csv'),
            index=False)
        print("Saved: visuals/tables/most_deceiving_comments_gen.csv")
 
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("SUCCESS! Sentiment analysis complete.")
    print("\nPart 1 — News Dataset:")
    print(f"  VADER misleading rate    : {vader_news:.1f}%")
    print(f"  TextBlob misleading rate : {textblob_news:.1f}%")
    print("  Figures → visuals/figures/sentiment_distribution_news.png")
    print("          → visuals/figures/misleading_rate_news.png")
    print("          → visuals/figures/positive_word_frequency_news.png")
    print("  Tables  → visuals/tables/sentiment_summary_vader_news.csv")
    print("          → visuals/tables/sentiment_summary_textblob_news.csv")
    print("          → visuals/tables/positive_word_frequency_news.csv")
    print("          → visuals/tables/most_deceiving_headlines_news.csv")
    if os.path.exists(GEN_PATH):
        print("\nPart 2 — GEN Dataset (Cross-Domain):")
        print(f"  VADER misleading rate    : {vader_gen:.1f}%")
        print(f"  TextBlob misleading rate : {textblob_gen:.1f}%")
        print("  Figures → visuals/figures/misleading_rate_crossdomain.png")
        print("  Tables  → visuals/tables/misleading_rate_crossdomain.csv")
        print("          → visuals/tables/sentiment_summary_vader_gen.csv")
        print("          → visuals/tables/sentiment_summary_textblob_gen.csv")
        print("          → visuals/tables/most_deceiving_comments_gen.csv")
    print("="*60)
 
if __name__ == "__main__":
    run_sentiment_analysis()