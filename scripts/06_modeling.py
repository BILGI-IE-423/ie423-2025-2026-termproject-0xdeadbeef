import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split, GridSearchCV)
from sklearn.metrics import (make_scorer, f1_score, precision_score,
                             recall_score, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')
 
# ----------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------
NEWS_PATH    = os.path.join("data", "processed", "cleaned_sarcasm_data.csv")
GEN_PATH     = os.path.join("data", "processed", "cleaned_gen_data.csv")
FIGURES_PATH = os.path.join("visuals", "figures")
TABLES_PATH  = os.path.join("visuals", "tables")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
K_FOLDS      = 5
 
ENGINEERED_FEATURES = [
    'char_count', 'word_count', 'avg_word_length',
    'quote_count', 'exclamation_count', 'question_mark_count'
]
 
# Class colors (used in sentiment/distribution plots)
COLOR_NORMAL = '#3182bd'   # blue  — normal class
COLOR_SARC   = '#e6550d'   # orange — sarcastic class
 
# Model colors (used in comparison plots)
COLOR_LR     = '#2ca02c'   # green  — Logistic Regression
COLOR_SVM    = '#9467bd'   # purple — LinearSVC
COLOR_RF     = '#8c564b'   # brown  — Random Forest
 
# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def get_cv_scores(pipeline, X, y, cv):
    scoring = {
        'accuracy':  'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall':    make_scorer(recall_score,    zero_division=0),
        'f1':        make_scorer(f1_score,        zero_division=0)
    }
    res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'Accuracy':  res['test_accuracy'].mean(),
        'Precision': res['test_precision'].mean(),
        'Recall':    res['test_recall'].mean(),
        'F1':        res['test_f1'].mean(),
        'F1_std':    res['test_f1'].std()
    }
 
def clean_ax(ax, remove_top=True, remove_right=True, remove_left=False):
    ax.spines['top'].set_visible(not remove_top)
    ax.spines['right'].set_visible(not remove_right)
    ax.spines['left'].set_visible(not remove_left)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
 
def add_bar_value(ax, bar, val, fmt='{:.3f}', offset=0.003, fontsize=9):
    ax.text(val + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va='center', fontsize=fontsize, fontweight='bold')
 
def plot_confusion_matrix(y_true, y_pred, title, path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
 
    # Diagonal (correct predictions) → green, off-diagonal → white
    COLOR_CORRECT   = '#2171b5'   # professional blue for correct predictions
    COLOR_INCORRECT = '#f0f0f0'   # light grey for incorrect predictions
    TEXT_ON_GREEN   = 'white'
    TEXT_ON_WHITE   = '#222222'
 
    for i in range(2):
        for j in range(2):
            is_correct = (i == j)
            bg_color   = COLOR_CORRECT if is_correct else COLOR_INCORRECT
            txt_color  = TEXT_ON_GREEN if is_correct else TEXT_ON_WHITE
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                facecolor=bg_color, edgecolor='#cccccc', linewidth=1))
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=18, fontweight='bold', color=txt_color)
 
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Normal', 'Sarcastic'], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Normal', 'Sarcastic'], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
 
# ================================================================
# MAIN
# ================================================================
def run_modeling():
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(TABLES_PATH, exist_ok=True)
    sns.set_theme(style="white")
 
    # ================================================================
    # SECTION A — News dataset: baseline models
    # ================================================================
    print("="*60)
    print(" SECTION A: NEWS DATASET — BASELINE MODELS ")
    print("="*60)
 
    print("\n--- Step A1: Loading News Dataset ---")
    news_df = pd.read_csv(NEWS_PATH)
    news_df = news_df.dropna(subset=['cleaned_headline'])
    y_news  = news_df['is_sarcastic'].values
    cv_news = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    print(f"News dataset: {news_df.shape[0]} rows")
 
    print("\n--- Step A2: Preparing Feature Sets ---")
    X_news_eng = news_df[ENGINEERED_FEATURES].values
 
    tfidf_uni  = TfidfVectorizer(ngram_range=(1, 1),
                                  max_features=10000, sublinear_tf=True)
    tfidf_bi   = TfidfVectorizer(ngram_range=(2, 2),
                                  max_features=10000, sublinear_tf=True)
    tfidf_comb = TfidfVectorizer(ngram_range=(1, 2),
                                  max_features=10000, sublinear_tf=True)
 
    X_news_uni    = tfidf_uni.fit_transform(news_df['cleaned_headline'])
    X_news_bi     = tfidf_bi.fit_transform(news_df['cleaned_headline'])
    X_news_comb   = tfidf_comb.fit_transform(news_df['cleaned_headline'])
    X_news_merged = sp.hstack([X_news_uni, sp.csr_matrix(X_news_eng)])
 
    print("\n--- Step A3: Running Baseline Experiments ---")
    experiments_a = [
        ('Engineered Only', 'Logistic Regression', X_news_eng,
         Pipeline([('scaler', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE))])),
        ('Engineered Only', 'SVM', X_news_eng,
         Pipeline([('scaler', StandardScaler()),
                   ('clf', LinearSVC(max_iter=2000,
                                     random_state=RANDOM_STATE))])),
        ('TF-IDF Unigram', 'Logistic Regression', X_news_uni,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE))])),
        ('TF-IDF Unigram', 'SVM', X_news_uni,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                     random_state=RANDOM_STATE))])),
        ('TF-IDF + Engineered', 'Logistic Regression', X_news_merged,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE))])),
        ('TF-IDF + Engineered', 'SVM', X_news_merged,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                     random_state=RANDOM_STATE))])),
 
        # Random Forest — tree-based, no StandardScaler needed
        ('Engineered Only', 'Random Forest', X_news_eng,
         Pipeline([('clf', RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE))])),
        ('TF-IDF Unigram', 'Random Forest', X_news_uni,
         Pipeline([('clf', RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE))])),
        ('TF-IDF + Engineered', 'Random Forest', X_news_merged,
         Pipeline([('clf', RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE))])),
 
        ('TF-IDF Bigram', 'Logistic Regression', X_news_bi,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE))])),
        ('TF-IDF Bigram', 'SVM', X_news_bi,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                     random_state=RANDOM_STATE))])),
        ('TF-IDF Bigram', 'Random Forest', X_news_bi,
         Pipeline([('clf', RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE))])),
        ('TF-IDF Combined (1-2)', 'Logistic Regression', X_news_comb,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                              random_state=RANDOM_STATE))])),
        ('TF-IDF Combined (1-2)', 'SVM', X_news_comb,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                     random_state=RANDOM_STATE))])),
        ('TF-IDF Combined (1-2)', 'Random Forest', X_news_comb,
         Pipeline([('clf', RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE))])),
    ]
 
    results_a = []
    for feat_name, model_name, X, pipeline in experiments_a:
        print(f"  Running: {feat_name} | {model_name} ...")
        scores = get_cv_scores(pipeline, X, y_news, cv_news)
        results_a.append({
            'Feature Set': feat_name, 'Model': model_name,
            'Accuracy':  round(scores['Accuracy'],  3),
            'Precision': round(scores['Precision'], 3),
            'Recall':    round(scores['Recall'],    3),
            'F1':        round(scores['F1'],        3),
            'F1 Std':    round(scores['F1_std'],    3)
        })
        print(f"    → F1: {scores['F1']:.3f} (±{scores['F1_std']:.3f})")
 
    results_a_df = pd.DataFrame(results_a)
    results_a_df.to_csv(
        os.path.join(TABLES_PATH, 'news_baseline_results.csv'), index=False)
    print("\n--- Section A Full Results ---")
    print(results_a_df.to_string(index=False))
 
    f1_news_tfidf = float(results_a_df[
        (results_a_df['Feature Set'] == 'TF-IDF Unigram') &
        (results_a_df['Model'] == 'Logistic Regression')]['F1'].values[0])
    f1_news_eng   = float(results_a_df[
        (results_a_df['Feature Set'] == 'Engineered Only') &
        (results_a_df['Model'] == 'Logistic Regression')]['F1'].values[0])
    f1_news_best  = results_a_df['F1'].max()
    best_news_row = results_a_df.loc[results_a_df['F1'].idxmax()]
    print(f"\nBest news model: {best_news_row['Feature Set']} | "
          f"{best_news_row['Model']} → F1: {f1_news_best:.3f}")
 
    # ── Section A: Single comprehensive overview plot ──────────────
    # One plot showing all 5 feature sets × 3 models.
    # Replaces separate news_baseline, rq1, and rq3 plots.
    all_feats = [
        'Engineered Only',
        'TF-IDF Unigram',
        'TF-IDF + Engineered',
        'TF-IDF Bigram',
        'TF-IDF Combined (1-2)'
    ]
    feat_labels = [
        'Stylistic Features\n(punctuation, length)',
        'Word Frequency\n(single words)',
        'Word Freq.\n+ Stylistic Features',
        'Word Pairs Only\n(bigrams)',
        'Words & Pairs\n(unigram + bigram)'
    ]
 
    fig, ax = plt.subplots(figsize=(15, 6))
    x   = np.arange(len(all_feats))
    w   = 0.22  # 3 bars × 0.22 = 0.66 per group
 
    for i, (model, color, lbl) in enumerate([
        ('Logistic Regression', COLOR_LR,  'LR'),
        ('SVM',                 COLOR_SVM, 'SVM'),
        ('Random Forest',       COLOR_RF,  'RF')
    ]):
        sub    = results_a_df[results_a_df['Model'] == model].set_index('Feature Set')
        vals   = [sub.loc[f, 'F1']     if f in sub.index else 0 for f in all_feats]
        errs   = [sub.loc[f, 'F1 Std'] if f in sub.index else 0 for f in all_feats]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w,
                        yerr=errs, capsize=3,
                        label=lbl, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.012, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')
 
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_ylabel('F1 Score (5-Fold Cross-Validation)', fontsize=11)
    ax.set_xlabel('Feature Set', fontsize=11, labelpad=12)
    ax.set_title(f'Section A — News Headlines Only | 5 Feature Sets × 3 Models\n'
                 f'F1 Score (5-Fold Cross-Validation) | Logistic Regression, Linear SVM, Random Forest',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.legend(title='Model', fontsize=10, loc='upper left',
              bbox_to_anchor=(0.0, 1.0))
    ax.axhline(y=0.5, color='#cccccc', linestyle='--', linewidth=1,
               label='Random baseline')
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'SectionA_News_F1_All_Feature_Sets.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/section_a_overview.png")
 
 
    # ================================================================
    # SECTION B — Source leakage detection
    # ================================================================
    print("\n" + "="*60)
    print(" SECTION B: SOURCE LEAKAGE DETECTION ")
    print("="*60)
 
    leak_vec = TfidfVectorizer(ngram_range=(1, 1),
                                max_features=10000, sublinear_tf=True)
    X_leak   = leak_vec.fit_transform(news_df['cleaned_headline'])
    leak_clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    leak_clf.fit(X_leak, y_news)
 
    feature_names = leak_vec.get_feature_names_out()
    coefficients  = leak_clf.coef_[0]
    coef_df       = pd.DataFrame({'token': feature_names,
                                  'coefficient': coefficients})
    top_sarc   = coef_df.nlargest(20, 'coefficient')
    top_normal = coef_df.nsmallest(20, 'coefficient')
 
    print("\nTop 20 tokens → Sarcastic:")
    print(top_sarc[['token', 'coefficient']].to_string(index=False))
    print("\nTop 20 tokens → Normal:")
    print(top_normal[['token', 'coefficient']].to_string(index=False))
 
    top_sarc.to_csv(
        os.path.join(TABLES_PATH, 'leakage_tokens_sarcastic.csv'), index=False)
    top_normal.to_csv(
        os.path.join(TABLES_PATH, 'leakage_tokens_normal.csv'), index=False)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, data, color, title in zip(
        axes,
        [top_sarc.head(15), top_normal.head(15)],
        [COLOR_SARC, COLOR_NORMAL],
        ['Top 15 Tokens → Predicted: Sarcastic',
         'Top 15 Tokens → Predicted: Normal']
    ):
        ds   = data.sort_values('coefficient', ascending=True)
        vals = ds['coefficient'].abs()
        bars = ax.barh(ds['token'], vals, color=color, alpha=0.85,
                       edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(val + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}',
                    va='center', fontsize=8, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Logistic Regression Coefficient Magnitude', fontsize=10)
        clean_ax(ax)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
 
    fig.suptitle('Section B — Source Leakage Detection\n'
                 'Logistic Regression | Word Frequency (Single Words) — Top tokens show the model learned source style, not sarcasm',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'SectionB_Leakage_Top_Influential_Words.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/leakage_tokens.png")
 
    # ================================================================
    # SECTION C — Cross-dataset validation
    # ================================================================
    print("\n" + "="*60)
    print(" SECTION C: CROSS-DATASET VALIDATION (LEAKAGE TEST) ")
    print("="*60)
 
    if not os.path.exists(GEN_PATH):
        print(f"[WARNING] {GEN_PATH} not found. Run 05_preprocess_gen.py first.")
        return
 
    gen_df    = pd.read_csv(GEN_PATH)
    gen_df    = gen_df.dropna(subset=['cleaned_headline'])
    y_gen     = gen_df['is_sarcastic'].values
    X_gen_eng = gen_df[ENGINEERED_FEATURES].values
    print(f"GEN dataset: {gen_df.shape[0]} rows")
 
    # Scenario 1: Word Frequency (Single Words)
    print("\nScenario 1: Word Frequency (Single Words)...")
    vec_c1 = TfidfVectorizer(ngram_range=(1, 1),
                              max_features=10000, sublinear_tf=True)
    clf_c1 = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf_c1.fit(vec_c1.fit_transform(news_df['cleaned_headline']), y_news)
    y_pred_c1 = clf_c1.predict(vec_c1.transform(gen_df['cleaned_headline']))
    f1_c1     = f1_score(y_gen, y_pred_c1, zero_division=0)
    print(f"  F1 on GEN: {f1_c1:.3f}")
    print(classification_report(y_gen, y_pred_c1,
                                target_names=['Normal', 'Sarcastic']))
 
    # Scenario 2: Stylistic Features
    print("Scenario 2: Stylistic Features (punctuation, word length)...")
    scaler_c2 = StandardScaler()
    clf_c2    = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf_c2.fit(scaler_c2.fit_transform(
        news_df[ENGINEERED_FEATURES].values), y_news)
    y_pred_c2 = clf_c2.predict(scaler_c2.transform(X_gen_eng))
    f1_c2     = f1_score(y_gen, y_pred_c2, zero_division=0)
    print(f"  F1 on GEN: {f1_c2:.3f}")
    print(classification_report(y_gen, y_pred_c2,
                                target_names=['Normal', 'Sarcastic']))
 
    # Scenario 3: Word Frequency + Stylistic Features (best news model)
    print("Scenario 3: Word Frequency + Stylistic Features (best news model)...")
    vec_c3    = TfidfVectorizer(ngram_range=(1, 1),
                                 max_features=10000, sublinear_tf=True)
    scaler_c3 = StandardScaler()
    X_tr_c3   = sp.hstack([
        vec_c3.fit_transform(news_df['cleaned_headline']),
        sp.csr_matrix(scaler_c3.fit_transform(
            news_df[ENGINEERED_FEATURES].values))
    ])
    clf_c3 = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf_c3.fit(X_tr_c3, y_news)
    X_te_c3 = sp.hstack([
        vec_c3.transform(gen_df['cleaned_headline']),
        sp.csr_matrix(scaler_c3.transform(X_gen_eng))
    ])
    y_pred_c3 = clf_c3.predict(X_te_c3)
    f1_c3     = f1_score(y_gen, y_pred_c3, zero_division=0)
    print(f"  F1 on GEN: {f1_c3:.3f}")
    print(classification_report(y_gen, y_pred_c3,
                                target_names=['Normal', 'Sarcastic']))
 
    # Scenario 4: Word Pairs Only (Bigrams)
    print("Scenario 4: Word Pairs Only (Bigrams)...")
    vec_c4 = TfidfVectorizer(ngram_range=(2, 2),
                              max_features=10000, sublinear_tf=True)
    clf_c4 = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf_c4.fit(vec_c4.fit_transform(news_df['cleaned_headline']), y_news)
    y_pred_c4 = clf_c4.predict(vec_c4.transform(gen_df['cleaned_headline']))
    f1_c4     = f1_score(y_gen, y_pred_c4, zero_division=0)
    f1_news_bigram = float(results_a_df[
        (results_a_df['Feature Set'] == 'TF-IDF Bigram') &
        (results_a_df['Model'] == 'Logistic Regression')]['F1'].values[0])
    print(f"  F1 on GEN: {f1_c4:.3f}")
    print(classification_report(y_gen, y_pred_c4,
                                target_names=['Normal', 'Sarcastic']))
 
    # Scenario 5: Words & Pairs (Unigram + Bigram)
    print("Scenario 5: Words & Pairs (Unigram + Bigram)...")
    vec_c5 = TfidfVectorizer(ngram_range=(1, 2),
                              max_features=10000, sublinear_tf=True)
    clf_c5 = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    clf_c5.fit(vec_c5.fit_transform(news_df['cleaned_headline']), y_news)
    y_pred_c5 = clf_c5.predict(vec_c5.transform(gen_df['cleaned_headline']))
    f1_c5     = f1_score(y_gen, y_pred_c5, zero_division=0)
    f1_news_comb12 = float(results_a_df[
        (results_a_df['Feature Set'] == 'TF-IDF Combined (1-2)') &
        (results_a_df['Model'] == 'Logistic Regression')]['F1'].values[0])
    print(f"  F1 on GEN: {f1_c5:.3f}")
    print(classification_report(y_gen, y_pred_c5,
                                target_names=['Normal', 'Sarcastic']))
 
    cross_summary = pd.DataFrame([
        {'Scenario':   'Word Frequency (Single Words)',
         'News CV F1': f1_news_tfidf,  'GEN F1': round(f1_c1, 3),
         'F1 Drop':    round(f1_news_tfidf - f1_c1, 3)},
        {'Scenario':   'Stylistic Features',
         'News CV F1': f1_news_eng,    'GEN F1': round(f1_c2, 3),
         'F1 Drop':    round(f1_news_eng - f1_c2, 3)},
        {'Scenario':   'Word Freq. + Stylistic Features',
         'News CV F1': f1_news_best,   'GEN F1': round(f1_c3, 3),
         'F1 Drop':    round(f1_news_best - f1_c3, 3)},
        {'Scenario':   'Word Pairs Only',
         'News CV F1': f1_news_bigram, 'GEN F1': round(f1_c4, 3),
         'F1 Drop':    round(f1_news_bigram - f1_c4, 3)},
        {'Scenario':   'Words & Pairs',
         'News CV F1': f1_news_comb12, 'GEN F1': round(f1_c5, 3),
         'F1 Drop':    round(f1_news_comb12 - f1_c5, 3)},
    ])
    print("\n--- Cross-Dataset Summary ---")
    print(cross_summary.to_string(index=False))
    cross_summary.to_csv(
        os.path.join(TABLES_PATH, 'cross_dataset_results.csv'), index=False)
 
    # Confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    COLOR_CORRECT   = '#2171b5'   # professional blue for correct predictions
    COLOR_INCORRECT = '#f0f0f0'   # light grey for incorrect predictions
    for ax, y_pred, title in zip(
        axes,
        [y_pred_c1, y_pred_c2, y_pred_c3, y_pred_c4, y_pred_c5],
        ['Word Freq. (single words)', 'Stylistic Features',
         'Word Freq. + Stylistic', 'Word Pairs Only',
         'Words & Pairs']
    ):
        cm_i = confusion_matrix(y_gen, y_pred)
        for i in range(2):
            for j in range(2):
                is_correct = (i == j)
                bg  = COLOR_CORRECT if is_correct else COLOR_INCORRECT
                tc  = 'white' if is_correct else '#222222'
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=bg, edgecolor='#cccccc', linewidth=1))
                ax.text(j + 0.5, i + 0.5, str(cm_i[i, j]),
                        ha='center', va='center',
                        fontsize=16, fontweight='bold', color=tc)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Normal', 'Sarcastic'], fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Normal', 'Sarcastic'], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    fig.suptitle('Section C — Logistic Regression: Trained on News Headlines, Tested on Forum Comments\n'
                 'Each panel shows a different feature set — diagonal (blue) = correct predictions',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH,
                             'SectionC_Leakage_Confusion_Matrices.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/cross_dataset_confusion_matrix.png")
 
    # F1 drop chart
    scenarios  = ['Word Freq.\n(single words)', 'Stylistic\nFeatures',
                  'Word Freq.\n+ Stylistic', 'Word Pairs\nOnly',
                  'Words\n& Pairs']
    news_vals  = [f1_news_tfidf, f1_news_eng, f1_news_best,
                  f1_news_bigram, f1_news_comb12]
    gen_vals   = [f1_c1, f1_c2, f1_c3, f1_c4, f1_c5]
 
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, scenario, f1_in, f1_cross in zip(
            axes, scenarios, news_vals, gen_vals):
        bars = ax.bar(['News Headlines\n(training domain)',
                       'Forum Comments\n(unseen domain)'],
                      [f1_in, f1_cross],
                      color=['#4393c3', '#e08214'],
                      alpha=0.88, edgecolor='white', width=0.5)
        for bar, val in zip(bars, [f1_in, f1_cross]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.012, f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_title(scenario.replace('\n', ' '),
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('F1 Score (Test Set)', fontsize=11)
        ax.set_ylim(0, 1.05)
        drop = f1_in - f1_cross
        ax.text(0.5, 0.85, f'Drop: {drop:.3f}',
                transform=ax.transAxes, ha='center',
                fontsize=11, color='#d62728', fontweight='bold')
        clean_ax(ax, remove_left=True)
        ax.set_yticks([])
 
    fig.suptitle('Section C — Source Leakage Test: Logistic Regression\n'
                 'Trained on News Headlines → Tested on Forum Comments (Test Set F1)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH,
                             'SectionC_Leakage_F1_Drop_News_vs_Forum.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/cross_dataset_f1_comparison.png")
 
    # RQ1 cross-domain insight
    rq1_crossdomain = pd.DataFrame([
        {'Feature': 'Word Frequency (Single Words)', 'In-Domain': f1_news_tfidf,
         'Cross-Domain': f1_c1},
        {'Feature': 'Stylistic Features', 'In-Domain': f1_news_eng,
         'Cross-Domain': f1_c2},
    ])
    rq1_crossdomain.to_csv(
        os.path.join(TABLES_PATH, 'rq1_crossdomain_insight.csv'), index=False)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels    = ['TF-IDF Unigram', 'Engineered Only']
    x         = np.arange(2)
 
    for ax, (domain, vals, subtitle) in zip(axes, [
        ('News Headlines\n(5-Fold CV)', [f1_news_tfidf, f1_news_eng],
         'Word Frequency wins'),
        ('Forum Comments\n(unseen domain)', [f1_c1, f1_c2],
         'Stylistic Features win'),
    ]):
        bars = ax.bar(x, vals, color=['#4393c3', '#59a14f'],
                      alpha=0.88, edgecolor='white', width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.012, f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['Word Frequency\n(single words)',
                             'Stylistic Features\n(punctuation, length)'],
                            fontsize=11)
        ax.set_ylabel('F1 Score (Test Set)', fontsize=11)
        ax.set_title(f'{domain}', fontweight='bold', fontsize=11)
        ax.text(0.5, -0.18, f'→ {subtitle}',
                transform=ax.transAxes, ha='center',
                fontsize=10, style='italic', color='#555555')
        ax.set_ylim(0, 1.0)
        clean_ax(ax, remove_left=True)
        ax.set_yticks([])
 
    fig.suptitle(
        'RQ1 — Does Feature Type Generalise Across Domains? (Logistic Regression, Test Set)\n'
        'Word Frequency wins on News Headlines — Stylistic Features win on Forum Comments',
        fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'RQ1_Cross_Domain_Generalizability.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/rq1_crossdomain_insight.png")
 
    # ================================================================
    # SECTION D — Combined dataset: domain-independent model
    # ================================================================
    print("\n" + "="*60)
    print(" SECTION D: COMBINED DATASET — DOMAIN-INDEPENDENT MODEL ")
    print("="*60)
 
    news_sub = news_df[['cleaned_headline'] + ENGINEERED_FEATURES +
                        ['is_sarcastic']].copy()
    gen_sub  = gen_df[['cleaned_headline'] + ENGINEERED_FEATURES +
                       ['is_sarcastic']].copy()
    news_sub['source'] = 'news'
    gen_sub['source']  = 'gen'
    combined_df = pd.concat(
        [news_sub, gen_sub], ignore_index=True
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
 
    y_combined = combined_df['is_sarcastic'].values
    print(f"\nCombined dataset: {combined_df.shape[0]} rows")
    print(f"  News : {len(news_sub)} rows")
    print(f"  GEN  : {len(gen_sub)} rows")
    cd = pd.Series(y_combined).value_counts()
    print(f"  Class 0 (Normal)   : {cd[0]} ({cd[0]/len(y_combined)*100:.1f}%)")
    print(f"  Class 1 (Sarcastic): {cd[1]} ({cd[1]/len(y_combined)*100:.1f}%)")
 
    train_df, test_df_d = train_test_split(
        combined_df, test_size=TEST_SIZE,
        stratify=y_combined, random_state=RANDOM_STATE)
    y_train = train_df['is_sarcastic'].values
    y_test  = test_df_d['is_sarcastic'].values
    print(f"\nTrain set : {len(train_df)} rows")
    print(f"Test set  : {len(test_df_d)} rows")
 
    scaler_d    = StandardScaler()
    X_train_eng = scaler_d.fit_transform(train_df[ENGINEERED_FEATURES].values)
    X_test_eng  = scaler_d.transform(test_df_d[ENGINEERED_FEATURES].values)
 
    vec_d_uni    = TfidfVectorizer(ngram_range=(1, 1),
                                    max_features=10000, sublinear_tf=True)
    X_train_uni  = vec_d_uni.fit_transform(train_df['cleaned_headline'])
    X_test_uni   = vec_d_uni.transform(test_df_d['cleaned_headline'])
 
    vec_d_bi     = TfidfVectorizer(ngram_range=(2, 2),
                                    max_features=10000, sublinear_tf=True)
    X_train_bi   = vec_d_bi.fit_transform(train_df['cleaned_headline'])
    X_test_bi    = vec_d_bi.transform(test_df_d['cleaned_headline'])
 
    vec_d_comb   = TfidfVectorizer(ngram_range=(1, 2),
                                    max_features=10000, sublinear_tf=True)
    X_train_comb = vec_d_comb.fit_transform(train_df['cleaned_headline'])
    X_test_comb  = vec_d_comb.transform(test_df_d['cleaned_headline'])
 
    X_train_merged = sp.hstack([X_train_uni, sp.csr_matrix(X_train_eng)])
    X_test_merged  = sp.hstack([X_test_uni,  sp.csr_matrix(X_test_eng)])
 
    print(f"\nRunning Nested {K_FOLDS}-Fold CV on combined train set...")
    outer_cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                random_state=RANDOM_STATE)
    scoring_d = {
        'accuracy':  'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall':    make_scorer(recall_score,    zero_division=0),
        'f1':        make_scorer(f1_score,        zero_division=0)
    }
 
    experiments_d = [
        # ── Engineered Only ───────────────────────────────────
        ('Engineered Only', 'LR', X_train_eng, X_test_eng,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced'))]),
         {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        ('Engineered Only', 'SVM', X_train_eng, X_test_eng,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                      random_state=RANDOM_STATE))]),
         {'clf__C': [0.01, 0.1, 1, 10]}),
        ('Engineered Only', 'RF', X_train_eng, X_test_eng,
         Pipeline([('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                                   class_weight='balanced'))]),
         {'clf__n_estimators': [100, 200]}),
 
        # ── TF-IDF Unigram ─────────────────────────────────────
        ('TF-IDF Unigram', 'LR', X_train_uni, X_test_uni,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced'))]),
         {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        ('TF-IDF Unigram', 'SVM', X_train_uni, X_test_uni,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                      random_state=RANDOM_STATE))]),
         {'clf__C': [0.01, 0.1, 1, 10]}),
        ('TF-IDF Unigram', 'RF', X_train_uni, X_test_uni,
         Pipeline([('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                                   class_weight='balanced'))]),
         {'clf__n_estimators': [100, 200]}),
 
        # ── TF-IDF + Engineered ────────────────────────────────
        ('TF-IDF + Engineered', 'LR', X_train_merged, X_test_merged,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced'))]),
         {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        ('TF-IDF + Engineered', 'SVM', X_train_merged, X_test_merged,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                      random_state=RANDOM_STATE))]),
         {'clf__C': [0.01, 0.1, 1, 10]}),
        ('TF-IDF + Engineered', 'RF', X_train_merged, X_test_merged,
         Pipeline([('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                                   class_weight='balanced'))]),
         {'clf__n_estimators': [100, 200]}),
 
        # ── TF-IDF Bigram ──────────────────────────────────────
        ('TF-IDF Bigram', 'LR', X_train_bi, X_test_bi,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced'))]),
         {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        ('TF-IDF Bigram', 'SVM', X_train_bi, X_test_bi,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                      random_state=RANDOM_STATE))]),
         {'clf__C': [0.01, 0.1, 1, 10]}),
        ('TF-IDF Bigram', 'RF', X_train_bi, X_test_bi,
         Pipeline([('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                                   class_weight='balanced'))]),
         {'clf__n_estimators': [100, 200]}),
 
        # ── TF-IDF Combined (1-2) ──────────────────────────────
        ('TF-IDF Combined (1-2)', 'LR', X_train_comb, X_test_comb,
         Pipeline([('clf', LogisticRegression(max_iter=5000,
                                               random_state=RANDOM_STATE,
                                               class_weight='balanced'))]),
         {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        ('TF-IDF Combined (1-2)', 'SVM', X_train_comb, X_test_comb,
         Pipeline([('clf', LinearSVC(max_iter=2000,
                                      random_state=RANDOM_STATE))]),
         {'clf__C': [0.01, 0.1, 1, 10]}),
        ('TF-IDF Combined (1-2)', 'RF', X_train_comb, X_test_comb,
         Pipeline([('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                                   class_weight='balanced'))]),
         {'clf__n_estimators': [100, 200]}),
    ]
 
    results_d   = []
    best_d_f1   = 0
    best_d_info = {}
 
    for feat_name, model_label, X_tr, X_te, pipeline, param_grid \
            in experiments_d:
        print(f"  Optimizing: {feat_name} | {model_label} ...")
        gs    = GridSearchCV(pipeline, param_grid, cv=inner_cv,
                             scoring='f1', n_jobs=-1, refit=True)
        outer = cross_validate(gs, X_tr, y_train, cv=outer_cv,
                               scoring=scoring_d, n_jobs=-1,
                               return_estimator=True)
        mean_f1 = outer['test_f1'].mean()
        std_f1  = outer['test_f1'].std()
        best_p  = outer['estimator'][0].best_params_
        print(f"    → CV F1: {mean_f1:.3f} (±{std_f1:.3f}) | "
              f"Best params: {best_p}")
        results_d.append({
            'Feature Set': feat_name, 'Model': model_label,
            'CV F1':       round(mean_f1, 3),
            'CV F1 Std':   round(std_f1,  3),
            'Best Params': str(best_p)
        })
        if mean_f1 > best_d_f1:
            best_d_f1   = mean_f1
            best_d_info = {
                'feat':     feat_name,
                'label':    model_label,
                'params':   best_p,
                'X_tr':     X_tr,
                'X_te':     X_te,
                'pipeline': pipeline
            }
 
    results_d_df = pd.DataFrame(results_d)
    results_d_df.to_csv(
        os.path.join(TABLES_PATH, 'combined_cv_results.csv'), index=False)
    print("\n--- Section D Full Results ---")
    print(results_d_df.to_string(index=False))
    print(f"\nBest combined model: {best_d_info['feat']} | "
          f"{best_d_info['label']} → CV F1: {best_d_f1:.3f}")
 
    # Final evaluation on held-out test set
    print("\n--- Final Evaluation on Held-Out Test Set ---")
    best_p    = best_d_info['params']
    param_key = list(best_p.keys())[0]
    param_val = best_p[param_key]
    gs_final  = GridSearchCV(best_d_info['pipeline'],
                              {param_key: [param_val]},
                              cv=inner_cv, scoring='f1',
                              n_jobs=-1, refit=True)
    gs_final.fit(best_d_info['X_tr'], y_train)
    y_pred_test = gs_final.predict(best_d_info['X_te'])
    test_f1     = f1_score(y_test, y_pred_test, zero_division=0)
    test_acc    = (y_pred_test == y_test).mean()
    print(f"  Best model : {best_d_info['feat']} | {best_d_info['label']}")
    print(f"  CV F1      : {best_d_f1:.3f}")
    print(f"  Test F1    : {test_f1:.3f}")
    print(f"  Test Acc   : {test_acc:.3f}")
    print("\nClassification Report (held-out test set):")
    print(classification_report(y_test, y_pred_test,
                                target_names=['Normal', 'Sarcastic']))
 
    # ── Detailed metrics ─────────────────────────────────────
    test_precision_normal   = precision_score(y_test, y_pred_test,
                                              pos_label=0, zero_division=0)
    test_recall_normal      = recall_score(y_test, y_pred_test,
                                           pos_label=0, zero_division=0)
    test_precision_sarcastic = precision_score(y_test, y_pred_test,
                                               pos_label=1, zero_division=0)
    test_recall_sarcastic    = recall_score(y_test, y_pred_test,
                                            pos_label=1, zero_division=0)
 
    metrics_df = pd.DataFrame([
        {'Class': 'Normal',    'Precision': round(test_precision_normal, 3),
         'Recall': round(test_recall_normal, 3),
         'F1':     round(f1_score(y_test, y_pred_test, pos_label=0,
                                  zero_division=0), 3)},
        {'Class': 'Sarcastic', 'Precision': round(test_precision_sarcastic, 3),
         'Recall': round(test_recall_sarcastic, 3),
         'F1':     round(test_f1, 3)},
        {'Class': 'Overall',   'Precision': '—',
         'Recall': '—', 'F1': round(test_f1, 3)},
    ])
    metrics_df['Accuracy'] = ['—', '—', round(test_acc, 3)]
    metrics_df.to_csv(
        os.path.join(TABLES_PATH, 'best_model_metrics.csv'), index=False)
    print("Saved: visuals/tables/best_model_metrics.csv")
 
    plot_confusion_matrix(
        y_test, y_pred_test,
        'Logistic Regression | Word Freq. + Stylistic Features\nConfusion Matrix — News Headlines + Forum Comments (Test Set)',
        os.path.join(FIGURES_PATH, 'SectionD_Best_Model_Confusion_Matrix.png'))
    print("Saved: visuals/figures/combined_confusion_matrix.png")
 
    # ── ROC Curve (LR only — produces probabilities) ──────────
    if best_d_info['label'] == 'LR':
        y_prob = gs_final.predict_proba(best_d_info['X_te'])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score   = roc_auc_score(y_test, y_prob)
        print(f"  AUC Score  : {auc_score:.3f}")
 
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=COLOR_LR, linewidth=2.5,
                label=f'LR — AUC = {auc_score:.3f}')
        ax.plot([0, 1], [0, 1], color='#cccccc', linestyle='--',
                linewidth=1.5, label='Random baseline (AUC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.08, color=COLOR_LR)
        ax.set_xlabel('False Positive Rate (1 — Specificity)', fontsize=11)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
        ax.set_title(
            'ROC Curve — Logistic Regression | Word Freq. + Stylistic Features\n'
            'News Headlines + Forum Comments (Test Set, AUC = Area Under Curve)',
            fontweight='bold', fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        clean_ax(ax)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.annotate(f'AUC = {auc_score:.3f}',
                    xy=(0.6, 0.4), fontsize=13, fontweight='bold',
                    color=COLOR_LR)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'SectionD_Best_Model_ROC_Curve.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print("Saved: visuals/figures/roc_curve.png")
 
        # Save AUC to final summary
        auc_val = auc_score
    else:
        auc_val = None
        print("  ROC/AUC: Not available for SVM (no probability output)")
 
    # ── Domain-specific test evaluation ──────────────────────
    # Split the held-out test set by source (news vs forum).
    # IMPORTANT: these rows were never seen during training or CV.
    print("\n--- Domain-Specific Test Evaluation ---")
 
    # Build best feature matrix for test set subsets
    # (reuse the already-fitted vectorizer and scaler from best model)
    vec_best    = best_d_info['vec']    if 'vec'    in best_d_info else vec_d_uni
    scaler_best = best_d_info['scaler'] if 'scaler' in best_d_info else scaler_d
 
    test_news_df  = test_df_d[test_df_d['source'] == 'news'].copy()
    test_gen_df   = test_df_d[test_df_d['source'] == 'gen'].copy()
    y_test_news   = test_news_df['is_sarcastic'].values
    y_test_gen    = test_gen_df['is_sarcastic'].values
 
    # Build features for each subset using fitted transformers
    X_test_news_uni  = vec_d_uni.transform(test_news_df['cleaned_headline'])
    X_test_gen_uni   = vec_d_uni.transform(test_gen_df['cleaned_headline'])
    X_test_news_eng  = scaler_d.transform(test_news_df[ENGINEERED_FEATURES].values)
    X_test_gen_eng   = scaler_d.transform(test_gen_df[ENGINEERED_FEATURES].values)
    X_test_news_best = sp.hstack([X_test_news_uni,
                                  sp.csr_matrix(X_test_news_eng)])
    X_test_gen_best  = sp.hstack([X_test_gen_uni,
                                  sp.csr_matrix(X_test_gen_eng)])
 
    y_pred_news = gs_final.predict(X_test_news_best)
    y_pred_gen  = gs_final.predict(X_test_gen_best)
 
    f1_test_news_only = f1_score(y_test_news, y_pred_news, zero_division=0)
    f1_test_gen_only  = f1_score(y_test_gen,  y_pred_gen,  zero_division=0)
    acc_test_news     = (y_pred_news == y_test_news).mean()
    acc_test_gen      = (y_pred_gen  == y_test_gen ).mean()
 
    print(f"  News Headlines only  : F1 = {f1_test_news_only:.3f} "
          f"| Accuracy = {acc_test_news:.3f} "
          f"| n = {len(y_test_news)}")
    print(f"  Forum Comments only  : F1 = {f1_test_gen_only:.3f}  "
          f"| Accuracy = {acc_test_gen:.3f}  "
          f"| n = {len(y_test_gen)}")
    print(f"  Combined (all)       : F1 = {test_f1:.3f} "
          f"| Accuracy = {test_acc:.3f} "
          f"| n = {len(y_test)}")
 
    # Save domain-specific results
    domain_results = pd.DataFrame([
        {'Test Set': 'News Headlines (unseen)',
         'F1':       round(f1_test_news_only, 3),
         'Accuracy': round(acc_test_news, 3),
         'N':        len(y_test_news)},
        {'Test Set': 'Forum Comments (unseen)',
         'F1':       round(f1_test_gen_only, 3),
         'Accuracy': round(acc_test_gen, 3),
         'N':        len(y_test_gen)},
        {'Test Set': 'Combined (News + Forum)',
         'F1':       round(test_f1, 3),
         'Accuracy': round(test_acc, 3),
         'N':        len(y_test)},
    ])
    domain_results.to_csv(
        os.path.join(TABLES_PATH, 'domain_specific_test_results.csv'),
        index=False)
    print("Saved: visuals/tables/domain_specific_test_results.csv")
 
    # ── Domain-specific test bar chart ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ['News Headlines\n(unseen test)',
              'Forum Comments\n(unseen test)',
              'Combined\n(News + Forum)']
    values = [f1_test_news_only, f1_test_gen_only, test_f1]
    colors = ['#4393c3', '#e08214', '#2ca02c']
 
    bars = ax.bar(labels, values, color=colors, alpha=0.88,
                  width=0.45, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.012, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=13)
 
    ax.set_ylabel('F1 Score (Test Set — never seen during training)', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('Domain-Specific Test Results — Logistic Regression\n'
                 'Word Frequency + Stylistic Features | All test rows are unseen',
                 fontweight='bold', fontsize=13)
    ax.axhline(y=0.5, color='#cccccc', linestyle='--', linewidth=1.2)
    ax.text(2.4, 0.52, 'Random baseline', fontsize=9,
            color='#999999', va='bottom')
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH,
                             'SectionD_Domain_Specific_Test_Results.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/SectionD_Domain_Specific_Test_Results.png")
 
    # Section D combined comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    feat_order = ['Engineered Only', 'TF-IDF Unigram',
                  'TF-IDF + Engineered', 'TF-IDF Bigram',
                  'TF-IDF Combined (1-2)']
    x   = np.arange(len(feat_order))
    w_d = 0.22  # 3 models × 0.22 = 0.66 total per group
 
    for i, (model, color, lbl) in enumerate([
        ('LR',  COLOR_LR,  'LR'),
        ('SVM', COLOR_SVM, 'SVM'),
        ('RF',  COLOR_RF,  'RF')
    ]):
        sub  = results_d_df[results_d_df['Model'] == model].set_index('Feature Set')
        vals = [sub.loc[f, 'CV F1'] if f in sub.index else 0 for f in feat_order]
        errs = [sub.loc[f, 'CV F1 Std'] if f in sub.index else 0 for f in feat_order]
        offset = (i - 1) * w_d
        bars = ax.bar(x + offset, vals, w_d,
                      yerr=errs, capsize=3,
                      label=lbl, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.012, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')
 
    ax.set_xticks(x)
    ax.set_xticklabels(['Stylistic\nFeatures',
                        'Word Frequency\n(single words)',
                        'Word Freq.\n+ Stylistic',
                        'Word Pairs\nOnly',
                        'Words\n& Pairs'],
                       fontsize=9)
    ax.set_ylabel('F1 Score (Cross-Validation)', fontsize=11)
    ax.set_title(f'Section D — News Headlines + Forum Comments | 5 Feature Sets × 3 Models\n'
                 f'F1 Score (Nested Cross-Validation) | Logistic Regression, Linear SVM, Random Forest',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0.2, 1.0)
    ax.legend(title='Model', fontsize=10, loc='upper right')
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'SectionD_Combined_F1_All_Feature_Sets.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/combined_model_comparison.png")
    plt.close()
    print("Saved: visuals/figures/combined_model_comparison.png")
 
    # RQ1 combined comparison
    rq1_c_feats = ['Engineered Only', 'TF-IDF Unigram', 'TF-IDF + Engineered']
    rq1_c_df    = results_d_df[
        results_d_df['Feature Set'].isin(rq1_c_feats)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rq1_c_feats))
    for i, (model, color, label) in enumerate([
        ('LR',  COLOR_LR,  'LR'),
        ('SVM', COLOR_SVM, 'SVM'),
        ('RF',  COLOR_RF,  'RF')
    ]):
        sub  = rq1_c_df[rq1_c_df['Model'] == model].set_index('Feature Set')
        vals = [sub.loc[f, 'CV F1'] if f in sub.index else 0
                for f in rq1_c_feats]
        ww   = 0.25
        offset = (i - 1) * ww
        bars = ax.bar(x + offset, vals, ww,
                      label=label,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.01, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Stylistic Features\n(punctuation, length)',
                        'Word Frequency\n(single words)',
                        'Word Freq.\n+ Stylistic Features'], fontsize=11)
    ax.set_ylabel('F1 Score (Cross-Validation)', fontsize=11)
    ax.set_title('RQ1 — Do Stylistic or Lexical Features Better Detect Sarcasm?\n'
                 'News Headlines + Forum Comments | LR, Linear SVM, RF | F1 (Nested Cross-Validation)',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0.2, 1.0)
    ax.legend(title='Model', fontsize=10, loc='upper left')
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'RQ1_Combined_Feature_Comparison.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/rq1_combined_comparison.png")
 
    # RQ3 combined comparison
    rq3_c_feats = ['TF-IDF Unigram', 'TF-IDF Bigram', 'TF-IDF Combined (1-2)']
    rq3_c_df    = results_d_df[
        results_d_df['Feature Set'].isin(rq3_c_feats)].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rq3_c_feats))
    for i, (model, color, label) in enumerate([
        ('LR',  COLOR_LR,  'LR'),
        ('SVM', COLOR_SVM, 'SVM'),
        ('RF',  COLOR_RF,  'RF')
    ]):
        sub  = rq3_c_df[rq3_c_df['Model'] == model].set_index('Feature Set')
        vals = [sub.loc[f, 'CV F1'] if f in sub.index else 0
                for f in rq3_c_feats]
        ww   = 0.25
        offset = (i - 1) * ww
        bars = ax.bar(x + offset, vals, ww,
                      label=label,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.01, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Word Frequency\n(single words)',
                        'Word Pairs Only\n(bigrams)',
                        'Words & Pairs\n(unigram + bigram)'], fontsize=11)
    ax.set_ylabel('F1 Score (Cross-Validation)', fontsize=11)
    ax.set_title('RQ3 — Do Word Pairs Improve Sarcasm Detection Beyond Single Words?\n'
                 'News Headlines + Forum Comments | LR, Linear SVM, RF | F1 (Nested Cross-Validation)',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0.5, 1.0)
    ax.legend(title='Model', fontsize=10, loc='upper left')
    ax.legend(title='Model', fontsize=10, loc='upper left')
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'RQ3_Combined_Ngram_Comparison.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/rq3_combined_comparison.png")
 
    # ================================================================
    # SECTION E — BALANCED DATASET EXPERIMENT
    # Same best model, but news downsampled to match GEN size (6,520)
    # Stratified sampling ensures ~50/50 sarcastic/normal within news
    # Train/test split done AFTER downsampling — test never seen in training
    # ================================================================
    print("\n" + "="*60)
    print(" SECTION E: BALANCED DATASET EXPERIMENT ")
    print("="*60)
 
    from sklearn.model_selection import train_test_split as tts_bal
 
    # Downsample news with stratification to preserve class ratio
    news_bal, _ = tts_bal(
        news_sub,
        train_size=len(gen_sub),
        stratify=news_sub['is_sarcastic'],
        random_state=RANDOM_STATE
    )
    news_bal['source'] = 'news'
 
    combined_bal = pd.concat(
        [news_bal, gen_sub], ignore_index=True
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
 
    y_bal = combined_bal['is_sarcastic'].values
    bal_cd = pd.Series(y_bal).value_counts()
    print(f"\nBalanced combined dataset: {len(combined_bal)} rows")
    print(f"  News  : {len(news_bal)} rows")
    print(f"  Forum : {len(gen_sub)} rows")
    print(f"  Class 0 (Normal)   : {bal_cd[0]} ({bal_cd[0]/len(y_bal)*100:.1f}%)")
    print(f"  Class 1 (Sarcastic): {bal_cd[1]} ({bal_cd[1]/len(y_bal)*100:.1f}%)")
 
    # Train/test split — test set never seen during training
    train_bal, test_bal = tts_bal(
        combined_bal, test_size=TEST_SIZE,
        stratify=y_bal, random_state=RANDOM_STATE)
    y_train_bal = train_bal['is_sarcastic'].values
    y_test_bal  = test_bal['is_sarcastic'].values
    print(f"\nTrain set : {len(train_bal)} rows")
    print(f"Test set  : {len(test_bal)} rows")
 
    # Build features — fit on train only, transform test
    scaler_bal   = StandardScaler()
    X_tr_bal_eng = scaler_bal.fit_transform(
        train_bal[ENGINEERED_FEATURES].values)
    X_te_bal_eng = scaler_bal.transform(
        test_bal[ENGINEERED_FEATURES].values)
 
    vec_bal = TfidfVectorizer(ngram_range=(1, 1),
                               max_features=10000, sublinear_tf=True)
    X_tr_bal_uni = vec_bal.fit_transform(train_bal['cleaned_headline'])
    X_te_bal_uni = vec_bal.transform(test_bal['cleaned_headline'])
 
    X_tr_bal = sp.hstack([X_tr_bal_uni, sp.csr_matrix(X_tr_bal_eng)])
    X_te_bal = sp.hstack([X_te_bal_uni, sp.csr_matrix(X_te_bal_eng)])
 
    # Use same model type as best model (LR, C=1, class_weight=balanced)
    from sklearn.model_selection import cross_val_score
    lr_bal = LogisticRegression(C=1, max_iter=5000,
                                 random_state=RANDOM_STATE,
                                 class_weight='balanced')
    cv_scores_bal = cross_val_score(
        lr_bal, X_tr_bal, y_train_bal,
        cv=K_FOLDS, scoring='f1')
    bal_cv_f1 = cv_scores_bal.mean()
    print(f"\nCV F1 (balanced, {K_FOLDS}-Fold): {bal_cv_f1:.3f} "
          f"(±{cv_scores_bal.std():.3f})")
 
    lr_bal.fit(X_tr_bal, y_train_bal)
    y_pred_bal    = lr_bal.predict(X_te_bal)
    bal_test_f1   = f1_score(y_test_bal, y_pred_bal, zero_division=0)
    bal_test_acc  = (y_pred_bal == y_test_bal).mean()
    print(f"Test F1  (balanced)        : {bal_test_f1:.3f}")
    print(f"Test Acc (balanced)        : {bal_test_acc:.3f}")
 
    # Domain-specific test for balanced model
    test_bal_news = test_bal[test_bal['source'] == 'news'].copy()
    test_bal_gen  = test_bal[test_bal['source'] == 'gen'].copy()
 
    X_te_bal_news_uni = vec_bal.transform(test_bal_news['cleaned_headline'])
    X_te_bal_news_eng = scaler_bal.transform(
        test_bal_news[ENGINEERED_FEATURES].values)
    X_te_bal_news = sp.hstack([X_te_bal_news_uni,
                                sp.csr_matrix(X_te_bal_news_eng)])
 
    X_te_bal_gen_uni  = vec_bal.transform(test_bal_gen['cleaned_headline'])
    X_te_bal_gen_eng  = scaler_bal.transform(
        test_bal_gen[ENGINEERED_FEATURES].values)
    X_te_bal_gen  = sp.hstack([X_te_bal_gen_uni,
                                sp.csr_matrix(X_te_bal_gen_eng)])
 
    f1_bal_news = f1_score(test_bal_news['is_sarcastic'].values,
                            lr_bal.predict(X_te_bal_news), zero_division=0)
    f1_bal_gen  = f1_score(test_bal_gen['is_sarcastic'].values,
                            lr_bal.predict(X_te_bal_gen),  zero_division=0)
    print(f"  News Headlines only : F1 = {f1_bal_news:.3f} "
          f"| n = {len(test_bal_news)}")
    print(f"  Forum Comments only : F1 = {f1_bal_gen:.3f}  "
          f"| n = {len(test_bal_gen)}")
 
    # Save balanced results
    balanced_results = pd.DataFrame([
        {'Dataset': 'Imbalanced (26,490 news + 6,520 forum)',
         'CV F1':   round(best_d_f1, 3),
         'Test F1': round(test_f1, 3),
         'Test Acc': round(test_acc, 3),
         'News F1': round(f1_test_news_only, 3),
         'Forum F1': round(f1_test_gen_only, 3)},
        {'Dataset': 'Balanced (6,520 news + 6,520 forum)',
         'CV F1':   round(bal_cv_f1, 3),
         'Test F1': round(bal_test_f1, 3),
         'Test Acc': round(bal_test_acc, 3),
         'News F1': round(f1_bal_news, 3),
         'Forum F1': round(f1_bal_gen, 3)},
    ])
    print("\n--- Imbalanced vs Balanced Comparison ---")
    print(balanced_results.to_string(index=False))
    balanced_results.to_csv(
        os.path.join(TABLES_PATH, 'balanced_vs_imbalanced.csv'), index=False)
    print("Saved: visuals/tables/balanced_vs_imbalanced.csv")
 
    # ── Comparison chart: Imbalanced vs Balanced ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Section E — Balanced vs Imbalanced Training Data\n'
                 'Logistic Regression | Word Frequency + Stylistic Features | Test Set F1',
                 fontweight='bold', fontsize=13)
 
    # Left: overall F1 comparison
    ax = axes[0]
    x  = np.arange(2)
    ax.bar(x, [test_f1, bal_test_f1],
           color=['#4393c3', '#59a14f'], alpha=0.88, width=0.4,
           edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(['Imbalanced\n(26,490 news + 6,520 forum)',
                         'Balanced\n(6,520 news + 6,520 forum)'], fontsize=10)
    ax.set_ylabel('F1 Score (Test Set)', fontsize=11)
    ax.set_title('Overall Test F1', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate([test_f1, bal_test_f1]):
        ax.text(i, v + 0.015, f'{v:.3f}',
                ha='center', fontweight='bold', fontsize=13)
    ax.axhline(0.5, color='#cccccc', linestyle='--', linewidth=1)
    clean_ax(ax)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
 
    # Right: domain-specific F1 comparison
    ax2  = axes[1]
    x2   = np.arange(2)
    w2   = 0.3
    imb  = [f1_test_news_only, f1_test_gen_only]
    bal  = [f1_bal_news,       f1_bal_gen]
    b1   = ax2.bar(x2 - w2/2, imb, w2, label='Imbalanced',
                   color='#4393c3', alpha=0.88, edgecolor='white')
    b2   = ax2.bar(x2 + w2/2, bal, w2, label='Balanced',
                   color='#59a14f', alpha=0.88, edgecolor='white')
    for bar, val in [(b, v) for bs, vs in [(b1, imb), (b2, bal)]
                     for b, v in zip(bs, vs)]:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.015, f'{val:.3f}',
                 ha='center', fontweight='bold', fontsize=11)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['News Headlines\n(unseen test)',
                          'Forum Comments\n(unseen test)'], fontsize=10)
    ax2.set_ylabel('F1 Score (Test Set)', fontsize=11)
    ax2.set_title('Domain-Specific Test F1', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=10)
    ax2.axhline(0.5, color='#cccccc', linestyle='--', linewidth=1)
    clean_ax(ax2)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
 
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH,
                             'SectionE_Balanced_vs_Imbalanced.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: visuals/figures/SectionE_Balanced_vs_Imbalanced.png")
 
    # Final summary table
    final_summary = pd.DataFrame([
        {'Stage': 'News baseline (best in-domain)',
         'Model': f"{best_news_row['Feature Set']} | "
                  f"{best_news_row['Model']}",
         'F1':      f1_news_best,
         'AUC':     '—',
         'Accuracy': '—',
         'Note':    f'{K_FOLDS}-Fold CV, news only'},
        {'Stage': 'Leakage confirmed (TF-IDF on GEN)',
         'Model': 'TF-IDF Unigram | LR',
         'F1':      round(f1_c1, 3),
         'AUC':     '—',
         'Accuracy': '—',
         'Note':    'Trained on news, tested on GEN'},
        {'Stage': 'Combined model (CV)',
         'Model': f"{best_d_info['feat']} | {best_d_info['label']}",
         'F1':      best_d_f1,
         'AUC':     '—',
         'Accuracy': '—',
         'Note':    f'Nested {K_FOLDS}-Fold CV, News Headlines + Forum Comments'},
        {'Stage': 'Combined model (test set)',
         'Model': f"{best_d_info['feat']} | {best_d_info['label']}",
         'F1':      round(test_f1, 3),
         'AUC':     round(auc_val, 3) if auc_val else '—',
         'Accuracy': round(test_acc, 3),
         'Note':    '20% held-out test set'},
    ])
    print("\n--- Final Project Summary ---")
    print(final_summary.to_string(index=False))
    final_summary.to_csv(
        os.path.join(TABLES_PATH, 'final_summary.csv'), index=False)
 
    # ----------------------------------------------------------------
    # Output summary
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("SUCCESS! Modeling complete.")
    print("\nSection A — News Baseline (LR vs SVM vs RF):")
    print("  Figures → visuals/figures/section_a_overview.png")
    print("  Tables  → visuals/tables/news_baseline_results.csv")
    print("\nSection B — Leakage Detection:")
    print("  Figures → visuals/figures/leakage_tokens.png")
    print("  Tables  → visuals/tables/leakage_tokens_sarcastic.csv")
    print("          → visuals/tables/leakage_tokens_normal.csv")
    print("\nSection C — Cross-Dataset Validation:")
    print("  Figures → visuals/figures/cross_dataset_confusion_matrix.png")
    print("          → visuals/figures/cross_dataset_f1_comparison.png")
    print("          → visuals/figures/rq1_crossdomain_insight.png")
    print("  Tables  → visuals/tables/cross_dataset_results.csv")
    print("          → visuals/tables/rq1_crossdomain_insight.csv")
    print("\nSection D — Combined Domain-Independent Model:")
    print("  Figures → visuals/figures/combined_model_comparison.png")
    print("          → visuals/figures/combined_confusion_matrix.png")
    print("          → visuals/figures/rq1_combined_comparison.png")
    print("          → visuals/figures/rq3_combined_comparison.png")
    print("  Tables  → visuals/tables/combined_cv_results.csv")
          
    print("\nBest Model — Detailed Evaluation:")
    print("  Figures → visuals/figures/roc_curve.png")
    print("  Tables  → visuals/tables/best_model_metrics.csv")
    print("          → visuals/tables/final_summary.csv")
    print("="*60)
 
if __name__ == "__main__":
    run_modeling()