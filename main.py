# Cancer Cell Type Classification — Gene Expression RNA-Seq (UCI id=401)
# 801 samples, 20531 genes, 5 cancer classes
# Models compared: Random Forest vs Logistic Regression


import matplotlib
matplotlib.use('Agg')

import io
import os
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE    = os.path.join(SCRIPT_DIR, "..", "TCGA-PANCAN-HiSeq-801x20531.tar.gz")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOADING THE DATA
# =============================================================================

print("[1/10] Loading dataset from local archive ...")
with tarfile.open(ARCHIVE, "r:gz") as tf:
    X = pd.read_csv(
        io.TextIOWrapper(tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv")),
        index_col=0,
    )
    y = pd.read_csv(
        io.TextIOWrapper(tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv")),
        index_col=0,
    )["Class"]

print(f"      Dataset shape : {X.shape}")
print(f"      Class distribution:\n{y.value_counts().to_string()}\n")


# =============================================================================
# 2. PREPROCESS OF DATA
# =============================================================================

print("[2/10] Preprocessing ...")

nan_count = int(X.isna().sum().sum())
if nan_count > 0:
    print(f"      Found {nan_count} NaN values — filling with column medians")
    X = X.fillna(X.median(numeric_only=True))
else:
    print("      No missing values found")

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"      Classes: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc,
)
print(f"      Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")


# =============================================================================
# 3. FEATURE SELECTION  (20531 -> ~19959 via variance -> 50 via ANOVA F-test)
# =============================================================================

print("[3/10] Feature selection ...")

vt = VarianceThreshold(threshold=0.01)
X_train_vt = vt.fit_transform(X_train)
X_test_vt  = vt.transform(X_test)
print(f"      After VarianceThreshold : {X_train_vt.shape[1]} features")

skb = SelectKBest(score_func=f_classif, k=50)
X_train_sel = skb.fit_transform(X_train_vt, y_train)
X_test_sel  = skb.transform(X_test_vt)

gene_names_after_vt = np.array(X_train.columns)[vt.get_support()]
selected_gene_names = gene_names_after_vt[skb.get_support()]

print(f"      After SelectKBest(k=50) : {X_train_sel.shape[1]} features")
print(f"      Top 5 selected genes    : {list(selected_gene_names[:5])}\n")


# =============================================================================
# 4. RANDOM FOREST — TRAIN
# =============================================================================

print("[4/10] Training RandomForestClassifier (100 trees) ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sel, y_train)
print("      Training complete\n")


# =============================================================================
# 5. RANDOM FOREST — CROSS-VALIDATION
# =============================================================================

print("[5/10] Random Forest — 5-fold stratified cross-validation ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf, X_train_sel, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"      Fold scores : {[round(s, 4) for s in rf_cv]}")
print(f"      CV Accuracy : {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}\n")


# =============================================================================
# 6. RANDOM FOREST — TEST EVALUATION + CONFUSION MATRIX
# =============================================================================

print("[6/10] Random Forest — evaluating on test set ...")
rf_pred = rf.predict(X_test_sel)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average='weighted')
print(f"      Test Accuracy : {rf_acc:.4f}")
print(f"      Weighted F1   : {rf_f1:.4f}")

cm_rf = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix — Random Forest', fontsize=14, pad=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
cm_rf_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_rf.png')
plt.savefig(cm_rf_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {cm_rf_path}\n")


# =============================================================================
# 7. RANDOM FOREST — FEATURE IMPORTANCE CHART (top 20 genes)
# =============================================================================

print("[7/10] Plotting Random Forest feature importances ...")
importances = rf.feature_importances_
indices     = np.argsort(importances)[::-1]
top_n       = 20
top_names   = selected_gene_names[indices[:top_n]]
top_values  = importances[indices[:top_n]]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(top_n), top_values, color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_xticks(range(top_n))
ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=9)
ax.set_title('Top 20 Most Important Genes (Random Forest — Mean Decrease in Impurity)',
             fontsize=13, pad=12)
ax.set_xlabel('Gene', fontsize=11)
ax.set_ylabel('Feature Importance (MDI)', fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {fi_path}\n")


# =============================================================================
# 8. LOGISTIC REGRESSION — SCALE + TRAIN
# =============================================================================

print("[8/10] Logistic Regression — scaling features and training ...")

# LR is sensitive to feature scale; scale with StandardScaler fit on train only
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sel)
X_test_sc   = scaler.transform(X_test_sel)

lr = LogisticRegression(
    max_iter=2000,   # generous limit for 5-class convergence on 50 features
    solver='lbfgs',  # uses multinomial softmax automatically for >2 classes
    random_state=42,
)
lr.fit(X_train_sc, y_train)
print("      Training complete\n")


# =============================================================================
# 9. LOGISTIC REGRESSION — CROSS-VALIDATION + TEST EVALUATION + CONFUSION MATRIX
# =============================================================================

print("[9/10] Logistic Regression — cross-validation and test evaluation ...")
lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"      Fold scores : {[round(s, 4) for s in lr_cv]}")
print(f"      CV Accuracy : {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}")

lr_pred = lr.predict(X_test_sc)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_f1   = f1_score(y_test, lr_pred, average='weighted')
print(f"      Test Accuracy : {lr_acc:.4f}")
print(f"      Weighted F1   : {lr_f1:.4f}")

cm_lr = confusion_matrix(y_test, lr_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix — Logistic Regression', fontsize=14, pad=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
cm_lr_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_lr.png')
plt.savefig(cm_lr_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {cm_lr_path}\n")


# =============================================================================
# 10. MODEL COMPARISON CHARTS
# =============================================================================

print("[10/10] Generating model comparison charts ...")

# --- 10a: Side-by-side confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cm_data, title, cmap in zip(
    axes,
    [cm_rf, cm_lr],
    ['Random Forest', 'Logistic Regression'],
    ['Blues', 'Oranges'],
):
    sns.heatmap(cm_data, annot=True, fmt='d', cmap=cmap,
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax)
    ax.set_title(f'Confusion Matrix\n{title}', fontsize=13, pad=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
plt.suptitle('Cancer Type Classification — Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
cm_cmp_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_comparison.png')
plt.savefig(cm_cmp_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {cm_cmp_path}")

# --- 10b: Grouped bar chart — CV Accuracy / Test Accuracy / Weighted F1 ---
metrics      = ['CV Accuracy', 'Test Accuracy', 'Weighted F1']
rf_scores    = [rf_cv.mean(),  rf_acc,          rf_f1]
lr_scores    = [lr_cv.mean(),  lr_acc,          lr_f1]

x     = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 6))
bars_rf = ax.bar(x - width / 2, rf_scores, width, label='Random Forest',
                 color='steelblue', edgecolor='white', linewidth=0.6)
bars_lr = ax.bar(x + width / 2, lr_scores, width, label='Logistic Regression',
                 color='darkorange', edgecolor='white', linewidth=0.6)

# Annotate each bar with its value
for bar in bars_rf:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
for bar in bars_lr:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0.90, 1.02)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Random Forest vs Logistic Regression — Performance Metrics', fontsize=13, pad=12)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
bar_cmp_path = os.path.join(OUTPUT_DIR, 'model_comparison_metrics.png')
plt.savefig(bar_cmp_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {bar_cmp_path}\n")


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 62)
print("  MODEL COMPARISON SUMMARY")
print("=" * 62)
print(f"  {'Metric':<22} {'Random Forest':>16} {'Logistic Reg':>14}")
print("-" * 62)
print(f"  {'CV Accuracy':<22} {rf_cv.mean():>14.4f}   {lr_cv.mean():>12.4f}")
print(f"  {'CV Std Dev':<22} {rf_cv.std():>14.4f}   {lr_cv.std():>12.4f}")
print(f"  {'Test Accuracy':<22} {rf_acc:>14.4f}   {lr_acc:>12.4f}")
print(f"  {'Weighted F1':<22} {rf_f1:>14.4f}   {lr_f1:>12.4f}")
print("=" * 62)
winner = 'Random Forest' if rf_acc >= lr_acc else 'Logistic Regression'
print(f"  Best test accuracy : {winner}")
print(f"  Outputs dir        : {OUTPUT_DIR}")
print("=" * 62)
