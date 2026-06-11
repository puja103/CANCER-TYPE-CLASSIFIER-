# =============================================================================
# Cancer Cell Type Classification — Gene Expression RNA-Seq (UCI id=401)
# Dataset: TCGA PanCan HiSeq, 801 samples, 20531 genes, 5 cancer classes
# =============================================================================

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before importing pyplot

import io
import os
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE    = os.path.join(SCRIPT_DIR, "..", "TCGA-PANCAN-HiSeq-801x20531.tar.gz")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("[1/7] Loading dataset from local archive ...")
with tarfile.open(ARCHIVE, "r:gz") as tf:
    X = pd.read_csv(
        io.TextIOWrapper(tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv")),
        index_col=0,  # first column is sample ID
    )
    y = pd.read_csv(
        io.TextIOWrapper(tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv")),
        index_col=0,
    )["Class"]

print(f"      Dataset shape : {X.shape}")
print(f"      Class distribution:\n{y.value_counts().to_string()}\n")


# =============================================================================
# 2. PREPROCESS
# =============================================================================

print("[2/7] Preprocessing ...")

# Fill any NaN values with column median (defensive — dataset is clean)
nan_count = int(X.isna().sum().sum())
if nan_count > 0:
    print(f"      Found {nan_count} NaN values — filling with column medians")
    X = X.fillna(X.median(numeric_only=True))
else:
    print("      No missing values found")

# Encode string class labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"      Classes: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc,
)
print(f"      Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")


# =============================================================================
# 3. FEATURE SELECTION  (20531 → ~18000 via variance → 50 via ANOVA F-test)
# =============================================================================

print("[3/7] Feature selection ...")

# Stage 1 — remove near-zero-variance genes (threshold keeps genes with var > 0.01)
vt = VarianceThreshold(threshold=0.01)
X_train_vt = vt.fit_transform(X_train)
X_test_vt  = vt.transform(X_test)
print(f"      After VarianceThreshold : {X_train_vt.shape[1]} features")

# Stage 2 — keep top 50 genes by ANOVA F-score (fit on train only — no leakage)
skb = SelectKBest(score_func=f_classif, k=50)
X_train_sel = skb.fit_transform(X_train_vt, y_train)
X_test_sel  = skb.transform(X_test_vt)

# Recover human-readable gene names for the 50 selected features
gene_names_after_vt  = np.array(X_train.columns)[vt.get_support()]
selected_gene_names  = gene_names_after_vt[skb.get_support()]

print(f"      After SelectKBest(k=50) : {X_train_sel.shape[1]} features")
print(f"      Top 5 selected genes    : {list(selected_gene_names[:5])}\n")


# =============================================================================
# 4. TRAIN RANDOM FOREST
# =============================================================================

print("[4/7] Training RandomForestClassifier (100 trees) ...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_sel, y_train)
print("      Training complete\n")


# =============================================================================
# 5. CROSS-VALIDATION (5-fold, stratified)
# =============================================================================

print("[5/7] 5-fold stratified cross-validation ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train_sel, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"      Fold scores : {[round(s, 4) for s in cv_scores]}")
print(f"      CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")


# =============================================================================
# 6. TEST-SET EVALUATION + CONFUSION MATRIX
# =============================================================================

print("[6/7] Evaluating on test set ...")
y_pred = clf.predict(X_test_sel)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')
print(f"      Test Accuracy  : {acc:.4f}")
print(f"      Weighted F1    : {f1:.4f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    linewidths=0.5,
    ax=ax,
)
ax.set_title('Confusion Matrix — Cancer Type Classification', fontsize=14, pad=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"      Saved : {cm_path}\n")


# =============================================================================
# 7. FEATURE IMPORTANCE CHART (top 20 genes)
# =============================================================================

print("[7/7] Plotting feature importances ...")
importances = clf.feature_importances_              # shape (50,), sums to 1.0
indices     = np.argsort(importances)[::-1]         # descending rank
top_n       = 20
top_names   = selected_gene_names[indices[:top_n]]
top_values  = importances[indices[:top_n]]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(top_n), top_values, color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_xticks(range(top_n))
ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=9)
ax.set_title('Top 20 Most Important Genes (Random Forest — Mean Decrease in Impurity)', fontsize=13, pad=12)
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
# SUMMARY
# =============================================================================

print("=" * 60)
print("  DONE")
print("=" * 60)
print(f"  CV Accuracy   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Test Accuracy : {acc:.4f}")
print(f"  Weighted F1   : {f1:.4f}")
print(f"  Outputs dir   : {OUTPUT_DIR}")
print("=" * 60)
