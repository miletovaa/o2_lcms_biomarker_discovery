# === Random Forest Metabolomics Pipeline — CLEAN STRUCTURE ===

# -----------------------------
# 1. IMPORTS & CONFIGURATION
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_fscore_support, f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.metrics import classification_report, confusion_matrix
# -----------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------
file_path = 'C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Output_file_merged_logtransformed_autoscaled_ANP_RP_PDvsCON.csv'
df = pd.read_csv(file_path)
df = df.set_index(df.columns[0])
labels = df.iloc[:, 0]              # Group column (e.g., Control/PD)
features = df.iloc[:, 1:]           # All feature columns
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X = features.copy()

# -----------------------------
# 3. TRAIN BASE MODEL & CROSS-VALIDATION (WITH EXPORTS)
# -----------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)

# Cross-validated predictions
y_proba = cross_val_predict(rf, X, y, cv=skf, method='predict_proba')[:, 1]
y_pred_cv = cross_val_predict(rf, X, y, cv=skf, method='predict')

# Classification report (text + dict form)
print("\n=== Cross-Validated Classification Report ===")
print(classification_report(y, y_pred_cv, target_names=label_encoder.classes_))

report_dict = classification_report(y, y_pred_cv, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/classification_report_cv.csv")

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred_cv)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
conf_matrix_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/confusion_matrix_cv.csv")

print("✅ Classification report and confusion matrix exported.")

# -----------------------------
# 4. THRESHOLD TUNING (WITH EXPORTS)
# -----------------------------

threshold = 0.56
y_pred_thresh = (y_proba >= threshold).astype(int)

# Classification report
print(f"\n=== Classification Report at Threshold {threshold} ===")
report_thresh = classification_report(y, y_pred_thresh, target_names=label_encoder.classes_)
print(report_thresh)

# Save classification report as CSV
report_thresh_dict = classification_report(y, y_pred_thresh, target_names=label_encoder.classes_, output_dict=True)
report_thresh_df = pd.DataFrame(report_thresh_dict).transpose()
report_thresh_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/classification_report_thresholded.csv")

# Save confusion matrix
conf_matrix_thresh = confusion_matrix(y, y_pred_thresh)
conf_matrix_thresh_df = pd.DataFrame(
    conf_matrix_thresh,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
conf_matrix_thresh_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/confusion_matrix_thresholded.csv")

print("✅ Thresholded classification report and confusion matrix exported.")

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/ROC_curve_CV.png", dpi=300)
plt.show()

# Threshold sensitivity plot
thresholds = np.linspace(0.1, 0.9, 81)
precision_control, recall_control, f1_control = [], [], []
precision_pd, recall_pd, f1_pd = [], [], []

for t in thresholds:
    y_t = (y_proba >= t).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_t, labels=[0, 1], zero_division=0)
    precision_control.append(precision[0])
    recall_control.append(recall[0])
    f1_control.append(f1[0])
    precision_pd.append(precision[1])
    recall_pd.append(recall[1])
    f1_pd.append(f1[1])

plt.figure(figsize=(12, 6))
plt.plot(thresholds, precision_control, label="Precision (Control)", linestyle="--")
plt.plot(thresholds, recall_control, label="Recall (Control)", linestyle="-")
plt.plot(thresholds, f1_control, label="F1-score (Control)", linestyle="-.")
plt.plot(thresholds, precision_pd, label="Precision (PD)", linestyle="--", alpha=0.5)
plt.plot(thresholds, recall_pd, label="Recall (PD)", linestyle="-", alpha=0.5)
plt.plot(thresholds, f1_pd, label="F1-score (PD)", linestyle="-.", alpha=0.5)
plt.axvline(threshold, color="gray", linestyle=":", label=f"Threshold = {threshold}")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-score at Different Thresholds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Threshold_sensitivity_plot.png", dpi=300)
plt.show()

# -----------------------------
# 5. FEATURE IMPORTANCE: MDI + Elbow Cutoff
# -----------------------------

# Fit model and get importances
rf.fit(X, y)
mdi_importances = pd.Series(rf.feature_importances_, index=X.columns)
mdi_sorted = mdi_importances.sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'Feature', 0: 'Importance'})

# Elbow detection: find point farthest from line between first and last
x_vals = np.arange(len(mdi_sorted))
y_vals = mdi_sorted['Importance'].values
start = [0, y_vals[0]]
end = [len(y_vals) - 1, y_vals[-1]]

distances = []
for i in range(len(y_vals)):
    numerator = np.abs((end[1] - start[1]) * i - (end[0] - start[0]) * y_vals[i] + end[0]*start[1] - end[1]*start[0])
    denominator = np.sqrt((end[1] - start[1])**2 + (end[0] - start[0])**2)
    distances.append(numerator / denominator)

elbow_idx = np.argmax(distances)
elbow_cutoff = y_vals[elbow_idx]

print(f"🔵 Auto-detected MDI cutoff (elbow): {elbow_cutoff:.5f}")

# Filter top features above the elbow threshold
#top_filtered = mdi_sorted[mdi_sorted['Importance'] >= elbow_cutoff].copy()
top_filtered = mdi_sorted[mdi_sorted['Importance'] >= 0.01].copy()
# Plot full with elbow marker
plt.figure(figsize=(10, 6))
plt.plot(y_vals, marker='o', label='MDI Importance')
plt.axvline(elbow_idx, linestyle='--', color='red', label=f'Elbow @ {elbow_cutoff:.5f}')
plt.title("MDI Feature Importances with Elbow Point")
plt.xlabel("Feature Rank")
plt.ylabel("Importance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/MDI_Elbow_Plot.png", dpi=300)
plt.show()

# Plot filtered top features only
plt.figure(figsize=(40, 24))
top_filtered.sort_values('Importance').plot(
    kind='barh', x='Feature', y='Importance', legend=False, color='slateblue')
plt.title("Top MDI Features (Above Elbow Cutoff)")
plt.xlabel("Importance Score")
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/MDI_Top_Above_Elbow.png", dpi=300)
plt.show()


# -----------------------------
# 6. PERMUTATION IMPORTANCE (THRESHOLDED F1)
# -----------------------------
def custom_thresholded_f1(estimator, X, y, threshold=0.56):
    y_proba = estimator.predict_proba(X)
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    return f1_score(y, y_pred)

#top_n = 30
#top_features = mdi_importances.sort_values(ascending=False).head(top_n).index
#X_top = X[top_features]  # ← Subset
X_top = X.copy()           # ← Full dataset
perm_scores = np.zeros(X_top.shape[1])
feature_names = X_top.columns

print(f"\n🔁 Running permutation importance on {'all features' if X_top.shape[1] == X.shape[1] else f'top {X_top.shape[1]} features'}...")

# === Loop over CV folds with progress bar
for fold_idx, (train_idx, test_idx) in enumerate(tqdm(skf.split(X_top, y), total=skf.get_n_splits(), desc="CV Folds")):
    X_train, X_test = X_top.iloc[train_idx], X_top.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = clone(rf)
    model.fit(X_train, y_train)

    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring=lambda est, X_, y_: custom_thresholded_f1(est, X_, y_, threshold)
    )
    perm_scores += perm_result.importances_mean

# === Average over folds
perm_scores /= skf.get_n_splits()
perm_importances = pd.Series(perm_scores, index=feature_names)

# === Plot
perm_importances.sort_values(ascending=True).tail(30).plot(
    kind='barh', figsize=(10, 6), color='cornflowerblue')
plt.title("Top 30 Permutation Importances (Thresholded F1)")
plt.xlabel("Mean Importance (F1 Drop)")
plt.ylabel("Feature")
plt.grid(True, axis='x', linestyle=':')
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Permutation_Feature_Importance.png", dpi=300)
plt.show()

# -----------------------------
# 7. DROP-COLUMN IMPORTANCE (THRESHOLDED F1)
# -----------------------------
def custom_thresholded_f1(estimator, X, y, threshold=0.56):
    y_proba = estimator.predict_proba(X)
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    return f1_score(y, y_pred)

# === Select feature set (toggle here)
#top_n = 30
#top_features = mdi_importances.sort_values(ascending=False).head(top_n).index
#X_top = X[top_features]  # Subset
X_top = X.copy()           # Full dataset
top_features = X_top.columns

# === Init storage
drop_scores = np.zeros(len(top_features))

print(f"\n🔁 Running drop-column importance on {'all features' if X_top.shape[1] == X.shape[1] else f'top {X_top.shape[1]} features'}...")

# === CV Loop with per-fold & per-feature progress
for fold_idx, (train_idx, test_idx) in enumerate(tqdm(skf.split(X_top, y), total=skf.get_n_splits(), desc='CV Folds')):
    X_train, X_test = X_top.iloc[train_idx], X_top.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    baseline_model = clone(rf)
    baseline_model.fit(X_train, y_train)
    baseline_f1 = custom_thresholded_f1(baseline_model, X_test, y_test)

    for i, feat in enumerate(tqdm(top_features, desc=f"Fold {fold_idx+1} — Drop Features", leave=False)):
        X_train_dropped = X_train.drop(columns=[feat])
        X_test_dropped = X_test.drop(columns=[feat])
        model_dropped = clone(rf)
        model_dropped.fit(X_train_dropped, y_train)
        f1_drop = custom_thresholded_f1(model_dropped, X_test_dropped, y_test)
        drop_scores[i] += (baseline_f1 - f1_drop)

# === Average over folds
drop_scores /= skf.get_n_splits()
drop_importances = pd.Series(drop_scores, index=top_features)

# === Plot top 30
plot_top_n = 50
drop_importances.sort_values(ascending=True).tail(plot_top_n).plot(
    kind='barh', figsize=(10, 6), color='mediumseagreen')
plt.title(f"Top {plot_top_n} Drop-Column Importances (Thresholded F1)")
plt.xlabel("Mean F1 Drop")
plt.ylabel("Feature")
plt.grid(True, axis='x', linestyle=':')
plt.tight_layout()
plt.savefig("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Drop_column_Feature_Importance.png", dpi=300)
plt.show()


# -----------------------------
# 8. EXPORTS: FEATURE TABLES
# -----------------------------
mdi_df = mdi_importances.reset_index().rename(columns={'index': 'Feature', 0: 'MDI'})
mdi_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/MDI_importances.csv", index=False)
perm_df = perm_importances.reset_index().rename(columns={'index': 'Feature', 0: 'Permutation'})
perm_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Permutation_importances.csv", index=False)
drop_df = drop_importances.reset_index().rename(columns={'index': 'Feature', 0: 'DropColumn'})
perm_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Dropcolumn_importances.csv", index=False)

# Merge and annotate
merged = mdi_df.merge(perm_df, on='Feature').merge(drop_df, on='Feature')
merged.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/feature_importance_all_methods.csv", index=False)

# -----------------------------
# METADATA ENRICHMENT FOR MERGED IMPORTANCE FILE
# -----------------------------

# === Load the merged importance file
merged_df = pd.read_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/feature_importance_all_methods.csv")

# === Add 'row_id' and 'Source' parsing from Feature column
merged_df["row_id"] = merged_df["Feature"].apply(lambda x: x.split("_")[0])
merged_df["Source"] = merged_df["Feature"].apply(lambda x: x.split("_")[-1])

# === Load metadata files
anp_meta = pd.read_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/ANP_quant_samples_area_metadata_rowIDs.csv")
rp_meta = pd.read_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/RP_quant_samples_area_metadata_rowIDs.csv")

# Standardize metadata format
anp_meta["Source"] = "ANP"
rp_meta["Source"] = "RP"
anp_meta["row_id"] = anp_meta["row ID"].astype(str)
rp_meta["row_id"] = rp_meta["row ID"].astype(str)

# === Merge metadata
anp_enriched = pd.merge(merged_df[merged_df["Source"] == "ANP"], anp_meta, on="row_id", how="left")
rp_enriched = pd.merge(merged_df[merged_df["Source"] == "RP"], rp_meta, on="row_id", how="left")

# Combine both
full_enriched = pd.concat([anp_enriched, rp_enriched], ignore_index=True)

# === Save enriched file
full_enriched.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/feature_importance_all_methods_with_metadata.csv", index=False)

print("✅ Enriched merged importance file with metadata saved.")

# -----------------------------
# 9. FILTER ORIGINAL FILE BY SIGNIFICANT FEATURES
# -----------------------------

#  Manually define your list of significant feature names
significant_features = [
    '22763_RP', '30084_RP', '22733_RP', '22332_RP', '30189_RP',
    '21610_RP', '30049_RP', '4973_ANP', '20195_RP', '16398_RP',
    '27049_RP', '22739_RP', '29103_RP', '21389_RP', '15559_RP',
    '22742_RP', '30593_RP', '30527_RP', '14602_RP', '29123_RP',
    '11549_ANP', '27427_RP', '4251_ANP', '33935_RP', '6935_ANP',
    '24678_RP', '22350_ANP', '20199_RP', '32741_RP', '21828_RP',
    '6561_ANP', '16243_RP', '9860_RP', '10024_ANP', '15494_ANP',
    '29407_RP', '15359_RP', '23606_RP', '6399_ANP', '6356_ANP',
    '29893_RP', '9640_RP', '22768_RP', '12573_RP', '14571_RP',
    '21346_ANP', '9913_ANP', '11828_ANP', '15473_RP', '4550_ANP',
    '28941_RP', '31585_RP', '27088_RP', '10707_ANP', '20854_RP',
    '14521_RP', '24340_RP', '20833_RP', '17589_ANP', '28927_RP'
]

# Filter full DataFrame (including group labels)
filtered_df = df.loc[:, [df.columns[0]] + significant_features]

# Save filtered file
filtered_df.to_csv("C:/Users/492051/OneDrive - MUNI/Dokumenty/Python Scripts/Random Forest/Output_filtered_for_SIMCA_1.csv")
print("✅ Filtered DataFrame with selected significant features saved.")


#RBD PREDICTION

