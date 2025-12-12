# %% [markdown]
# # Random Forest Metabolomics Pipeline

# %% [markdown]
# ### Imports and config loading

# %%
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import cfg
from utils.output import save_figure, create_output_file_name

pd.set_option('future.no_silent_downcasting', True)

def main(cfg, output_rp=None, output_anp=None):
    # %% [markdown]
    if output_rp is None or output_anp is None:
        print("ERROR: Missing input data from previous steps. Please provide output_rp and output_anp dataframes.")
        return

    def get_output_file_name(type: str, description: str, ext: str) -> str:
        return create_output_file_name(
            step="s02rforest",
            type=type,
            description=description,
            ext=ext,
            cfg=cfg,
            output_dir=cfg.OUTPUT_DIR
        )

    # ### Merging output files from previous steps

    # %%
    input_anp = os.path.join(output_anp, 'p01_lcms_biomarker_discovery_output_01prep_anp_001_20251128.csv')
    input_rp = os.path.join(output_rp, 'p01_lcms_biomarker_discovery_output_01prep_rp_002_20251128.csv')

    df_anp = pd.read_csv(input_anp)
    df_rp = pd.read_csv(input_rp)

    # add group column for dataframes (PD/RBD/CON)
    df_anp["group"] = df_anp["row ID"].str.extract(r"(PD|RBD|CON)")
    df_rp["group"]  = df_rp["row ID"].str.extract(r"(PD|RBD|CON)")

    # remove "-ANP.raw Peak area" / "-RP.raw Peak area" from sample names
    pattern = r"-(ANP|RP)\.raw Peak area$"
    df_anp["row ID"] = df_anp["row ID"].str.replace(pattern, "", regex=True)
    df_rp["row ID"]  = df_rp["row ID"].str.replace(pattern, "", regex=True)

    # renaming feature column names with postfixes
    metadata_cols = ["row ID", "group", "batch"]
    anp_feature_cols = [c for c in df_anp.columns if c not in metadata_cols]
    rp_feature_cols  = [c for c in df_rp.columns  if c not in metadata_cols]
    df_anp = df_anp.rename(columns={c: f"{c}_ANP" for c in anp_feature_cols})
    df_rp  = df_rp.rename(columns={c: f"{c}_RP"  for c in rp_feature_cols})

    print("ANP dataframe shape: ", df_anp.shape)
    print("RP dataframe shape: ", df_rp.shape)

    # merge on row ID
    df_merged = df_anp.merge(df_rp, on=metadata_cols, how="outer")

    # reorder columns to have group column first
    cols = df_merged.columns.tolist()
    cols.remove("group")
    cols.insert(1, "group")
    df_merged = df_merged[cols]

    # %% [markdown]
    # ### Input file preprocessing
    # 
    # Making script compatible with different input files by renaming group column and standardizing group names 

    # %%
    # Preserve reproducibility with different input files
    df_merged = df_merged.dropna(subset=["group"])
    df_merged = df_merged.rename(columns={"Group": "group"})
    GROUP_ALIASES = {
        "CON": {"CON", "CONTROL", "CTRL"},
        "PD":  {"PD"},
        "RBD": {"RBD"},
    }
    alias2canon = {a.strip().upper(): canon
                for canon, aliases in GROUP_ALIASES.items()
                for a in aliases}
    df_merged["group"] = df_merged["group"].str.upper().map(alias2canon)

    print(df_merged.head(50))

    # %% [markdown]
    # ### Create dataframe

    # %%
    df = df_merged.set_index(df_merged.columns[0])
    labels = df["group"] # Group column (e.g., Control/PD)
    features = [c for c in df.columns if c not in metadata_cols]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    X = df[features].copy()

    print(labels.unique())

    # %% [markdown]
    # ## Train base model and cross-validation
    # 
    # ### Run Random Forest, Cross-Validation, and generate classification report and confusion matrix *(function definition)*

    # %%
    def do_rf_cross_val_conf_matrix_report(X_bin, y_bin, target_names):
        # Initialize Random Forest classifier
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_forest.seed)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=cfg.random_forest.seed)

        # Cross-validated predictions
        y_proba = cross_val_predict(rf, X_bin, y_bin, cv=skf, method='predict_proba')[:, 1]
        y_pred_cv = cross_val_predict(rf, X_bin, y_bin, cv=skf, method='predict')

        # Classification report (text + dict form)
        print("Cross-Validated Classification Report")
        print(classification_report(y_bin, y_pred_cv, target_names=target_names))

        report_dict = classification_report(y_bin, y_pred_cv, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(get_output_file_name(type="ml", description="classification_report", ext="csv"))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_bin, y_pred_cv)
        df_conf_matrix = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)
        print("Confusion Matrix:")
        print(df_conf_matrix)
        df_conf_matrix.to_csv(get_output_file_name(type="viz", description="rf_confusion_matrix", ext="csv"))

        print("✅ Classification report and confusion matrix exported.")

        return rf, y_proba

    # %% [markdown]
    # ### Threshold tuning *(function definition)*

    # %%
    def do_threshold_tuning(y_proba, y_bin, target_names):
        threshold = cfg.random_forest.threshold
        y_pred_thresh = (y_proba >= threshold).astype(int)

        # Classification report
        print(f"Classification Report at Threshold {threshold}")
        report_thresh = classification_report(y_bin, y_pred_thresh, target_names=target_names)
        print(report_thresh)

        # Save classification report as CSV
        report_thresh_dict = classification_report(y_bin, y_pred_thresh, target_names=target_names, output_dict=True)
        report_thresh_df = pd.DataFrame(report_thresh_dict).transpose()
        report_thresh_df.to_csv(get_output_file_name(type="ml", description="classification_report_thresholded", ext="csv"))

        # Save confusion matrix
        conf_matrix_thresh = confusion_matrix(y_bin, y_pred_thresh)
        conf_matrix_thresh_df = pd.DataFrame(
            conf_matrix_thresh,
            index=target_names,
            columns=target_names
        )
        conf_matrix_thresh_df.to_csv(get_output_file_name(type="viz", description="rf_confusion_matrix_thresholded", ext="csv"))

        print("✅ Thresholded classification report and confusion matrix exported.")


    # %% [markdown]
    # ### Visualization of ROC Curve and Threshold Sensitivity Plot

    # %%
    def plot_roc_curve_threshold(y_proba, y_bin):
        threshold = cfg.random_forest.threshold

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_bin, y_proba)
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
        save_figure(plt, get_output_file_name(type="fig", description="roc_curve", ext="png"))

        # Threshold sensitivity plot
        thresholds = np.linspace(0.1, 0.9, 81)
        precision_control, recall_control, f1_control = [], [], []
        precision_pd, recall_pd, f1_pd = [], [], []

        for t in thresholds:
            y_t = (y_proba >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_bin, y_t, labels=[0, 1], zero_division=0)
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
        save_figure(plt, get_output_file_name(type="fig", description="roc_curve_cv", ext="png"))

    # %% [markdown]
    # ### Feature importance: Mean Decrease in Impurity (MDI)

    # %%
    def get_mdi_importance(rf, X_bin, y_bin):
        # Fit model and get importances
        rf.fit(X_bin, y_bin)
        mdi_importances = pd.Series(rf.feature_importances_, index=X_bin.columns)
        return mdi_importances.sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'Feature', 0: 'Importance'})

    # %% [markdown]
    # ### Elbow method to select important features

    # %%
    def do_elbow_cutoff(mdi_sorted):
        # Elbow detection: find point farthest from line between first and last
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
            
        elbow_importance = cfg.random_forest.elbow_importance
        top_filtered = mdi_sorted[mdi_sorted['Importance'] >= elbow_importance].copy()
        
        return elbow_cutoff, elbow_idx, y_vals, top_filtered

    # %% [markdown]
    # ### Plot full with elbow marker and filtered top features only

    # %%
    def plot_top_features(elbow_cutoff, elbow_idx, y_vals, top_filtered):
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
        save_figure(plt, get_output_file_name(type="fig", description="mdi_elbow_plot", ext="png"))

        # Plot filtered top features only
        plt.figure(figsize=(40, 24))
        top_filtered.sort_values('Importance').plot(
            kind='barh', x='Feature', y='Importance', legend=False, color='slateblue')
        plt.title("Top MDI Features (Above Elbow Cutoff)")
        plt.xlabel("Importance Score")
        plt.grid(True, axis='x')
        plt.tight_layout()
        save_figure(plt, get_output_file_name(type="fig", description="mdi_top_features", ext="png"))

    # %% [markdown]
    # ### Exports: feature tables

    # %%
    def export_feature_table(mdi_importances):
        mdi_df = mdi_importances.reset_index().rename(columns={'index': 'Feature', 0: 'MDI'})
        mdi_df.to_csv(get_output_file_name(type="ml", description="mdi_importances", ext="csv"), index=False)

    # %% [markdown]
    # ### Initialization of the RF pipeline definition

    # %%
    def run_rf_pipeline(target_names):
        POSITIVE_CLASS, NEGATIVE_CLASS = target_names

        mask = labels.isin([POSITIVE_CLASS, NEGATIVE_CLASS])
        X_bin = X[mask]
        labels_bin = labels[mask]

        y_bin = (labels_bin == POSITIVE_CLASS).astype(int).values
        
        rf, y_proba = do_rf_cross_val_conf_matrix_report(X_bin, y_bin, target_names)
        do_threshold_tuning(y_proba, y_bin, target_names)
        plot_roc_curve_threshold(y_proba, y_bin)
        mdi_importances = get_mdi_importance(rf, X_bin, y_bin)
        elbow_cutoff, elbow_idx, y_vals, top_filtered = do_elbow_cutoff(mdi_importances)
        plot_top_features(elbow_cutoff, elbow_idx, y_vals, top_filtered)
        export_feature_table(mdi_importances)
        return top_filtered

    # %% [markdown]
    # ### Retrieving groups for pairwise classification

    # %%
    pair_to_model = cfg.random_forest.class_pair_to_model

    # ? pair_to_model is a variable from PARAMS file with the following possible integer options:
    pairs = [
        # 0 -- all 3 pairwise
        ["PD", "CON"], # 1
        ["PD", "RBD"], # 2
        ["RBD", "CON"] # 3
    ]

    if (pair_to_model is None or pair_to_model == 0):
        all_top_filtered = pd.DataFrame()
        for pair in pairs:
            print(f"Modeling {pair[0]} vs {pair[1]}")
            top_filtered = run_rf_pipeline(pair)
            top_filtered.to_csv(get_output_file_name(type="ml", description=f"mdi_top_{pair[0]}_vs_{pair[1]}", ext="csv"), index=False)
        print("✅ All pairwise modeling completed.")
    else:
        print(f"Modeling {pairs[pair_to_model - 1][0]} vs {pairs[pair_to_model - 1][1]}.")
        top_filtered = run_rf_pipeline(pairs[pair_to_model - 1])
        top_filtered.to_csv(get_output_file_name(type="ml", description=f"mdi_top_{pairs[pair_to_model - 1][0]}_vs_{pairs[pair_to_model - 1][1]}", ext="csv"), index=False)
        print(f"✅ Pairwise {pairs[pair_to_model - 1][0]} vs {pairs[pair_to_model - 1][1]} modeling completed.")

    print(all_top_filtered.head())

    # %%
    # # TODO: Manually define your list of significant feature names
    # significant_features = [
    #     '22763_RP', '30084_RP', '22733_RP', '22332_RP', '30189_RP',
    #     '21610_RP', '30049_RP', '4973_ANP', '20195_RP', '16398_RP',
    #     '27049_RP', '22739_RP', '29103_RP', '21389_RP', '15559_RP',
    #     '22742_RP', '30593_RP', '30527_RP', '14602_RP', '29123_RP',
    #     '11549_ANP', '27427_RP', '4251_ANP', '33935_RP', '6935_ANP',
    #     '24678_RP', '22350_ANP', '20199_RP', '32741_RP', '21828_RP',
    #     '6561_ANP', '16243_RP', '9860_RP', '10024_ANP', '15494_ANP',
    #     '29407_RP', '15359_RP', '23606_RP', '6399_ANP', '6356_ANP',
    #     '29893_RP', '9640_RP', '22768_RP', '12573_RP', '14571_RP',
    #     '21346_ANP', '9913_ANP', '11828_ANP', '15473_RP', '4550_ANP',
    #     '28941_RP', '31585_RP', '27088_RP', '10707_ANP', '20854_RP',
    #     '14521_RP', '24340_RP', '20833_RP', '17589_ANP', '28927_RP'
    # ]

    # # Filter full DataFrame (including group labels)
    # filtered_df = df.loc[:, [df.columns[0]] + significant_features]

    # # Save filtered file
    # filtered_df.to_csv(get_output_file_name(type="output", description="filtered_significant_features", ext="csv"))
    # print("✅ Filtered DataFrame with selected significant features saved.")


if __name__ == "__main__":
    output_rp = sys.argv[1] if len(sys.argv) > 1 else None
    output_anp = sys.argv[2] if len(sys.argv) > 2 else None
    if output_rp is None or output_anp is None:
        print("ERROR: Please provide output_rp and output_anp paths as command-line arguments.")
    main(cfg, output_rp=output_rp, output_anp=output_anp)