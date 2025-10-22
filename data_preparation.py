import sys
import os
from dotenv import load_dotenv
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Step 0.1: Loading environment variables from .env file
env = load_dotenv()
def get_bool_env(var_name, default=False):
    return os.getenv(var_name, str(default)).lower() in ("1", "true", "yes", "on")

FILE_PATH_INPUT = os.getenv("FILE_PATH_INPUT")
FILE_NAME_INPUT = os.getenv("FILE_NAME_INPUT")
FILE_PATH_OUTPUT = os.getenv("FILE_PATH_OUTPUT")
FILE_NAME_OUTPUT = os.getenv("FILE_NAME_OUTPUT")

IS_COL = os.getenv("IS_COL", "11974")
DEFAULT_ZERO_THRESHOLD = float(os.getenv("DEFAULT_ZERO_THRESHOLD", 0.2))
SAVE_INTERMEDIATE = get_bool_env("SAVE_INTERMEDIATE", True)

LOGGING_ENABLED = get_bool_env("LOGGING_ENABLED", False)
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")


# Step 0.2: Generating output directory
try:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # avoid ':' for Windows
    OUTPUT_DIR = os.path.join(FILE_PATH_OUTPUT, f"output_{stamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"Failed to create output directory: {e}")
    sys.exit(1)


# Step 0.3: Setting up logging
if LOGGING_ENABLED:
    logging_path = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(logging_path, exist_ok=True)
    log_handlers = [logging.StreamHandler()]
    log_file_path = os.path.join(logging_path, 'data_preparation.log')
    log_handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(
        level=logging.INFO, 
        handlers=log_handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S'
    )
logging.info(f"Logging is set up successfully. Starting data preparation.")


# Step 0.4: Load your CSV
try:
    df = pd.read_csv(FILE_PATH_INPUT + '/' + FILE_NAME_INPUT)
    logging.info(f"Input CSV file ({FILE_NAME_INPUT}) loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")


# Step 1: Identify feature columns 
# Feature columns: everything except "row ID", "Metabolomics_sequence", and "batch"
try:
    non_feature_cols = ["row ID", "Metabolomics_sequence", "batch"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    logging.info(f"Step 1. Got non-feature columns: {non_feature_cols}")
except Exception as e:
    logging.error(f"Error in identifying non-feature columns: {e}")


# Step 2: Normalize by IS (internal standard)
# We'll divide every feature column by the values in IS_COL column (default: '11974') row-wise
try:
    if IS_COL not in df.columns:
        logging.error(f"Internal standard column '{IS_COL}' not found in input.")
        raise SystemExit(1)
    logging.info(f"Internal standard column '{IS_COL}' found with {df[IS_COL].notna().sum()} non-missing values.")
    
    # Coerce everything to numeric
    # And voiding zero-division by substituting zeros with NaN
    IS_vector = pd.to_numeric(df[IS_COL], errors="coerce").replace(0, np.nan)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    
    df[feature_cols] = df[feature_cols].div(IS_vector, axis=0)
    logging.info(f"Step 2. IS normalization complete.")
except Exception as e:
    logging.error(f"Error in IS-normalization: {e}")
    sys.exit(1)


# Optional. Step 2.1: Save intermediate IS-normalized result CSV
if SAVE_INTERMEDIATE:
    file_name_intermediate_file_is_normalized = "intermediateFile_IS_normalized.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, file_name_intermediate_file_is_normalized), index=False)
    logging.info(f"Step 2.1. Intermediate result file ({file_name_intermediate_file_is_normalized}) is saved.")


# Step 3: Preparing for imputation
# Copying values from the dataframe
metadata_df = df[non_feature_cols].copy()
features_df = df[feature_cols].copy()
logging.info(f"Step 3. Copying values from the dataframe complete.")


# Step 4: Removing features with more than 20% zero values
# Keep only features where ≤ 20% of values are zero
try:
    zero_threshold = DEFAULT_ZERO_THRESHOLD if DEFAULT_ZERO_THRESHOLD is not None else 0.2
    zero_fractions = (features_df.eq(0) | features_df.isna()).sum() / len(features_df)
    features_df = features_df.loc[:, zero_fractions <= zero_threshold]
    feature_cols = features_df.columns.tolist()
    logging.info(f"Step 4. Zero/NaN fraction applied at threshold {zero_threshold}. Kept {features_df.shape[1]} features out of {len(zero_fractions)}.")
except Exception as e:
    logging.error(f"Error in 'zero-threshold'-ing: {e}")
    sys.exit(1)


# Step 5: Batch-wise imputation using LOD/2
imputation_log = {}
def impute_lod_half(group):
    for col in group.columns:
        # Count how many values need imputation (NaN or 0)
        to_impute = group[col].isna().sum() + group[col].eq(0).sum()
        real_values = group[col][(group[col] != 0) & (group[col].notna())]
        
        if not real_values.empty:
            lod_half = real_values.min() / 2
            group[col] = group[col].apply(lambda x: lod_half if pd.isna(x) or x == 0 else x)
        
        # Add to imputation log
        imputation_log[col] = imputation_log.get(col, 0) + to_impute
    return group

try:
    # Apply imputation
    features_imputed = features_df.groupby(df["Metabolomics_sequence"], group_keys=False).apply(impute_lod_half)

    # Reassemble full DataFrame
    df_imputed = pd.concat([metadata_df.reset_index(drop=True), features_imputed.reset_index(drop=True)], axis=1)
    logging.info(f"Step 5. Imputation complete.")
except Exception as e:
    logging.error(f"Imputation error: {e}")
    sys.exit(1)


# Optional. Step 5.1: Save intermediate the imputed data
if SAVE_INTERMEDIATE:
    file_name_intermediate_file_is_imputed = "intermediateFile_normalized_imputed.csv"
    df_imputed.to_csv(os.path.join(OUTPUT_DIR, file_name_intermediate_file_is_imputed), index=False)
    logging.info(f"Step 5.1. Intermediate result file ({file_name_intermediate_file_is_imputed}) is saved.")


# Step 6: Visualize imputation counts
# Convert imputation log to DataFrame
try:
    imputation_df = pd.DataFrame.from_dict(imputation_log, orient='index', columns=["Imputed_Count"])
    imputation_df.sort_values("Imputed_Count", ascending=False, inplace=True)

    # Plot
    plt.figure(figsize=(16, 6))
    plt.bar(imputation_df.index, imputation_df["Imputed_Count"])
    plt.xticks(rotation=90)
    plt.ylabel("Number of Imputed Values")
    plt.title("Imputed 0s and NaNs per Feature (LOD/2 Imputation)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'imputed_counts_per_feature.png'), dpi=300)
    # plt.show()

    logging.info("Step 6. Visualization of imputation counts complete.")
except Exception as e:
    logging.error(f"Error in visualization of imputation counts: {e}")


# Step 7: Batch-wise median normalization
def normalize_by_batch_median(group):
    return group[feature_cols] / group[feature_cols].median()

try:
    normalized_features = df_imputed.groupby("Metabolomics_sequence", group_keys=False)[feature_cols].apply(normalize_by_batch_median)
    logging.info("Step 7. Median normalization complete.")
except Exception as e:
    logging.error(f"Error in median normalization: {e}")
    sys.exit(1)


# Step 8: Final reassembly
try:
    # Replace infinities (just in case)
    normalized_features.replace([float('inf'), -float('inf')], 0, inplace=True)

    final_df = pd.concat([df_imputed[non_feature_cols].reset_index(drop=True),
                          normalized_features.reset_index(drop=True)], axis=1)
    logging.info("Step 8. Final reassembly complete.")
except Exception as e:
    logging.error(f"Error in final reassembly: {e}")
    sys.exit(1)


# Step 9: Save output
try:
    final_df.to_csv(os.path.join(OUTPUT_DIR, FILE_NAME_OUTPUT), index=False)
    logging.info(f"Step 9. Output is saved successfully.")
except Exception as e:
    logging.error(f"Error in saving output: {e}")
    sys.exit(1)


# Step 10: Preview the result
final_df.head()
logging.info(f"Step 10: Preview the result: {final_df.head()}")


# Step 11: Reload raw data
try:
    raw_df = pd.read_csv(FILE_PATH_INPUT + '/' + FILE_NAME_INPUT)
    logging.info(f"Step 11. Reloaded input CSV ({FILE_NAME_INPUT}).")
except Exception as e:
    logging.error(f"Error in reloading input CSV: {e}")


# Step 12: Get feature columns from final_df (only retained features)
try:
    feature_cols = [col for col in final_df.columns if col not in non_feature_cols]
    # Raw feature columns (intersect only those retained in final_df)
    raw_feature_cols = [col for col in raw_df.columns if col in feature_cols]
    logging.info("Step 12: Getting feature columns complete.")
except Exception as e:
    logging.error(f"Error in getting feature columns from final_df: {e}")
    sys.exit(1)


# Step 13: Prepare IS-normalized version (already done earlier as `df`)
try:
    rsd_is = df[feature_cols].std() / df[feature_cols].mean() * 100
    logging.info("Step 13: IS-normalization complete.")
except Exception as e:
    logging.error(f"Error in IS-normalization: {e}")
    sys.exit(1)


# Step 14: Prepare median-normalized-only version
try:
    median_only_df = raw_df.copy()
    median_only_norm = median_only_df.groupby("Metabolomics_sequence", group_keys=False)[feature_cols].apply(
        lambda g: g / g.median()
    )
    logging.info("Step 14: Median normalization complete.")
except Exception as e:
    logging.error(f"Error in median-normalization: {e}")
    sys.exit(1)


# Step 15: Calculate RSDs (Relative Standard Deviations)
def rsd(df_subset, feature_cols):
    return df_subset[feature_cols].std() / df_subset[feature_cols].mean() * 100

try:
    rsd_raw = rsd(raw_df, raw_feature_cols)
    rsd_median_only = rsd(median_only_norm, feature_cols)
    rsd_final = rsd(final_df, feature_cols)
    logging.info("Step 15: RSDs calculated.")
except Exception as e:
    logging.error(f"Error in RSDs calculation: {e}")
    sys.exit(1)


# Step 16: Combine into one DataFrame
rsd_df = pd.DataFrame({
    "Feature": feature_cols,
    "RSD_Raw": rsd_raw.values,
    "RSD_IS_Normalized": rsd_is.values,
    "RSD_Median_Only": rsd_median_only.values,
    "RSD_IS_and_Median": rsd_final.values
})
logging.info("Step 16: All combined in one dataframe.")


# Step 17: Save RSD values per feature
try:
    rsd_df.to_csv(os.path.join(OUTPUT_DIR, 'RSDs_per_feature.csv'), index=False)
    logging.info("Step 17. RSD values per feature CSV saved.")
except Exception as e:
    logging.error(f"Error in saving RSD values per feature: {e}")


# Step 18: Summary statistics
try:
    summary = rsd_df.drop(columns="Feature").agg(["min", "max", "median"]).T
    summary = summary.rename(columns={
        "min": "Min RSD (%)",
        "max": "Max RSD (%)",
        "median": "Median RSD (%)"
    })
    summary.index.name = "Normalization Step"
    summary.to_csv(os.path.join(OUTPUT_DIR, 'RSD_summary.csv'))
    logging.info("Step 18. Summary statistics saved.")
except Exception as e:
    logging.error(f"Error in summary statistics: {e}")


# Step 19: Visualization of RSDs
try:
    plt.figure(figsize=(14, 6))
    plt.plot(rsd_raw.values, label="Raw", marker='o', linestyle='-', alpha=0.5)
    plt.plot(rsd_is.values, label="IS Normalized", marker='x', linestyle='-', alpha=0.7)
    plt.plot(rsd_final.values, label="IS + Median Normalized", marker='s', linestyle='-', alpha=0.9)
    plt.plot(rsd_median_only.values, label="Median Normalized (no IS)", marker='d', linestyle='-', alpha=0.7)
    plt.xlabel("Feature Index")
    plt.ylabel("RSD (%)")
    plt.title("RSD per Feature: Raw vs. Normalizations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Export the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "RSD_comparison_plot_20percMV.png"), dpi=300, bbox_inches='tight')
    # plt.show()
except Exception as e:
    logging.error(f"Error in visualization of RSDs: {e}")


# Step 20: Preview summary of of RSDs per normalization step
logging.info(f"Step 20: Summary of RSDs per normalization step: {summary}")
