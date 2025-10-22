import os
from dotenv import load_dotenv

import pandas as pd
import matplotlib.pyplot as plt

env = load_dotenv()

FILE_PATH = os.getenv("FILE_PATH")
FILE_NAME_INPUT = os.getenv("FILE_NAME_INPUT")
FILE_NAME_OUTPUT = os.getenv("FILE_NAME_OUTPUT")

DEFAULT_ZERO_THRESHOLD = os.getenv("DEFAULT_ZERO_THRESHOLD")
SAVE_INTERMEDIATE = os.getenv("SAVE_INTERMEDIATE") == 'True'

# Step 0: Load your CSV
df = pd.read_csv(FILE_PATH + '/' + FILE_NAME_INPUT)

# Step 1: Identify feature columns (everything except row ID and Metabolomics_sequence)
non_feature_cols = ["row ID", "Metabolomics_sequence"]
feature_cols = [col for col in df.columns if col not in non_feature_cols]

# Step 2: Normalize by internal standard (column '11974')
# We'll divide every feature column by the values in column '11974' row-wise
df[feature_cols] = df[feature_cols].div(df["11974"], axis=0)

# Optional: Save intermediate result
if SAVE_INTERMEDIATE:
    df.to_csv(FILE_PATH + "/intermediateFile_IS_normalized.csv", index=False)

# Step 5: Prepare for imputation ===
metadata_df = df[non_feature_cols].copy()
features_df = df[feature_cols].copy()

# Remove features with more than 20% zero values
zero_threshold = DEFAULT_ZERO_THRESHOLD or 0.2

# Calculate fraction of zeros per feature
zero_fractions = (features_df == 0).sum() / len(features_df)

# Keep only features where ≤ 20% of values are zero
features_df = features_df.loc[:, zero_fractions <= zero_threshold]

# Update feature_cols list accordingly
feature_cols = features_df.columns.tolist()

# === Step 6: Batch-wise imputation using LOD/2 ===
imputation_log = {}

def impute_lod_half(group):
    for col in group.columns:
        # Count how many values need imputation (NaN or 0)
        to_impute = group[col].isna().sum() + (group[col] == 0).sum()
        real_values = group[col][(group[col] != 0) & (group[col].notna())]
        
        if not real_values.empty:
            lod_half = real_values.min() / 2
            group[col] = group[col].apply(lambda x: lod_half if pd.isna(x) or x == 0 else x)
        
        # Add to imputation log
        imputation_log[col] = imputation_log.get(col, 0) + to_impute
    return group

# Apply imputation
features_imputed = features_df.groupby(df["Metabolomics_sequence"], group_keys=False).apply(impute_lod_half)

# Reassemble full DataFrame
df_imputed = pd.concat([metadata_df.reset_index(drop=True), features_imputed.reset_index(drop=True)], axis=1)

# Save the imputed data
df_imputed.to_csv(FILE_PATH + "/intermediateFile_normalized_imputed.csv", index=False)

# === Step 7: Visualize imputation counts ===
# Convert imputation log to DataFrame
imputation_df = pd.DataFrame.from_dict(imputation_log, orient='index', columns=["Imputed_Count"])
imputation_df.sort_values("Imputed_Count", ascending=False, inplace=True)

# Plot
plt.figure(figsize=(16, 6))
plt.bar(imputation_df.index, imputation_df["Imputed_Count"])
plt.xticks(rotation=90)
plt.ylabel("Number of Imputed Values")
plt.title("Imputed 0s and NaNs per Feature (LOD/2 Imputation)")
plt.tight_layout()
plt.savefig("imputed_counts_per_feature.png", dpi=300)
plt.show()

# === Step 8: Batch-wise median normalization ===
def normalize_by_batch_median(group):
    return group[feature_cols] / group[feature_cols].median()

normalized_features = df_imputed.groupby("Metabolomics_sequence", group_keys=False)[feature_cols].apply(normalize_by_batch_median)

# === Step 9: Replace infinities (just in case) ===
normalized_features.replace([float('inf'), -float('inf')], 0, inplace=True)

# === Step 10: Final reassembly ===
final_df = pd.concat([df_imputed[non_feature_cols].reset_index(drop=True),
                      normalized_features.reset_index(drop=True)], axis=1)

# === Step 11: Save output ===
final_df.to_csv(FILE_PATH + '/' + FILE_NAME_OUTPUT, index=False)

# === Preview the result ===
#final_df.head()


# Step 1: Reload raw data
raw_df = pd.read_csv(FILE_PATH + '/' + FILE_NAME_INPUT)

# Define metadata columns
non_feature_cols = ["row ID", "Metabolomics_sequence"]

# Get feature columns from final_df (only retained features)
feature_cols = [col for col in final_df.columns if col not in non_feature_cols]

# Raw feature columns (intersect only those retained in final_df)
raw_feature_cols = [col for col in raw_df.columns if col in feature_cols]

# Step 2: Prepare IS-normalized version (already done earlier as `df`)
rsd_is = df[feature_cols].std() / df[feature_cols].mean() * 100

# Step 3: Prepare median-normalized-only version
median_only_df = raw_df.copy()
median_only_norm = median_only_df.groupby("Metabolomics_sequence", group_keys=False)[feature_cols].apply(
    lambda g: g / g.median()
)

# Step 4: Define function to calculate RSD
def rsd(df_subset, feature_cols):
    return df_subset[feature_cols].std() / df_subset[feature_cols].mean() * 100

# Step 5: Calculate RSDs
rsd_raw = rsd(raw_df, raw_feature_cols)
rsd_median_only = rsd(median_only_norm, feature_cols)
rsd_final = rsd(final_df, feature_cols)

# Step 6: Combine into one DataFrame
rsd_df = pd.DataFrame({
    "Feature": feature_cols,
    "RSD_Raw": rsd_raw.values,
    "RSD_IS_Normalized": rsd_is.values,
    "RSD_Median_Only": rsd_median_only.values,
    "RSD_IS_and_Median": rsd_final.values
})

# Step 7: Save RSD values per feature
rsd_df.to_csv("RSDs_per_feature.csv", index=False)

# Step 8: Summary statistics
summary = rsd_df.drop(columns="Feature").agg(["min", "max", "median"]).T
summary = summary.rename(columns={
    "min": "Min RSD (%)",
    "max": "Max RSD (%)",
    "median": "Median RSD (%)"
})
summary.index.name = "Normalization Step"
summary.to_csv("RSD_summary.csv")

# Step 9: Plot RSDs
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
plt.savefig(FILE_PATH + "/RSD_comparison_plot_20percMV.png", dpi=300, bbox_inches='tight')
plt.show()

# Step 10: Display summary
print("🔍 Summary of RSDs per normalization step:")
print(summary)


# Step 1: Combine RSDs into one DataFrame
rsd_df = pd.DataFrame({
    "Feature": feature_cols,
    "RSD_Raw": rsd_raw.values,
    "RSD_IS_Normalized": rsd_is.values,
    "RSD_Median_Only": rsd_median_only.values,
    "RSD_IS_and_Median": rsd_final.values
})

# Step 2: Save to CSV
rsd_df.to_csv(FILE_PATH + "/RSDs_per_feature.csv", index=False)

# Step 3: Summary stats
summary = rsd_df.drop(columns="Feature").agg(["min", "max", "median"]).T
summary = summary.rename(columns={
    "min": "Min RSD (%)",
    "max": "Max RSD (%)",
    "median": "Median RSD (%)"
})
summary.index.name = "Normalization Step"

# Step 4: Save summary to CSV
summary.to_csv(FILE_PATH + "/RSD_summary.csv")

# Optional: Display summary
print("🔍 Summary of RSDs per normalization step:")
print(summary)
