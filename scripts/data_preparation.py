import sys
import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import cfg


# Step 0.1: Loading environment variables from .env file
FILE_PATH_INPUT = cfg.get("FILE_PATH_INPUT", "./data")
FILE_PATH_OUTPUT = cfg.get("FILE_PATH_OUTPUT", "./results")
PARAMS_ID = cfg.get("PARAMS_ID", "000")
IS_COL = str(cfg.preprocessing.is_col)


# Step 0.2: Generating output directory
try:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # avoid ':' for Windows
    OUTPUT_DIR = os.path.join(FILE_PATH_OUTPUT, f"output_{PARAMS_ID}_{stamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"Failed to create output directory: {e}")
    sys.exit(1)


# Step 0.3: Setting up logging
if cfg.logging.enabled:
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
    df = pd.read_csv(FILE_PATH_INPUT + '/' + cfg.data.file_name_input)
    logging.info(f"Input CSV file ({cfg.data.file_name_input}) loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    sys.exit(1)


# Step 0.5: Orientation checks & transpose if needed
# Rules:
#   - If IS_COL is present as a column => assume CORRECT (samples as rows)
#   - Else transpose.
#   - If any column starts with SAMPLE_PREFIX (e.g., 'BIO-') => assume RP-style and TRANSPOSE.
try:
    sample_cols = [c for c in df.columns if str(c).startswith(cfg.data.sample_prefix)]
    meta_cols   = [c for c in df.columns if c not in sample_cols]
    need_transpose = (IS_COL not in df.columns) or (len(sample_cols) > 0)

    print(IS_COL not in meta_cols, len(sample_cols) > 0)
    # print(IS_COL, df.columns)

    if need_transpose:
        id_col = cfg.data.feature_id_colname if cfg.data.feature_id_colname in meta_cols else (meta_cols[0] if meta_cols else df.columns[0])
        logging.info(f"Transpose: using feature id column '{id_col}'. meta_cols={len(meta_cols)}; sample_cols={len(sample_cols)}")
        
        # Build a samples x features table from: [id_col] + sample columns
        t = (
            df[[id_col] + sample_cols]
              .set_index(id_col)
              .T
              .reset_index()
              .rename(columns={"index": "row ID"})
        )

        # Ensure required metadata columns exist for downstream steps
        if not {"Metabolomics_sequence", "batch"}.issubset(t.columns):
            df_batches = pd.read_csv(os.path.join(FILE_PATH_INPUT, cfg.data.file_name_batch_info))
            t["row ID"] = t["row ID"].str.split(".").str[0]
            mapping = df_batches.set_index("row ID")["Metabolomics_sequence"].to_dict()

            t["Metabolomics_sequence"] = t["row ID"].map(mapping)
            t["batch"] = t["row ID"].map(mapping)

            missing_batches = t.loc[t["batch"].isna(), "row ID"]
            logging.warning(f"{len(missing_batches)} rows missing batch info: {', '.join(missing_batches)}")

            t = t.dropna(subset=["batch"])

        # Reorder metadata first, keep all feature columns (the transposed ids)
        meta_first = ["row ID", "Metabolomics_sequence", "batch"]
        remaining = [c for c in t.columns if c not in meta_first]
        df = t[meta_first + remaining]

        rename_map = {c: str(c) for c in remaining}
        df = df.rename(columns=rename_map)
        logging.info(f"After transpose, first 5 feature columns: {list(rename_map.values())[:5]}")

        logging.info(f"Transpose complete. New shape={df.shape}. Samples={len(df)}; Features={len(remaining)}")

        df.to_csv(os.path.join(OUTPUT_DIR, "transposed_input_file.csv"), index=False)
    else:
        logging.info("Orientation check: looks correct (IS_COL present and no SAMPLE_PREFIX columns). No transpose performed.")
except Exception as e:
    logging.error(f"Error during orientation check/transpose: {e}")
    sys.exit(1)

# Keeping a copy of raw data for later RSD calculations
raw_df = df.copy()


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
if cfg.run.save_intermediate:
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
    zero_threshold = cfg.preprocessing.zero_threshold if cfg.preprocessing.zero_threshold is not None else 0.2
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
if cfg.run.save_intermediate:
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
    final_df.to_csv(os.path.join(OUTPUT_DIR, "OutputFile_normalized_imputed_median_normalized.csv"), index=False)
    logging.info(f"Step 9. Output is saved successfully.")
except Exception as e:
    logging.error(f"Error in saving output: {e}")
    sys.exit(1)


# Step 10: Preview the result
final_df.head()
logging.info(f"Step 10: Preview the result: {final_df.head()}")


# Step 11: Get feature columns from final_df (only retained features)
try:
    feature_cols = [col for col in final_df.columns if col not in non_feature_cols]
    # Raw feature columns (intersect only those retained in final_df)
    raw_feature_cols = [col for col in raw_df.columns if col in feature_cols]
    logging.info("Step 11: Getting feature columns complete.")
except Exception as e:
    logging.error(f"Error in getting feature columns from final_df: {e}")
    sys.exit(1)


# Step 12: Prepare IS-normalized version (already done earlier as `df`)
try:
    rsd_is = df[feature_cols].std() / df[feature_cols].mean() * 100
    logging.info("Step 12: IS-normalization complete.")
except Exception as e:
    logging.error(f"Error in IS-normalization: {e}")
    sys.exit(1)


# Step 13: Prepare median-normalized-only version
try:
    median_only_df = raw_df.copy()
    median_only_norm = median_only_df.groupby("Metabolomics_sequence", group_keys=False)[feature_cols].apply(
        lambda g: g / g.median()
    )
    logging.info("Step 13: Median normalization complete.")
except Exception as e:
    logging.error(f"Error in median-normalization: {e}")
    sys.exit(1)


# Step 14: Calculate RSDs (Relative Standard Deviations)
def rsd(df_subset, feature_cols):
    return df_subset[feature_cols].std() / df_subset[feature_cols].mean() * 100

try:
    rsd_raw = rsd(raw_df, raw_feature_cols)
    rsd_median_only = rsd(median_only_norm, feature_cols)
    rsd_final = rsd(final_df, feature_cols)
    logging.info("Step 14: RSDs calculated.")
except Exception as e:
    logging.error(f"Error in RSDs calculation: {e}")
    sys.exit(1)


# Step 15: Combine into one DataFrame
rsd_df = pd.DataFrame({
    "Feature": feature_cols,
    "RSD_Raw": rsd_raw.values,
    "RSD_IS_Normalized": rsd_is.values,
    "RSD_Median_Only": rsd_median_only.values,
    "RSD_IS_and_Median": rsd_final.values
})
logging.info("Step 15: All combined in one dataframe.")


# Step 16: Save RSD values per feature
try:
    rsd_df.to_csv(os.path.join(OUTPUT_DIR, 'RSDs_per_feature.csv'), index=False)
    logging.info("Step 16. RSD values per feature CSV saved.")
except Exception as e:
    logging.error(f"Error in saving RSD values per feature: {e}")


# Step 17: Summary statistics
try:
    summary = rsd_df.drop(columns="Feature").agg(["min", "max", "median"]).T
    summary = summary.rename(columns={
        "min": "Min RSD (%)",
        "max": "Max RSD (%)",
        "median": "Median RSD (%)"
    })
    summary.index.name = "Normalization Step"
    summary.to_csv(os.path.join(OUTPUT_DIR, 'RSD_summary.csv'))
    logging.info("Step 17. Summary statistics saved.")
except Exception as e:
    logging.error(f"Error in summary statistics: {e}")


# Step 18: Visualization of RSDs
def save_fig(fig, fig_name):
    fig.write_html(os.path.join(OUTPUT_DIR, fig_name + ".html"))
    fig.write_image(os.path.join(OUTPUT_DIR, fig_name + ".png"), scale=2, width=800, height=600)
try:
    # Plot RSDs comparison (initial plot)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=rsd_raw.values,
        mode='lines+markers',
        name='Raw',
        marker=dict(symbol='circle', opacity=0.5)
    ))
    fig1.add_trace(go.Scatter(
        y=rsd_is.values,
        mode='lines+markers',
        name='IS Normalized',
        marker=dict(symbol='x', opacity=0.7)
    ))
    fig1.add_trace(go.Scatter(
        y=rsd_final.values,
        mode='lines+markers',
        name='IS + Median Normalized',
        marker=dict(symbol='square', opacity=0.9)
    ))
    fig1.add_trace(go.Scatter(
        y=rsd_median_only.values,
        mode='lines+markers',
        name='Median Normalized (no IS)',
        marker=dict(symbol='diamond', opacity=0.7)
    ))
    fig1.update_layout(
        title="RSD per Feature: Raw vs. Normalizations",
        xaxis_title="Feature Index",
        yaxis_title="RSD (%)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=600
    )
    save_fig(fig1, "RSD_comparison_plot_20percMV")

    # Plot RSDs comparison on separate grids
    fig2 = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "Raw",
            "IS Normalized",
            "Median Normalized",
            "IS + Median Normalized"
        ]
    )
    fig2.add_trace(go.Scatter(
        y=rsd_raw.values,
        mode='lines+markers',
        name='Raw',
        marker=dict(symbol='circle', opacity=0.5)
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        y=rsd_is.values,
        mode='lines+markers',
        name='IS Normalized',
        marker=dict(symbol='x', opacity=0.7)
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        y=rsd_median_only.values,
        mode='lines+markers',
        name='Median Normalized (no IS)',
        marker=dict(symbol='diamond', opacity=0.7)
    ), row=3, col=1)
    fig2.add_trace(go.Scatter(
        y=rsd_final.values,
        mode='lines+markers',
        name='IS + Median Normalized',
        marker=dict(symbol='square', opacity=0.9)
    ), row=4, col=1)
    fig2.update_layout(
        title="RSD per Feature comparison",
        width=700,
        height=1500
    )
    save_fig(fig2, "RSD_comparison_per_norm_20percMV")

    # Scatter-plot of RSDs comparison (separate grids)
    fig3 = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                         subplot_titles=[
                            "IS Normalized vs Raw",
                            "Median Normalized vs Raw",
                            "IS + Median Normalized vs Raw"
                        ])
    comparisons = [
        (rsd_is, "IS Normalized"),
        (rsd_median_only, "Median Normalized (no IS)"),
        (rsd_final, "IS + Median Normalized")
    ]
    for i, (data, label) in enumerate(comparisons, start=1):
        max_val = max(rsd_raw.max(), data.max())
        fig3.add_trace(
            go.Scatter(
                x=rsd_raw,
                y=data,
                mode='markers',
                marker=dict(opacity=0.5, size=6),
                name=label
            ),
            row=1, col=i
        )
        fig3.update_xaxes(title_text="Raw RSD (%)", range=[0, max_val], row=1, col=i)
        fig3.update_yaxes(title_text="Normalized RSD (%)", range=[0, max_val], row=1, col=i)
        fig3.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=False
            ),
            row=1, col=i
        )
    fig3.update_layout(
        title="Feature-wise RSD Comparison After Normalizations",
        template="plotly_white",
        width=1600,
        height=450,
        showlegend=False
    )
    save_fig(fig3, "RSD_scatter_comparison_20percMV")

    # Scatter-plot of RSDs comparison (single grid)
    fig4 = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    labels = ["IS Normalized", "Median Normalized (no IS)", "IS + Median Normalized"]
    datasets = [rsd_is, rsd_median_only, rsd_final]
    max_val = max(rsd_raw.max(), max(d.max() for d in datasets))
    for data, color, label in zip(datasets, colors, labels):
        fig4.add_trace(go.Scatter(
            x=rsd_raw,
            y=data,
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=6),
            name=label
        ))
    fig4.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='y=x'
    ))
    fig4.update_layout(
        title="All Normalizations vs Raw (Feature-wise RSD)",
        xaxis_title="Raw RSD (%)",
        yaxis_title="Normalized RSD (%)",
        template="plotly_white",
        width=600,
        height=600,
        legend=dict(x=0.02, y=0.98)
    )
    save_fig(fig4, "RSD_scatter_all_in_one_20percMV")

    logging.info("Step 18. Visualization of RSDs complete.")
    # fig1.show(); fig2.show(); fig3.show(); fig4.show();
except Exception as e:
    logging.error(f"Error in visualization of RSDs: {e}")


# Step 19: Preview summary of of RSDs per normalization step
logging.info(f"Step 19: Summary of RSDs per normalization step: {summary}")


# Step 20: Finish
logging.info("Step 20. Data preparation completed successfully.")

