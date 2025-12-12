# %% [markdown]
# # LS‑HRMS Data Preparation (Notebook Version)
# 
# **What it does:**
# 
# - Detects table orientation and transposes if needed
# - Normalizes by internal standard (IS)
# - Removes features with >20% zeros/NaNs
# - Performs batch-wise LOD/2 imputation
# - Does batch-wise median normalization
# - Computes and visualizes RSDs across stages
# - Saves intermediate CSVs, plots, and final outputs to a timestamped folder

# %% [markdown]
# ## Imports and environment variables

# %%
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import cfg
from utils.output import save_figure, create_output_file_name


def main(mode: str, cfg):
    # Get cfg variables
    cfg.data = cfg[mode].get("data", {})
    cfg.preprocessing = cfg[mode].get("preprocessing", {})
    del cfg[mode]

    def get_output_file_name(type: str, description: str, ext: str) -> str:
        return create_output_file_name(
            step="s01prep",
            type=type,
            description=f"{mode}_{description}",
            ext=ext,
            cfg=cfg,
            output_dir=cfg.OUTPUT_DIR
        )

    # Read CSV input file 
    df = pd.read_csv(cfg.FILE_PATH_INPUT + '/' + cfg.data.file_name_input)
    print(f"Input CSV file ({cfg.data.file_name_input}) loaded successfully.")

    # Check if transpose is needed
    sample_cols = [c for c in df.columns if str(c).startswith(cfg.data.sample_prefix)]
    meta_cols   = [c for c in df.columns if c not in sample_cols]

    is_col = str(cfg.preprocessing.is_col)
    need_transpose = (is_col not in df.columns) or (len(sample_cols) > 0)

    if not need_transpose:
        non_feature_cols = ["row ID", "batch"]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        print(f"Got non-feature columns: {non_feature_cols}")

        raw_df = df.copy()
        df.head()
    else:
        print(f"⚠️ Transpose needed.")

    # %% [markdown]
    # ## Orientation check & transpose if needed
    # 
    # Orientation rules:
    # - If IS_COL is present as a column => assume CORRECT (samples as rows)
    # - Else transpose.
    # - If any column starts with SAMPLE_PREFIX (e.g., 'BIO') => assume RP-style and TRANSPOSE.
    # 

    # %%
    if need_transpose:
        id_col = cfg.data.feature_id_colname if cfg.data.feature_id_colname in meta_cols else (meta_cols[0] if meta_cols else df.columns[0])
        print(f"Transpose: using feature id column '{id_col}'. meta_cols={len(meta_cols)}; sample_cols={len(sample_cols)}")

        t = (
            df[[id_col] + sample_cols]
            .set_index(id_col)
            .T
            .reset_index()
            .rename(columns={"index": "row ID"})
        )

        # Ensure required metadata columns exist for downstream steps
        if "batch" not in t.columns:
            t["batch"] = "1a"

        meta_first = ["row ID", "batch"]
        remaining = [c for c in t.columns if c not in meta_first]
        df = t[meta_first + remaining]

        # Canonicalize feature column names to strings
        rename_map = {c: str(c) for c in remaining}
        df = df.rename(columns=rename_map)
        print(f"After transpose: shape={df.shape}; Samples={len(df)}; Features={len(remaining)}")
        if cfg.run.save_intermediate:
            filename = get_output_file_name(type="prep", description="transposed_input_file", ext="csv")
            df.to_csv(filename, index=False)
            print(f"Intermediate result file ({filename}) is saved.")

        non_feature_cols = ["row ID", "batch"]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        print(f"Got non-feature columns: {non_feature_cols}")

        raw_df = df.copy()
        df.head()
    else:
        print("Orientation check: looks correct (IS_COL present and no SAMPLE_PREFIX columns). No transpose performed.")

    # %% [markdown]
    # ## Normalize by internal standard (IS)

    # %%
    try:
        if is_col not in df.columns:
            print(f"ERROR: Internal standard column '{is_col}' not found in input.")
            raise SystemExit(1)
        print(f"Internal standard column '{is_col}' found with {df[is_col].notna().sum()} non-missing values.")
        # Coerce everything to numeric
        # And voiding zero-division by substituting zeros with NaN
        IS_vector = pd.to_numeric(df[is_col], errors="coerce").replace(0, np.nan)
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        df[feature_cols] = df[feature_cols].div(IS_vector, axis=0)
        print(f"✅ IS normalization complete.")
    except Exception as e:
        print(f"ERROR during IS normalization: {e}")

    if cfg.run.save_intermediate:
        filename = get_output_file_name(type="norm", description="IS_normalized", ext="csv")
        df.to_csv(filename, index=False)
        print(f"Intermediate result file ({filename}) is saved.")

    # %% [markdown]
    # ## Prep for imputation & remove features with >20% zeros/NaNs

    # %%
    metadata_df = df[non_feature_cols].copy()
    features_df = df[feature_cols].copy()
    print(f"Copying values from the dataframe complete.")

    try:
        zero_threshold = cfg.preprocessing.zero_threshold if cfg.preprocessing.zero_threshold is not None else 0.2
        zero_fractions = (features_df.eq(0) | features_df.isna()).sum() / len(features_df)
        features_df = features_df.loc[:, zero_fractions <= zero_threshold]
        feature_cols = features_df.columns.tolist()
        print(f"✅ Zero/NaN fraction applied at threshold {zero_threshold}. Kept {features_df.shape[1]} features out of {len(zero_fractions)}.")
    except Exception as e:
        print(f"ERROR during zero/NaN filtering: {e}")

    # %% [markdown]
    # ## Batch-wise LOD/2 imputation

    # %%
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
        features_imputed = features_df.groupby(df["batch"], group_keys=False).apply(impute_lod_half)
        df_imputed = pd.concat([metadata_df.reset_index(drop=True), features_imputed.reset_index(drop=True)], axis=1)
        print(f"✅ Imputation complete.")
    except Exception as e:
        print(f"ERROR during imputation: {e}")

    if cfg.run.save_intermediate:
        filename = get_output_file_name(type="norm", description="normalized_imputed", ext="csv")
        df_imputed.to_csv(filename, index=False)
        print(f"Intermediate result file ({filename}) is saved.")

    try:
        imputation_df = pd.DataFrame.from_dict(imputation_log, orient='index', columns=["Imputed_Count"])
        imputation_df.sort_values("Imputed_Count", ascending=False, inplace=True)

        plt.figure(figsize=(16, 6))
        plt.bar(imputation_df.index, imputation_df["Imputed_Count"])
        plt.xticks(rotation=90)
        plt.ylabel("Number of Imputed Values")
        plt.title("Imputed 0s and NaNs per Feature (LOD/2 Imputation)")
        plt.tight_layout()
        filename = get_output_file_name(type="norm", description="imputed_counts_per_feature", ext="png")
        plt.savefig(filename, dpi=300)
    except Exception as e:
        print(f"ERROR during imputation visualization: {e}")

    # %% [markdown]
    # ## Batch-wise median normalization

    # %%
    def normalize_by_batch_median(group):
        return group[feature_cols] / group[feature_cols].median()

    try:
        normalized_features = df_imputed.groupby("batch", group_keys=False)[feature_cols].apply(normalize_by_batch_median)
        print("✅ Median normalization complete.")
    except Exception as e:
        print(f"ERROR during median normalization: {e}")

    # Replace infinities (just in case)
    normalized_features.replace([float('inf'), -float('inf')], 0, inplace=True)
    final_df = pd.concat([df_imputed[non_feature_cols].reset_index(drop=True),
                        normalized_features.reset_index(drop=True)], axis=1)

    final_df.head(30)

    # %% [markdown]
    # ## Save output 

    # %%
    try:
        output_filename = get_output_file_name(type="output", description="final", ext="csv")
        final_df.to_csv(output_filename, index=False)
        final_df.head()
        print(f"✅ Output is saved successfully.")
    except Exception as e:
        print(f"ERROR in saving output: {e}")
        sys.exit(1)

    # %% [markdown]
    # ## RSD (Relative Standard Deviation) calculation

    # %%
    feature_cols = [col for col in final_df.columns if col not in non_feature_cols]
    raw_feature_cols = [col for col in raw_df.columns if col in feature_cols]

    def rsd(df_subset: pd.DataFrame, feature_cols):
        return df_subset[feature_cols].std() / df_subset[feature_cols].mean() * 100

    try:
        # Raw
        rsd_raw = rsd(raw_df, raw_feature_cols)

        # IS-normalized-only (already 'df')
        rsd_is = df[feature_cols].std() / df[feature_cols].mean() * 100

        # Median-only on raw
        median_only_df = raw_df.copy()
        median_only_norm = median_only_df.groupby("batch", group_keys=False)[feature_cols].apply(
            lambda g: g / g.median()
        )
        rsd_median_only = rsd(median_only_norm, feature_cols)

        # IS + Median (final_df)
        rsd_final = rsd(final_df, feature_cols)

        rsd_df = pd.DataFrame({
            "Feature": feature_cols,
            "RSD_Raw": rsd_raw.values,
            "RSD_IS_Normalized": rsd_is.values,
            "RSD_Median_Only": rsd_median_only.values,
            "RSD_IS_and_Median": rsd_final.values
        })

        filename = get_output_file_name(type="stats", description="RSDs_per_feature", ext="csv")
        rsd_df.to_csv(filename, index=False)

        summary = rsd_df.drop(columns="Feature").agg(["min", "max", "median"]).T
        summary = summary.rename(columns={
            "min": "Min RSD (%)",
            "max": "Max RSD (%)",
            "median": "Median RSD (%)"
        })
        summary.index.name = "Normalization Step"
        filename = get_output_file_name(type="stats", description="RSD_summary", ext="csv")
        summary.to_csv(filename)
        print("✅ RSD calculations complete.")
        summary
    except Exception as e:
        print(f"ERROR during RSD calculations: {e}")

    # %% [markdown]
    # ## Visualisation of RSD calculation results

    # %% [markdown]
    # ### RSDs comparison (initial plot)

    # %%
    try:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=rsd_raw.values,
            text=rsd_raw.index,
            mode='lines+markers',
            name='Raw',
            marker=dict(symbol='circle', opacity=0.5)
        ))
        fig1.add_trace(go.Scatter(
            y=rsd_is.values,
            text=rsd_is.index,
            mode='lines+markers',
            name='IS Normalized',
            marker=dict(symbol='x', opacity=0.7)
        ))
        fig1.add_trace(go.Scatter(
            y=rsd_final.values,
            text=rsd_final.index,
            mode='lines+markers',
            name='IS + Median Normalized',
            marker=dict(symbol='square', opacity=0.9)
        ))
        fig1.add_trace(go.Scatter(
            y=rsd_median_only.values,
            text=rsd_median_only.index,
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
        save_figure(fig1, get_output_file_name(type="fig", description="RSDs_comparison_initial_plot", ext="png"), show=False)
    except Exception as e:
        print(f"ERROR during RSD plotting: {e}")

    # %% [markdown]
    # ### Plot RSDs comparison on separate grids

    # %%
    try:    
        fig2 = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=[
                "Raw",
                "IS Normalized",
                "Median Normalized",
                "IS + Median Normalized"
            ]
        )
        
        def add_rsd_trace(fig, y_vals, feature_ids, name, symbol, row):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(y_vals))),          # X-axis = feature index
                    y=y_vals,
                    text=feature_ids,                    # text shown on hover
                    hovertemplate="Feature: %{text}<br>RSD: %{y}<extra></extra>",
                    mode='lines+markers',
                    name=name,
                    marker=dict(symbol=symbol, opacity=0.7),
                ),
                row=row, col=1
            )

        add_rsd_trace(fig2, rsd_raw.values, rsd_raw.index, "Raw", "circle", 1)
        add_rsd_trace(fig2, rsd_is.values, rsd_is.index, "IS Normalized", "x", 2)
        add_rsd_trace(fig2, rsd_median_only.values, rsd_median_only.index, "Median Normalized", "diamond", 3)
        add_rsd_trace(fig2, rsd_final.values, rsd_final.index, "IS + Median Normalized", "square", 4)

        fig2.update_layout(
            title="RSD per Feature comparison",
            width=700,
            height=1500
        )

        fig2.update_yaxes(title_text="RSD (%)")
        fig2.update_xaxes(title_text="Feature Index")

        save_figure(fig2, get_output_file_name(type="fig", description="RSDs_comparison_separate_grids", ext="png"), show=False)
    except Exception as e:
        print(f"ERROR during RSD subplot plotting: {e}")

    # %% [markdown]
    # ### Scatter-plot of RSDs comparison (separate grids)

    # %%
    try:
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
        save_figure(fig3, get_output_file_name(type="fig", description="RSDs_scatterplot_comparison_separate_grids", ext="png"), show=False)
    except Exception as e:
        print(f"ERROR during RSD scatter-plot comparison: {e}")

    # %% [markdown]
    # ### Scatter-plot of RSDs comparison (single grid)

    # %%
    try:
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
        save_figure(fig4, get_output_file_name(type="fig", description="RSDs_scatterplot_comparison_single_grid", ext="png"), show=False)
    except Exception as e:
        print(f"ERROR during RSD scatter-plot (single grid) comparison: {e}")

    return output_filename


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "anp"
    main(mode, cfg)