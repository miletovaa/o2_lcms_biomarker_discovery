import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import probplot, shapiro, f_oneway, kruskal, gaussian_kde
from statsmodels.stats.multitest import multipletests

from utils.config import cfg

pd.set_option('future.no_silent_downcasting', True)

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
    # ! logging_path = os.path.join(OUTPUT_DIR, 'logs')
    # ! os.makedirs(logging_path, exist_ok=True)
    log_handlers = [logging.StreamHandler()]
    # ! log_file_path = os.path.join(logging_path, 'proteomics_analysis.log')
    # ! log_handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(
        level=logging.INFO, 
        handlers=log_handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S'
    )
    logging.info(f"Logging is set up successfully. Starting data preparation.")


# Step 0.4: Loading xlsx file to dataframe
try:
    with pd.ExcelFile(os.path.join(FILE_PATH_INPUT, cfg.data.file_name_proteomics_input)) as xls:
        sheet_names = xls.sheet_names
        final_sheets = [name for name in sheet_names if name.upper().startswith("FINAL")]

        if not final_sheets:
            raise ValueError(f"No sheet starting with 'FINAL' found in {cfg.data.file_name_proteomics_input}. Available sheets: {sheet_names}")
        
        df = pd.read_excel(xls, sheet_name=final_sheets[0])
        logging.info(f"Loaded sheet '{final_sheets[0]}' from {cfg.data.file_name_proteomics_input}")
except Exception as e:
    logging.error(f"Error loading Excel file: {e}")
    sys.exit(1)


# Step 1: Dataframe cleaning and preview
try:
    # Fixing double-row header
    df = df[1:]
    # Forward fill missing values (aka fixing merged cells)
    df.ffill(inplace=True)
    logging.info("Successfully performed forward fill on missing values.")

    sample_prefix_arr = ("BIOPD", "RBD", "CON")
    value_cols = [c for c in df.columns if c.startswith(sample_prefix_arr)]

    groups = {
        "PD": [c for c in value_cols if c.startswith("BIOPD")],
        "RBD": [c for c in value_cols if c.startswith("RBD")],
        "CON": [c for c in value_cols if c.startswith("CON")]
    }

    logging.info("Step 1: Preview of the first 15 rows of the dataframe after forward fill:")
    logging.info(df.head(15))
except Exception as e:
    logging.error(f"Error during forward fill operation: {e}")
    sys.exit(1)


# Step 2: Removing samples with more than specified threshold of zero values (<LOQ; default: 60%)
try:
    zero_threshold = cfg.preprocessing.zero_threshold if cfg.preprocessing.zero_threshold is not None else 0.6

    df[value_cols] = df[value_cols].replace("<LOQ", np.nan).astype(float)

    loq_fraction = df[value_cols].isna().mean(axis=1)

    df_filtered = df.loc[loq_fraction <= zero_threshold].reset_index(drop=True)
    logging.info("Step 2: Filtered peptides that have more than %d%% of '<LOQ': kept %d of %d", zero_threshold * 100, len(df_filtered), len(df))
    df = df_filtered.copy()
except Exception as e:
    logging.error(f"Error during zero thresholding: {e}")
    sys.exit(1)


# Step 3: Imputing remaining '<LOQ' values with LOQ/2 for each peptide
try:
    for col in value_cols:
        min_detected = df[col].min(skipna=True)
        impute_value = min_detected / 2 if pd.notna(min_detected) else 0
        df.fillna({col: impute_value}, inplace=True)

    logging.info("Step 3: Imputed remaining '<LOQ' values with LOQ/2 for each peptide.")
except Exception as e:
    logging.error(f"Error during imputation of '<LOQ' values: {e}")
    sys.exit(1)


# Step 3.1: (Optional) Saving intermediate dataframe to CSV
try:
    if cfg.run.save_intermediate:
        intermediate_csv_path = os.path.join(OUTPUT_DIR, "proteomics_data_preprocessed.csv")
        df.to_csv(intermediate_csv_path, index=False)
        logging.info(f"Step 3.1: Saved intermediate preprocessed dataframe to {intermediate_csv_path}")
except Exception as e:
    logging.error(f"Error saving intermediate CSV: {e}")
    sys.exit(1)


# Step 4: Shapiro-Wilk test for normality on each peptide across samples
def do_shapiro(df, groups):
    results = []
    for _, row in df.iterrows():
        for group, cols in groups.items():
            values = row[cols].astype(float).values
            if len(values) < 3:
                continue

            stat, pval = shapiro(values)
            _, (slope, intercept, r) = probplot(values)
            
            results.append({
                "peptide": row["Peptide"],
                "protein": row["Protein"],
                "group": group,

                "is_normal_shapiro": pval > 0.05,
                "is_normal_qq": r >= 0.990,

                "shapiro_stat": stat,
                "shapiro_pval": pval,

                "qq_slope": slope,
                "qq_intercept": intercept,
                "qq_rvalue": r
            })
    return results

try:
    # * Applied Shapiro-Wilk test on INITIAL peptide data (before log transformation)
    # ? normality_df: peptide, protein, group, is_normal_shapiro, is_normal_qq, 
    # ?               shapiro_stat, shapiro_pval, qq_slope, qq_intercept, qq_rvalue
    normality_df = pd.DataFrame(do_shapiro(df, groups))
    normality_df.sort_values(by=["peptide"], inplace=True) # by=["group", "peptide"]
    normality_df.to_csv(os.path.join(OUTPUT_DIR, "proteomics_normality_by_group.csv"), index=False)

    # * Count percentage of non-normal peptides
    # ? normal: array of peptides and their groups that have is_normal_shapiro == True
    normal = normality_df.loc[normality_df["is_normal_shapiro"], ["peptide", "group"]]
    logging.info("Number of normal group-peptides  (before log transformation): %d out of %d (%.2f%%)", normal.shape[0], normality_df.shape[0], normal.shape[0] / normality_df.shape[0] * 100)

    # * Peptides normal in all 3 groups (by Shapiro)
    normal_all_3 = (
        normality_df.groupby("peptide")
        .agg(n_groups=("group", "nunique"), all_normal=("is_normal_shapiro", "all"))
        .query("n_groups == 3 and all_normal")
        .index.tolist()
    )
    logging.info("Peptides normal in all 3 groups (Shapiro): %d", len(normal_all_3))


    # * Apply log10 transformation and re-test normality
    df_log = df.copy()
    df_log[value_cols] = np.log10(
        df[value_cols].astype(float).clip(lower=1e-12)
    )
    normality_df_log = pd.DataFrame(do_shapiro(df_log, groups))
    normality_df_log.sort_values(by=["peptide", "group"], inplace=True)
    normality_df_log.to_csv(os.path.join(OUTPUT_DIR, "proteomics_log_normality_by_group.csv"), index=False)

    # * Count percentage of non-normal peptides after log transformation
    normal_log = normality_df_log.loc[normality_df_log["is_normal_shapiro"], ["peptide", "group"]]
    logging.info("Number of normal group-peptides (after log transformation): %d out of %d (%.2f%%)", normal_log.shape[0], normality_df_log.shape[0], normal_log.shape[0] / normality_df_log.shape[0] * 100)

    # * Peptides normal in all 3 groups (by Shapiro)
    normal_all_3 = (
        normality_df_log.groupby("peptide")
        .agg(n_groups=("group", "nunique"), all_normal=("is_normal_shapiro", "all"))
        .query("n_groups == 3 and all_normal")
        .index.tolist()
    )
    logging.info("Peptides normal in all 3 groups (Shapiro): %d", len(normal_all_3))


    logging.info("Step 4: Completed Shapiro-Wilk tests for each peptide within each group.")
except Exception as e:
    logging.error(f"Error during Shapiro-Wilk test: {e}")
    sys.exit(1)


# Step 5: Doing statistical tests between groups
try:
    # TODO: condition for choosing test based on normality results
    # * Using Kruskal-Wallis test for all peptides (non-parametric)
    kw_results = []
    group_names = list(groups.keys())
    for _, row in df.iterrows():
        values_by_group = [row[groups[g]].astype(float).values for g in group_names]
        stat, pval = kruskal(*values_by_group)
        kw_results.append({
            "peptide": row["Peptide"],
            "protein": row["Protein"],
            "kw_stat": stat,
            "kw_pval": pval,
            "is_significant": pval < 0.05,
        })

    kw_df = pd.DataFrame(kw_results)
    kw_df.sort_values(by=["is_significant", "kw_pval", "peptide"], ascending=[False, True, True], inplace=True)
    kw_df.to_csv(os.path.join(OUTPUT_DIR, "proteomics_kruskal_by_peptide.csv"), index=False)


    # * Using ANOVA for peptides normal in all 3 groups
    anova_results = []
    for peptide in normal_all_3:
        row = df.loc[df["Peptide"] == peptide].iloc[0]
        values_by_group = [row[groups[g]].astype(float).values for g in group_names]
        stat, pval = f_oneway(*values_by_group)
        anova_results.append({
            "peptide": row["Peptide"],
            "protein": row["Protein"],
            "anova_stat": stat,
            "anova_pval": pval,
            "is_significant": pval < 0.05,
        })
    anova_df = pd.DataFrame(anova_results)
    anova_df.sort_values(by=["is_significant", "anova_pval", "peptide"], ascending=[False, True, True], inplace=True)
    anova_df.to_csv(os.path.join(OUTPUT_DIR, "proteomics_anova_by_peptide.csv"), index=False)

    logging.info("Step 5: Statistical tests between groups done.")    
except Exception as e:
    logging.error(f"Error during statistical tests: {e}")
    sys.exit(1)


# Step 6: Applying Bonferroni and Benjamini-Hochberg correction and detecting significant peptides
try:
    sig_kw = set(kw_df.loc[kw_df["is_significant"], "peptide"])
    sig_anova = set(anova_df.loc[anova_df["is_significant"], "peptide"])
    sig_peptides = sorted(sig_kw.union(sig_anova))
    n_sig = len(sig_peptides)

    anova_pvals = anova_df["anova_pval"].values

    _, anova_pvals_bonf, _, _ = multipletests(anova_pvals, method="bonferroni")
    _, anova_pvals_bh, _, _    = multipletests(anova_pvals, method="fdr_bh")

    anova_df["anova_pval_bonf"] = anova_pvals_bonf
    anova_df["anova_pval_fdr_bh"] = anova_pvals_bh

    anova_df["is_significant_bonf"] = anova_df["anova_pval_bonf"] < 0.05
    anova_df["is_significant_fdr"]  = anova_df["anova_pval_fdr_bh"] < 0.05

    sig_anova_fdr = set(anova_df.loc[anova_df["is_significant_fdr"], "peptide"])
    sig_peptides_fdr = sorted(sig_anova_fdr)

    sig_anova_bonf = set(anova_df.loc[anova_df["is_significant_bonf"], "peptide"])
    sig_peptides_bonf = sorted(sig_anova_bonf)

    if n_sig == 0:
        logging.info("Step 7: No significant peptides found. Skipping visualizations.")
        sys.exit(1)
    logging.info("Step 6: %d unique significant peptides from Kruskal/ANOVA after Bonferroni correction.", len(sig_peptides_fdr))
    logging.info("Step 6: %d unique significant peptides from Kruskal/ANOVA after FDR correction.", len(sig_peptides_bonf))
    logging.info("Step 6: %d unique significant peptides from Kruskal/ANOVA.", n_sig)
except Exception as e:
    logging.error(f"Error during corrections: {e}")
    sys.exit(1)


# Step 7: Visualizations of the groups of peptides that are significant
def save_fig(fig, fig_name, width, height):
    output = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(output, exist_ok=True)

    html_path = os.path.join(output, fig_name + ".html")
    fig.write_html(html_path)
    try:
        fig.write_image(os.path.join(output, fig_name + ".png"),
                        scale=2, width=width, height=height)
    except Exception as e:
        logging.warning("Could not save PNG for %s: %s", fig_name, e)
try:
    # Step 7: Visualization of significant peptides
    # 7.1: Get distinct significant peptides from Kruskal and ANOVA
    group_colors = {
        "PD":  "#636EFA",   # blue
        "RBD": "#EF553B",   # red
        "CON": "#00CC96",   # green
    }

    group_names = list(groups.keys())
    
    # 7.2 Box plots
    fig_box = make_subplots(
        rows=1,
        cols=len(sig_peptides_fdr),
        shared_yaxes=False,
        subplot_titles=sig_peptides_fdr
    )
    for col_idx, peptide in enumerate(sig_peptides_fdr, start=1):
        row_pep = df.loc[df["Peptide"] == peptide].iloc[0]
        for g in group_names:
            values = row_pep[groups[g]].astype(float).values
            fig_box.add_trace(
                go.Box(
                    y=values,
                    name=g,
                    boxpoints="outliers",
                    showlegend=(col_idx == 1),
                    marker_color=group_colors.get(g),
                ),
                row=1,
                col=col_idx
            )
    width_box = max(400 * len(sig_peptides_fdr), 800)
    height_box = 600
    fig_box.update_layout(
        title="Significant Peptides – Box Plots per Peptide (3 groups each)",
        width=width_box,
        height=height_box
    )
    save_fig(fig_box, "significant_peptides_boxplots", width_box, height_box)
    logging.info("Step 7.2: Box plots saved.")

    # 7.3 Histograms: median bars
    fig_hist = make_subplots(
        rows=1,
        cols=len(sig_peptides_fdr),
        shared_yaxes=False,
        subplot_titles=sig_peptides_fdr
    )
    for col_idx, peptide in enumerate(sig_peptides_fdr, start=1):
        row_pep = df.loc[df["Peptide"] == peptide].iloc[0]
        for g in group_names:
            values = row_pep[groups[g]].astype(float).values
            values = values[~np.isnan(values)]
            if len(values) == 0:
                median_val = np.nan
            else:
                median_val = float(np.median(values))
            fig_hist.add_trace(
                go.Bar(
                    x=[g],
                    y=[median_val],
                    name=g,
                    marker_color=group_colors.get(g),
                    showlegend=(col_idx == 1)
                ),
                row=1,
                col=col_idx
            )
    fig_hist.update_layout(
        barmode="group"
    )
    width_hist = max(400 * len(sig_peptides_fdr), 800)
    height_hist = 600
    fig_hist.update_layout(
        title="Significant Peptides – Group Medians per Peptide (3 bars each)",
        width=width_hist,
        height=height_hist
    )
    save_fig(fig_hist, "significant_peptides_histograms", width_hist, height_hist)
    logging.info("Step 7.3: Histograms (median bars) saved.")

    # 7.4 Distributions
    fig_dist = make_subplots(
        rows=1,
        cols=len(sig_peptides_fdr),
        shared_yaxes=False,
        subplot_titles= sig_peptides_fdr
    )
    for col_idx, peptide in enumerate( sig_peptides_fdr, start=1):
        row_pep = df.loc[df["Peptide"] == peptide].iloc[0]

        all_vals = []
        group_vals = {}
        for g in group_names:
            vals = row_pep[groups[g]].astype(float).values
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            group_vals[g] = vals
            all_vals.append(vals)
        if not all_vals:
            continue
        all_vals = np.concatenate(all_vals)
        x_min, x_max = float(all_vals.min()), float(all_vals.max())
        if x_min == x_max:
            x_min -= 1e-6
            x_max += 1e-6
        xs = np.linspace(x_min, x_max, 200)

        for g in group_names:
            if g not in group_vals:
                continue
            vals = group_vals[g]
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals)
            ys = kde(xs)
            fig_dist.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=g,
                    showlegend=(col_idx == 1),
                    line=dict(color=group_colors.get(g)),
                ),
                row=1,
                col=col_idx
            )
    width_dist = max(400 * len(sig_peptides_fdr), 800)
    height_dist = 500
    fig_dist.update_layout(
        title="Significant Peptides – Density Distributions per Peptide (3 groups each)",
        width=width_dist,
        height=height_dist
    )
    save_fig(fig_dist, "significant_peptides_distributions", width_dist, height_dist)
    logging.info("Step 7.4: Distribution plots saved.")

    logging.info("Step 7: Visualization of significant peptides completed.")
except Exception as e:
    logging.error(f"Error during visualizations: {e}")
    sys.exit(1)
