import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import probplot, shapiro

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
    logging.info("Step 2: Filtered proteins that have more than %d%% of '<LOQ': kept %d of %d", zero_threshold * 100, len(df_filtered), len(df))
    df = df_filtered.copy()
except Exception as e:
    logging.error(f"Error during zero thresholding: {e}")
    sys.exit(1)


# Step 3: Imputing remaining '<LOQ' values with LOQ/2 for each protein
try:
    for col in value_cols:
        min_detected = df[col].min(skipna=True)
        impute_value = min_detected / 2 if pd.notna(min_detected) else 0
        df.fillna({col: impute_value}, inplace=True)

    logging.info("Step 3: Imputed remaining '<LOQ' values with LOQ/2 for each protein.")
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


# Step 4: Shapiro-Wilk test for normality on each protein across samples
try:
    results = []
    for idx, row in df.iterrows():
        protein_name = row["Protein"] if "Protein" in df.columns else f"Protein_{idx}"
        peptide_name = row["Peptide"] if "Peptide" in df.columns else f"Peptide_{idx}"

        for group, cols in groups.items():
            values = row[cols].astype(float).values
            if len(values) < 3:
                continue

            stat, pval = shapiro(values)
            (osm, osr), (slope, intercept, r) = probplot(values)
            
            results.append({
                "protein": protein_name,
                "peptide": peptide_name,
                "group": group,
                "shapiro_stat": stat,
                "shapiro_pval": pval,
                "qq_slope": slope,
                "qq_intercept": intercept,
                "qq_rvalue": r
            })

    normality_df = pd.DataFrame(results)
    normality_df["is_normal"] = normality_df["shapiro_pval"] > 0.05
    normality_df.sort_values(by=["group", "protein"], inplace=True)
    normality_df.to_csv(os.path.join(OUTPUT_DIR, "proteomics_normality_by_group.csv"), index=False)

    logging.info("Step 4: Completed Shapiro-Wilk tests for each protein within each group.")
    logging.info(normality_df.head(15))
except Exception as e:
    logging.error(f"Error during Shapiro-Wilk test: {e}")
    sys.exit(1)

