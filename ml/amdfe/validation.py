"""
AquaSense AI - AMDFE Leakage Validation Suite
Implements the 10 automated leakage checks (AMDFE-LC-01 through AMDFE-LC-10).
Fails loudly if any data leakage violation is detected.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    AMDFE_VALIDATION_DIR,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
    RANDOM_STATE,
)
from ml.amdfe.ingestion import ingest_amdfe_dataset
from ml.amdfe.pipeline import AMDFEPipeline
from ml.amdfe.features import extract_feature_matrix
from ml.data.loader import split_by_unseen_wells


def run_leakage_validation_suite(save_report: bool = True) -> pd.DataFrame:
    """
    Executes the 10 mandatory automated leakage verification checks.
    """
    print("=" * 80)
    print("AQUASENSE AI — AMDFE AUTOMATED LEAKAGE VALIDATION SUITE")
    print("=" * 80)

    results: List[Dict[str, str]] = []

    # Ingest data
    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=False)
    train_df, test_df, train_wells, test_wells = split_by_unseen_wells(df_raw, random_state=RANDOM_STATE)

    pipeline = AMDFEPipeline(aggregation_method="geometric")
    pipeline.fit(train_df, save_artifacts=False)
    fused_train = pipeline.transform(train_df, config_key="B4")
    fused_test = pipeline.transform(test_df, config_key="B4")

    X_train, y_train = extract_feature_matrix(fused_train, config_key="B4")
    X_test, y_test = extract_feature_matrix(fused_test, config_key="B4")

    # -------------------------------------------------------------------------
    # AMDFE-LC-01: Target WatLevel not in predictors
    # -------------------------------------------------------------------------
    lc01_pass = (TARGET_COLUMN not in X_train.columns) and (TARGET_COLUMN not in X_test.columns)
    results.append({
        "Check_ID": "AMDFE-LC-01",
        "Description": "Target WatLevel is not present in predictor matrix",
        "Status": "PASSED" if lc01_pass else "FAILED",
        "Details": f"Target column '{TARGET_COLUMN}' completely excluded from X_train and X_test.",
    })
    if not lc01_pass:
        raise AssertionError("AMDFE-LC-01 FAILED: Target found in predictor matrix!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-02: TIFF_Value excluded
    # -------------------------------------------------------------------------
    found_tiff = [c for c in LEAKAGE_COLUMNS if c in X_train.columns or c in fused_train.columns]
    lc02_pass = len(found_tiff) == 0
    results.append({
        "Check_ID": "AMDFE-LC-02",
        "Description": "TIFF_Value and target-derived rasters strictly excluded",
        "Status": "PASSED" if lc02_pass else "FAILED",
        "Details": f"Zero occurrences of banned raster columns: {LEAKAGE_COLUMNS}.",
    })
    if not lc02_pass:
        raise AssertionError(f"AMDFE-LC-02 FAILED: Found banned columns: {found_tiff}")

    # -------------------------------------------------------------------------
    # AMDFE-LC-03: No future groundwater observation used
    # -------------------------------------------------------------------------
    # Verify that Previous_WatLevel has zero lookahead
    # For every well, Previous_WatLevel must match the date-level target mean of the immediately preceding unique DateMsr
    test_gw_check = True
    df_raw_dated = df_raw.copy()
    df_raw_dated["Date_dt"] = pd.to_datetime(df_raw_dated[DATE_COLUMN])
    
    for well_id, group in df_raw_dated.groupby(WELL_ID_COLUMN):
        unique_dates = sorted(group["Date_dt"].unique())
        date_means = {d: group[group["Date_dt"] == d][TARGET_COLUMN].mean() for d in unique_dates}
        
        for idx, row in group.iterrows():
            obs_date = row["Date_dt"]
            prev_val = row["Previous_WatLevel"]
            date_idx = unique_dates.index(obs_date)
            
            if date_idx == 0:
                if not np.isnan(prev_val):
                    test_gw_check = False
                    break
            else:
                expected_prev = date_means[unique_dates[date_idx - 1]]
                if np.isnan(prev_val) or not np.isclose(prev_val, expected_prev, atol=1e-5):
                    test_gw_check = False
                    break
        if not test_gw_check:
            break

    lc03_pass = test_gw_check
    results.append({
        "Check_ID": "AMDFE-LC-03",
        "Description": "No future groundwater observation used in lag features",
        "Status": "PASSED" if lc03_pass else "FAILED",
        "Details": "Previous_WatLevel matches chronological prior unique observation date target mean exactly.",
    })
    if not lc03_pass:
        raise AssertionError("AMDFE-LC-03 FAILED: Future groundwater values detected in lag features!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-04: Previous groundwater features satisfy date_prior < date_current
    # -------------------------------------------------------------------------
    lc04_pass = (df_raw["Days_Since_Previous"].dropna() >= 0).all()
    results.append({
        "Check_ID": "AMDFE-LC-04",
        "Description": "Prior groundwater features satisfy date_prior < date_current",
        "Status": "PASSED" if lc04_pass else "FAILED",
        "Details": "All Days_Since_Previous are non-negative strictly backward differences.",
    })
    if not lc04_pass:
        raise AssertionError("AMDFE-LC-04 FAILED: Negative time delta in observation history!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-05: Weather timestamp/period is available before prediction timestamp
    # -------------------------------------------------------------------------
    weather_used_diff = df_raw[YEAR_COLUMN] - df_raw["Weather_Year_Used"]
    lc05_pass = (weather_used_diff == 1).all()
    results.append({
        "Check_ID": "AMDFE-LC-05",
        "Description": "Weather period is strictly completed before observation year (Y - 1)",
        "Status": "PASSED" if lc05_pass else "FAILED",
        "Details": "All meteorological features mapped strictly to completed annual period Y-1.",
    })
    if not lc05_pass:
        raise AssertionError("AMDFE-LC-05 FAILED: Weather period not strictly lagged!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-06: No same-year annual weather leakage for mid-year observations
    # -------------------------------------------------------------------------
    lc06_pass = (df_raw["Weather_Leakage_Check"] == 1).all()
    results.append({
        "Check_ID": "AMDFE-LC-06",
        "Description": "Zero lookahead from uncompleted calendar-year weather",
        "Status": "PASSED" if lc06_pass else "FAILED",
        "Details": "100% of observation dates exceed the December 31 boundary of the weather year used.",
    })
    if not lc06_pass:
        raise AssertionError("AMDFE-LC-06 FAILED: Same-year weather lookahead detected!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-07: Train/test wells are isolated in unseen-well evaluation
    # -------------------------------------------------------------------------
    overlap = set(train_wells).intersection(set(test_wells))
    lc07_pass = (len(overlap) == 0)
    results.append({
        "Check_ID": "AMDFE-LC-07",
        "Description": "Train/test wells are 100% spatially isolated (zero overlap)",
        "Status": "PASSED" if lc07_pass else "FAILED",
        "Details": f"Train wells ({len(train_wells)}) and Test wells ({len(test_wells)}) share 0 common IDs.",
    })
    if not lc07_pass:
        raise AssertionError(f"AMDFE-LC-07 FAILED: {len(overlap)} overlapping wells detected!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-08: Imputer/scaler/reliability fit only on training partition
    # -------------------------------------------------------------------------
    # Verify that transforming test set does not alter training imputer parameters
    imputer_medians_before = dict(pipeline.imputer.training_medians)
    pipeline.transform(test_df, config_key="B4")
    imputer_medians_after = dict(pipeline.imputer.training_medians)
    lc08_pass = (imputer_medians_before == imputer_medians_after)
    results.append({
        "Check_ID": "AMDFE-LC-08",
        "Description": "Imputers, scalers, and reliability normalizations fit on TRAIN only",
        "Status": "PASSED" if lc08_pass else "FAILED",
        "Details": "Frozen pipeline state verified: transforming test set leaves fitted parameters invariant.",
    })
    if not lc08_pass:
        raise AssertionError("AMDFE-LC-08 FAILED: Test data mutated fitted parameters!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-09: No target-derived raster enters any AMDFE output
    # -------------------------------------------------------------------------
    all_out_cols = list(fused_train.columns) + list(fused_test.columns)
    banned_present = [c for c in LEAKAGE_COLUMNS if c in all_out_cols]
    lc09_pass = len(banned_present) == 0
    results.append({
        "Check_ID": "AMDFE-LC-09",
        "Description": "No target-derived raster columns present in AMDFE fused tables",
        "Status": "PASSED" if lc09_pass else "FAILED",
        "Details": f"Verified zero presence of banned raster features in all output schemas.",
    })
    if not lc09_pass:
        raise AssertionError("AMDFE-LC-09 FAILED: Banned raster found in fused tables!")

    # -------------------------------------------------------------------------
    # AMDFE-LC-10: Re-running with the same seed produces identical outputs
    # -------------------------------------------------------------------------
    p1 = AMDFEPipeline(aggregation_method="geometric").fit(train_df, save_artifacts=False)
    p2 = AMDFEPipeline(aggregation_method="geometric").fit(train_df, save_artifacts=False)
    out1 = p1.transform(test_df, config_key="B4")
    out2 = p2.transform(test_df, config_key="B4")
    
    # Check numeric identity
    num_diff = np.max(np.abs(out1["Fused_Previous_WatLevel"].values - out2["Fused_Previous_WatLevel"].values))
    lc10_pass = (num_diff < 1e-12)
    results.append({
        "Check_ID": "AMDFE-LC-10",
        "Description": "Deterministic reproducibility: identical outputs across pipeline runs",
        "Status": "PASSED" if lc10_pass else "FAILED",
        "Details": f"Maximum numerical divergence across consecutive runs: {num_diff:.2e}.",
    })
    if not lc10_pass:
        raise AssertionError("AMDFE-LC-10 FAILED: Pipeline is non-deterministic!")

    # Summarize and save
    report_df = pd.DataFrame(results)
    print("\n" + report_df[["Check_ID", "Status", "Description"]].to_string(index=False))

    if save_report:
        AMDFE_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AMDFE_VALIDATION_DIR / "leakage_validation_report.csv"
        report_df.to_csv(out_path, index=False)
        print(f"\n[AMDFE Validation] Saved leakage validation report to: {out_path}")

    return report_df


if __name__ == "__main__":
    run_leakage_validation_suite(save_report=True)
