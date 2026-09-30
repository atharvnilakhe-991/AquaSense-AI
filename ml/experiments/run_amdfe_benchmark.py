"""
AquaSense AI - AMDFE Benchmark & Ablation Runner
Executes comprehensive SCI/Q1-grade ablation experiments across configurations B0 - B4:
- Spatial Unseen-Well Evaluation (Grouped Holdout)
- Repeated Grouped Validation (Multi-Seed Uncertainty Quantification)
- Temporal Cutoff Holdout
- Scientific Hypothesis Testing (H1 - H5)
- Controlled Robustness & Degradation Stress-Testing
- Paper-Ready Metric Summaries & Effect Size Analysis
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ml.config import RANDOM_STATE, TEST_WELL_RATIO
from ml.models.factory import get_benchmark_models
from ml.training.trainer import train_model
from ml.evaluation.metrics import calculate_regression_metrics
from ml.data.loader import split_by_unseen_wells, split_by_temporal_cutoff

from ml.amdfe.config import (
    AMDFE_EXPERIMENTS_DIR,
    AMDFE_FUSED_DIR,
    ABLATION_CONFIGS,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
)
from ml.amdfe.ingestion import ingest_amdfe_dataset
from ml.amdfe.pipeline import AMDFEPipeline, generate_all_fused_datasets
from ml.amdfe.features import extract_feature_matrix


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive regression and distribution metrics:
    R2, MAE, RMSE, Median AE, P90 AE, P95 AE, Mean Error (Bias), Pearson r.
    """
    base_metrics = calculate_regression_metrics(y_true, y_pred)
    errors = np.asarray(y_true) - np.asarray(y_pred)
    abs_errors = np.abs(errors)

    med_ae = float(np.median(abs_errors))
    p90_ae = float(np.percentile(abs_errors, 90))
    p95_ae = float(np.percentile(abs_errors, 95))
    mean_error = float(np.mean(errors))

    return {
        "R2": base_metrics["R2"],
        "MAE": base_metrics["MAE"],
        "RMSE": base_metrics["RMSE"],
        "Median_AE": med_ae,
        "P90_AE": p90_ae,
        "P95_AE": p95_ae,
        "Mean_Error": mean_error,
        "Pearson_r": base_metrics["Pearson_r"],
    }


def run_single_ablation_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config_key: str,
    seed: int = RANDOM_STATE,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Fits AMDFEPipeline on train_df, transforms train and test for config_key,
    trains all candidate models, and evaluates on test_df.
    """
    pipeline = AMDFEPipeline(aggregation_method="geometric")
    pipeline.fit(train_df, save_artifacts=False)

    fused_train = pipeline.transform(train_df, config_key=config_key)
    fused_test = pipeline.transform(test_df, config_key=config_key)

    X_train, y_train = extract_feature_matrix(fused_train, config_key=config_key)
    X_test, y_test = extract_feature_matrix(fused_test, config_key=config_key)

    models = get_benchmark_models(random_state=seed)
    model_results = {}

    for model_name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        y_pred = trained_model.predict(X_test)
        metrics = compute_extended_metrics(y_test.values, y_pred)
        model_results[model_name] = metrics

    return model_results, fused_test


def run_full_amdfe_ablation_suite(
    seeds: List[int] = [42, 101, 2024, 777, 999],
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes repeated unseen-well spatial holdout across configurations B0 - B4.
    """
    print("=" * 85)
    print("AQUASENSE AI — AMDFE RESEARCH ABLATION & BENCHMARK SUITE")
    print(f"Configurations : {list(ABLATION_CONFIGS.keys())}")
    print(f"Random Seeds   : {seeds}")
    print("=" * 85)

    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=True)
    raw_runs: List[Dict[str, Any]] = []
    weight_records: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n>>> Running Unseen-Well Evaluation with Seed = {seed}...")
        train_df, test_df, train_wells, test_wells = split_by_unseen_wells(
            df_raw, test_size=TEST_WELL_RATIO, random_state=seed
        )

        for cfg_key, cfg_info in ABLATION_CONFIGS.items():
            model_results, fused_test = run_single_ablation_experiment(
                train_df, test_df, config_key=cfg_key, seed=seed
            )

            # Record weight distributions
            if cfg_key == "B4":
                weight_records.append({
                    "Seed": seed,
                    "Mean_w_A": float(fused_test["w_modality_a"].mean()),
                    "Std_w_A": float(fused_test["w_modality_a"].std()),
                    "Mean_w_B": float(fused_test["w_modality_b"].mean()),
                    "Std_w_B": float(fused_test["w_modality_b"].std()),
                    "Mean_w_C": float(fused_test["w_modality_c"].mean()),
                    "Std_w_C": float(fused_test["w_modality_c"].std()),
                    "Fallback_Fraction": float(fused_test["Fallback_Flag"].mean()) if "Fallback_Flag" in fused_test.columns else 0.0,
                })

            for model_name, metrics in model_results.items():
                run_record = {
                    "Seed": seed,
                    "Config_Key": cfg_key,
                    "Config_Name": cfg_info["name"],
                    "Fusion_Type": cfg_info["fusion_type"],
                    "Model": model_name,
                    "Train_Wells": len(train_wells),
                    "Test_Wells": len(test_wells),
                    "Train_Rows": len(train_df),
                    "Test_Rows": len(test_df),
                    **metrics,
                }
                raw_runs.append(run_record)
                print(f"  [{cfg_key} | {model_name:<13}] R²={metrics['R2']:.4f} | MAE={metrics['MAE']:.2f}m | RMSE={metrics['RMSE']:.2f}m | P90={metrics['P90_AE']:.2f}m")

    raw_df = pd.DataFrame(raw_runs)

    # Compute Summary Statistics across repeated runs
    summary_rows = []
    group_cols = ["Config_Key", "Config_Name", "Fusion_Type", "Model"]

    for keys, group in raw_df.groupby(group_cols):
        summary_rows.append({
            "Config_Key": keys[0],
            "Config_Name": keys[1],
            "Fusion_Type": keys[2],
            "Model": keys[3],
            "R2_Mean": float(group["R2"].mean()),
            "R2_Std": float(group["R2"].std()),
            "MAE_Mean": float(group["MAE"].mean()),
            "MAE_Std": float(group["MAE"].std()),
            "RMSE_Mean": float(group["RMSE"].mean()),
            "RMSE_Std": float(group["RMSE"].std()),
            "Median_AE_Mean": float(group["Median_AE"].mean()),
            "Median_AE_Std": float(group["Median_AE"].std()),
            "P90_AE_Mean": float(group["P90_AE"].mean()),
            "P90_AE_Std": float(group["P90_AE"].std()),
            "P95_AE_Mean": float(group["P95_AE"].mean()),
            "P95_AE_Std": float(group["P95_AE"].std()),
            "Mean_Error_Mean": float(group["Mean_Error"].mean()),
            "Pearson_r_Mean": float(group["Pearson_r"].mean()),
            "Num_Seeds": len(group),
            "Test_Wells_Avg": float(group["Test_Wells"].mean()),
            "Test_Rows_Avg": float(group["Test_Rows"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["Model", "Config_Key"]).reset_index(drop=True)

    # Weight Distribution DataFrame
    weights_df = pd.DataFrame(weight_records)

    # Effect size analysis (Cohen's d and Wilcoxon / Paired differences relative to B2 and B0)
    effect_rows = []
    for model_name in summary_df["Model"].unique():
        sub_raw = raw_df[raw_df["Model"] == model_name]
        b0_rmse = sub_raw[sub_raw["Config_Key"] == "B0"].sort_values(by="Seed")["RMSE"].values
        b2_rmse = sub_raw[sub_raw["Config_Key"] == "B2"].sort_values(by="Seed")["RMSE"].values
        b3_rmse = sub_raw[sub_raw["Config_Key"] == "B3"].sort_values(by="Seed")["RMSE"].values
        b4_rmse = sub_raw[sub_raw["Config_Key"] == "B4"].sort_values(by="Seed")["RMSE"].values

        # Delta B4 vs B2 (Adaptive vs Fixed)
        diff_b4_b2 = b2_rmse - b4_rmse  # Positive means B4 reduced RMSE
        diff_b3_b2 = b2_rmse - b3_rmse  # Positive means B3 reduced RMSE
        diff_b4_b0 = b0_rmse - b4_rmse  # Positive means B4 reduced RMSE

        effect_rows.append({
            "Model": model_name,
            "Delta_RMSE_B4_vs_B2_Mean": float(np.mean(diff_b4_b2)),
            "Delta_RMSE_B4_vs_B2_Std": float(np.std(diff_b4_b2)),
            "Delta_RMSE_B3_vs_B2_Mean": float(np.mean(diff_b3_b2)),
            "Delta_RMSE_B4_vs_B0_Mean": float(np.mean(diff_b4_b0)),
        })

    effect_df = pd.DataFrame(effect_rows)

    if save_outputs:
        AMDFE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(AMDFE_EXPERIMENTS_DIR / "amdfe_ablation_raw_runs.csv", index=False)
        summary_df.to_csv(AMDFE_EXPERIMENTS_DIR / "amdfe_ablation_summary.csv", index=False)
        effect_df.to_csv(AMDFE_EXPERIMENTS_DIR / "amdfe_effect_size_summary.csv", index=False)
        weights_df.to_csv(AMDFE_EXPERIMENTS_DIR / "amdfe_weight_distribution.csv", index=False)
        print(f"\n[AMDFE Experiments] Saved all experiment summaries to: {AMDFE_EXPERIMENTS_DIR}")

    return raw_df, summary_df


def run_controlled_robustness_stress_test(
    seed: int = RANDOM_STATE,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Evaluates Hypothesis H5: Controlled data degradation stress-testing.
    Applies synthetic missingness drops to test partition ONLY to assess robustness.
    NEVER contaminates or modifies original observational records.
    """
    print("\n" + "=" * 85)
    print("AQUASENSE AI — AMDFE HYPOTHESIS H5: CONTROLLED ROBUSTNESS STRESS TEST")
    print("=" * 85)

    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=False)
    train_df, test_df, _, _ = split_by_unseen_wells(df_raw, random_state=seed)

    # Compare B2 (Fixed), B3 (Global Reliability), and B4 (Adaptive AMDFE) under degradation
    configs_to_test = ["B2", "B3", "B4"]
    scenarios = [
        ("Clean_Baseline", 0.0, "None"),
        ("Weather_Missing_30Pct", 0.30, "Weather"),
        ("Weather_Missing_60Pct", 0.60, "Weather"),
        ("Groundwater_History_Missing_30Pct", 0.30, "Groundwater"),
        ("Groundwater_History_Missing_60Pct", 0.60, "Groundwater"),
    ]

    robustness_records = []

    for scenario_name, drop_rate, mod_type in scenarios:
        degraded_test = test_df.copy()
        np.random.seed(seed)

        if mod_type == "Weather" and drop_rate > 0:
            # Set weather features to NaN on a random subset
            mask = np.random.rand(len(degraded_test)) < drop_rate
            for col in ["Annual_Temperature_Mean", "Annual_Precipitation_Total", "Annual_Humidity_Mean", "Annual_WindSpeed_Mean", "Annual_SolarRadiation_Mean"]:
                if col in degraded_test.columns:
                    degraded_test.loc[mask, col] = np.nan
            degraded_test.loc[mask, "Weather_Available_At_Prediction"] = 0

        elif mod_type == "Groundwater" and drop_rate > 0:
            # Mask prior groundwater observation state to simulate cold-start degradation
            mask = np.random.rand(len(degraded_test)) < drop_rate
            for col in ["Previous_WatLevel", "Previous2_WatLevel", "Rolling_Mean_3", "Rolling_Std_3", "Historical_Mean"]:
                if col in degraded_test.columns:
                    degraded_test.loc[mask, col] = np.nan
            degraded_test.loc[mask, "Previous_Observation_Count"] = 0

        for cfg_key in configs_to_test:
            model_results, _ = run_single_ablation_experiment(
                train_df, degraded_test, config_key=cfg_key, seed=seed
            )
            for model_name, metrics in model_results.items():
                robustness_records.append({
                    "Scenario": scenario_name,
                    "Degradation_Modality": mod_type,
                    "Drop_Rate": drop_rate,
                    "Config_Key": cfg_key,
                    "Model": model_name,
                    "R2": metrics["R2"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "P90_AE": metrics["P90_AE"],
                })

    robust_df = pd.DataFrame(robustness_records)
    print(robust_df[["Scenario", "Config_Key", "Model", "R2", "MAE", "RMSE"]].to_string(index=False))

    if save_outputs:
        AMDFE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AMDFE_EXPERIMENTS_DIR / "amdfe_robustness_results.csv"
        robust_df.to_csv(out_path, index=False)
        print(f"\n[AMDFE Robustness] Saved robustness stress-test results to: {out_path}")

    return robust_df


def run_all_amdfe_experiments():
    """
    Master runner executing dataset generation, full ablation, and stress testing.
    """
    # 1. Generate canonical fused datasets
    generate_all_fused_datasets(save_to_disk=True)

    # 2. Run multi-seed ablation benchmark
    raw_df, summary_df = run_full_amdfe_ablation_suite(
        seeds=[42, 101, 2024, 777, 999], save_outputs=True
    )

    # 3. Run controlled robustness stress test
    run_controlled_robustness_stress_test(seed=RANDOM_STATE, save_outputs=True)

    print("\n" + "=" * 85)
    print("AMDFE ABLATION SUMMARY TABLE (MEAN OVER REPEATED UNSEEN-WELL HOLDOUTS)")
    print("=" * 85)
    cols_to_print = ["Config_Key", "Config_Name", "Model", "R2_Mean", "MAE_Mean", "RMSE_Mean", "P90_AE_Mean"]
    print(summary_df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    run_all_amdfe_experiments()
