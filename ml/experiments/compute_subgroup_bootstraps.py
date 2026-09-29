"""
Compute exact well-clustered bootstrap CIs for Warm-Start C1 vs C2 and Robustness degradation differences.
"""
import numpy as np
import pandas as pd
from ml.amdfe.config import AMDFE_V21_VALIDATION_DIR, WELL_ID_COLUMN
from ml.experiments.run_amdfe_v21_benchmark import cluster_bootstrap_by_well

def compute_subgroup_bootstraps():
    df_preds = pd.read_csv(AMDFE_V21_VALIDATION_DIR / "prediction_logs_v21.csv")
    
    # 1. Warm-Start C1 vs C2 for LightGBM across seeds
    lgb_preds = df_preds[df_preds["Model"] == "LightGBM"]
    warm_preds = lgb_preds[lgb_preds["Start_Type"] == "Warm-Start"]
    
    warm_boot_list = []
    for seed in warm_preds["Seed"].unique():
        sub_s = warm_preds[warm_preds["Seed"] == seed]
        pivoted = sub_s.pivot(index=["Sample_ID", "CSD_ID", "y_true"], columns="Configuration", values="y_pred").reset_index()
        
        boot_warm = cluster_bootstrap_by_well(
            pivoted,
            baseline_pred_col="C1",
            candidate_pred_col="C2",
            target_col="y_true",
            cluster_col="CSD_ID",
            n_bootstraps=1000,
            random_state=seed,
        )
        warm_boot_list.append(boot_warm)
        
    df_warm_boot = pd.DataFrame(warm_boot_list)
    print("=== WARM-START C1 vs C2 (LightGBM) ===")
    print(f"Baseline (C1) RMSE Mean: {df_warm_boot['Baseline_RMSE'].mean():.4f} m")
    print(f"Candidate (C2) RMSE Mean: {df_warm_boot['Candidate_RMSE'].mean():.4f} m")
    print(f"Delta RMSE Mean: {df_warm_boot['Delta_RMSE'].mean():.4f} m")
    print(f"Delta RMSE 95% CI: [{df_warm_boot['Delta_RMSE_CI_Low'].mean():.4f}, {df_warm_boot['Delta_RMSE_CI_High'].mean():.4f}] m")
    print(f"Improvement %: {df_warm_boot['Improvement_RMSE_Pct'].mean():.2f}%")
    print(f"Improvement % 95% CI: [{df_warm_boot['Improvement_RMSE_Pct_CI_Low'].mean():.2f}%, {df_warm_boot['Improvement_RMSE_Pct_CI_High'].mean():.2f}%]")
    print(f"p-value Mean: {df_warm_boot['P_Value_RMSE'].mean():.4f}")
    
    # 2. Long-Gap Cohort (>365d) C0 vs C2
    gap_preds = lgb_preds[lgb_preds["Gap_Cohort"] == "Long_Gap"]
    gap_boot_list = []
    for seed in gap_preds["Seed"].unique():
        sub_s = gap_preds[gap_preds["Seed"] == seed]
        pivoted = sub_s.pivot(index=["Sample_ID", "CSD_ID", "y_true"], columns="Configuration", values="y_pred").reset_index()
        
        boot_gap = cluster_bootstrap_by_well(
            pivoted,
            baseline_pred_col="C0",
            candidate_pred_col="C2",
            target_col="y_true",
            cluster_col="CSD_ID",
            n_bootstraps=1000,
            random_state=seed,
        )
        gap_boot_list.append(boot_gap)
        
    df_gap_boot = pd.DataFrame(gap_boot_list)
    print("\n=== LONG-GAP COHORT (>365d) C0 vs C2 (LightGBM) ===")
    print(f"Baseline (C0) RMSE Mean: {df_gap_boot['Baseline_RMSE'].mean():.4f} m")
    print(f"Candidate (C2) RMSE Mean: {df_gap_boot['Candidate_RMSE'].mean():.4f} m")
    print(f"Delta RMSE Mean: {df_gap_boot['Delta_RMSE'].mean():.4f} m")
    print(f"Delta RMSE 95% CI: [{df_gap_boot['Delta_RMSE_CI_Low'].mean():.4f}, {df_gap_boot['Delta_RMSE_CI_High'].mean():.4f}] m")
    print(f"Improvement %: {df_gap_boot['Improvement_RMSE_Pct'].mean():.2f}%")
    print(f"Improvement % 95% CI: [{df_gap_boot['Improvement_RMSE_Pct_CI_Low'].mean():.2f}%, {df_gap_boot['Improvement_RMSE_Pct_CI_High'].mean():.2f}%]")
    print(f"p-value Mean: {df_gap_boot['P_Value_RMSE'].mean():.4f}")

if __name__ == "__main__":
    compute_subgroup_bootstraps()
