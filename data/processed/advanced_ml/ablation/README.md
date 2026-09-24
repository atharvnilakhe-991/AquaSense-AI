# Phase 08.7: Feature Ablation Study Report

## Executive Summary
Phase 08.7 quantifies the incremental predictive contribution of geographic, elevation, environmental, observation-history, and observation-quality feature blocks under repeated grouped-well holdout (10 seeds, 150 fits, 450 raw metric rows) and spatial cluster holdout (5 folds, 75 fits, 225 raw metric rows), totaling 225 model fits and 675 raw metric rows.

### Key Incremental Findings:
1. **A0 $\to$ A1 (Incremental Elevation Contribution)**:
   - Adding `Surf_Elev` to planar coordinates (`LatDD`, `LongDD`) systematically reduces MAE across unseen wells.
   - For Random Forest (Warm-Start), MAE decreases from 21.473 ft to 14.544 ft (Relative Change: -29.65%).

2. **A1 $\to$ A2 (Incremental Climate/Weather Contribution)**:
   - Incorporating 5 annual PRISM climate covariates (`Annual_Temperature_Mean`, `Annual_Precipitation_Total`, `Annual_Humidity_Mean`, `Annual_WindSpeed_Mean`, `Annual_SolarRadiation_Mean`) produces marginal error changes across static well-level spatial holdouts (-13.01% for XGBoost).

3. **A2 $\to$ A3 (Incremental Observation History Block Contribution)**:
   - Adding the 12-feature observation-history block as a whole provides the dominant error collapse across all models.
   - For XGBoost (Warm-Start), MAE collapses from 15.808 ft to 3.023 ft, achieving a relative error reduction of -80.23% ($\Delta R^2 = +0.1294$).

4. **A3 $\to$ A4 (Incremental Observation Quality Proxy Block Contribution)**:
   - Adding monitoring density and gap proxies (`Previous_Observation_Count`, `Observation_Density`, `Long_Gap_Flag`, `Very_Long_Gap_Flag`) produces minor marginal adjustments (1.80% for XGBoost).

5. **Phase 08.6 Cross-Phase Replication Audit**:
   - Configurations A2, A3, and A4 replicated Phase 08.6 Models A, B, and C with exact numerical equivalence (maximum difference < 1e-4).

---

## Directory Contents
- `ablation_summary.csv`: Aggregated performance metrics across 10 repeated grouped-well splits for A0–A4.
- `ablation_raw_runs.csv`: Full 450-row record of individual runs across 150 fits.
- `paired_ablation_comparison.csv`: Stepwise paired deltas for A0 $\to$ A1, A1 $\to$ A2, A2 $\to$ A3, A3 $\to$ A4.
- `spatial_ablation_summary.csv`: Aggregated performance metrics across 5 spatial cluster folds for A0–A4.
- `spatial_ablation_raw_runs.csv`: Full 225-row record of individual spatial fold runs across 75 fits.
- `phase_08_6_replication_audit.csv`: Cross-phase verification log confirming exact replication of Phase 08.6.
- `leakage_validation_report.csv`: Verification log of all 10 automated leakage checks.
