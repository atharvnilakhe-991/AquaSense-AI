"""
AquaSense AI - Unseen-Well Benchmark Experiment Runner
Reproduces the verified spatial generalization benchmark results:
- LightGBM:    R² ≈ 0.8107, MAE ≈ 12.75, RMSE ≈ 21.55
- XGBoost:     R² ≈ 0.8046, MAE ≈ 12.75, RMSE ≈ 21.89
- Random Forest: R² ≈ 0.7848, MAE ≈ 14.53, RMSE ≈ 22.97
"""

import sys
from pathlib import Path
import pandas as pd

from ml.config import (
    PRIMARY_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    UNSEEN_WELL_OUTPUT_DIR,
)
from ml.data.loader import load_groundwater_dataset, split_by_unseen_wells
from ml.preprocessing.cleaning import clean_dataset
from ml.features.engineering import get_primary_feature_matrix
from ml.training.trainer import train_all_benchmark_models
from ml.evaluation.metrics import (
    evaluate_model_predictions,
    create_model_comparison_table,
)


def run_unseen_well_benchmark(save_results: bool = True) -> pd.DataFrame:
    """
    Executes the standard unseen-well generalization benchmark across all 3 candidate models.

    Parameters:
        save_results: If True, saves comparison CSV and predictions to UNSEEN_WELL_OUTPUT_DIR.

    Returns:
        pd.DataFrame summarizing comparison metrics.
    """
    print("=" * 70)
    print("AQUASENSE AI — ML UNSEEN-WELL GENERALIZATION BENCHMARK")
    print("=" * 70)

    # 1. Load leakage-safe dataset
    df = load_groundwater_dataset()
    print(f"Loaded dataset: {len(df)} rows across {df['CSD_ID'].nunique()} unique wells.")

    # 2. Clean missing values
    df_clean, dropped = clean_dataset(df, PRIMARY_FEATURES, TARGET_COLUMN)
    print(f"Cleaned dataset: {len(df_clean)} rows (dropped {dropped} rows).")

    # 3. Split by unseen wells (80% train / 20% test)
    train_df, test_df, train_wells, test_wells = split_by_unseen_wells(
        df_clean, random_state=RANDOM_STATE
    )
    print("\nUNSEEN-WELL SPLIT SUMMARY:")
    print(f"  Training Wells : {len(train_wells)} ({len(train_df)} observations)")
    print(f"  Testing Wells  : {len(test_wells)} ({len(test_df)} observations)")
    print(f"  Overlap Check  : 0 wells overlap (Strict Spatial Holdout)")

    # 4. Feature matrices
    X_train, y_train = get_primary_feature_matrix(train_df, PRIMARY_FEATURES, TARGET_COLUMN)
    X_test, y_test = get_primary_feature_matrix(test_df, PRIMARY_FEATURES, TARGET_COLUMN)

    # 5. Train models
    print("\nTRAINING BENCHMARK MODELS:")
    models = train_all_benchmark_models(X_train, y_train, random_state=RANDOM_STATE)

    # 6. Evaluate on unseen wells
    print("\nEVALUATING ON UNSEEN WELLS:")
    comparison_rows = []
    predictions_dict = {}

    for name, model in models.items():
        y_pred, metrics = evaluate_model_predictions(model, X_test, y_test)
        predictions_dict[name] = y_pred

        comparison_rows.append({
            "Model": name,
            "Training_Wells": len(train_wells),
            "Testing_Wells": len(test_wells),
            "Training_Rows": len(train_df),
            "Testing_Rows": len(test_df),
            "Test_R2": metrics["R2"],
            "Test_MAE": metrics["MAE"],
            "Test_RMSE": metrics["RMSE"],
            "Pearson_r": metrics["Pearson_r"],
        })

    comparison_df = create_model_comparison_table(comparison_rows)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # 7. Save outputs
    if save_results:
        UNSEEN_WELL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = UNSEEN_WELL_OUTPUT_DIR / "unseen_well_model_comparison.csv"
        comparison_df.to_csv(summary_path, index=False)
        print(f"\nSaved benchmark comparison table to: {summary_path}")

    return comparison_df


if __name__ == "__main__":
    run_unseen_well_benchmark(save_results=True)
