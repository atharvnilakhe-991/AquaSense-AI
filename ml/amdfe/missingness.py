"""
AquaSense AI - AMDFE Missing-Data Management & Safe Imputation
Implements train-only fitted imputation with explicit missingness indicators.
Strictly forbids temporal lookahead or test-partition snooping.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ml.amdfe.config import (
    AMDFE_MISSINGNESS_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
)


class LeakageSafeImputer:
    """
    Stateful imputer that fits imputation values strictly on training data
    and generates traceable missingness indicator columns.
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self.feature_imputers: Dict[str, SimpleImputer] = {}
        self.training_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "LeakageSafeImputer":
        """
        Fits imputers on the training partition only.
        """
        for col in feature_cols:
            if col not in train_df.columns:
                continue
            
            imputer = SimpleImputer(strategy=self.strategy)
            imputer.fit(train_df[[col]])
            self.feature_imputers[col] = imputer
            self.training_medians[col] = float(imputer.statistics_[0])

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Imputes missing values using frozen training statistics and adds missingness indicators.
        """
        if not self.is_fitted:
            raise RuntimeError("LeakageSafeImputer must be fitted on training data before transform!")

        data = df.copy()

        for col in feature_cols:
            if col not in data.columns or col not in self.feature_imputers:
                continue

            # Create binary missingness indicator
            indicator_col = f"Missing_{col}"
            data[indicator_col] = data[col].isnull().astype(int)

            # Impute value using training imputer
            imputed_vals = self.feature_imputers[col].transform(data[[col]])
            data[col] = imputed_vals.ravel()

        return data

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Convenience method to fit on train and return transformed train.
        """
        return self.fit(train_df, feature_cols).transform(train_df, feature_cols)


def audit_missingness_and_imputation(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    imputer: Optional[LeakageSafeImputer] = None,
    save_reports: bool = True,
) -> pd.DataFrame:
    """
    Generates missingness summaries and audits imputation values.
    """
    all_features = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
    records = []

    for col in all_features:
        if col not in train_df.columns:
            continue
        
        train_nulls = int(train_df[col].isnull().sum())
        train_pct = float(train_nulls / len(train_df) * 100)
        test_nulls = int(test_df[col].isnull().sum()) if test_df is not None and col in test_df.columns else 0
        test_pct = float(test_nulls / len(test_df) * 100) if test_df is not None and len(test_df) > 0 else 0.0

        impute_val = imputer.training_medians.get(col, np.nan) if imputer is not None else np.nan

        records.append({
            "Feature": col,
            "Train_Missing_Count": train_nulls,
            "Train_Missing_Pct": train_pct,
            "Test_Missing_Count": test_nulls,
            "Test_Missing_Pct": test_pct,
            "Fitted_Train_Impute_Value": impute_val,
        })

    summary_df = pd.DataFrame(records)

    if save_reports:
        AMDFE_MISSINGNESS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(AMDFE_MISSINGNESS_DIR / "missingness_summary.csv", index=False)
        summary_df.to_csv(AMDFE_MISSINGNESS_DIR / "imputation_audit.csv", index=False)
        print(f"[AMDFE Missingness] Saved missingness and imputation audits to: {AMDFE_MISSINGNESS_DIR}")

    return summary_df
