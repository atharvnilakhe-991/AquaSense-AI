"""
AquaSense AI - AMDFE Comprehensive Unit & Integration Test Suite
Verifies:
1. Reliability score bounds [0, 1]
2. Modality weights partition of unity (sum to 1.0)
3. Unavailable modality receives zero weight (Modality D = 0)
4. Missingness indicators correctly constructed
5. Strict target leakage exclusion
6. TIFF_Value and raster exclusion
7. Weather temporal precedence (Y_weather < Y_obs)
8. Train-only fitted transformations (invariance under test partition)
9. Deterministic reproducibility
10. Cold-start vs warm-start availability behavior
11. Unseen-well spatial isolation (zero overlap)
"""

import unittest
import numpy as np
import pandas as pd

from ml.config import RANDOM_STATE, TEST_WELL_RATIO
from ml.data.loader import split_by_unseen_wells
from ml.amdfe.config import (
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
)
from ml.amdfe.ingestion import ingest_amdfe_dataset
from ml.amdfe.quality import assess_groundwater_quality, assess_weather_quality, assess_spatial_quality
from ml.amdfe.reliability import (
    compute_global_reliability_scores,
    compute_context_aware_reliability,
    compute_sample_availability,
)
from ml.amdfe.fusion import compute_adaptive_weights
from ml.amdfe.pipeline import AMDFEPipeline
from ml.amdfe.features import extract_feature_matrix


class TestAMDFEEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=False)
        cls.train_df, cls.test_df, cls.train_wells, cls.test_wells = split_by_unseen_wells(
            cls.df_raw, test_size=TEST_WELL_RATIO, random_state=RANDOM_STATE
        )
        cls.pipeline = AMDFEPipeline(aggregation_method="geometric").fit(cls.train_df, save_artifacts=False)
        cls.fused_train = cls.pipeline.transform(cls.train_df, config_key="B4")
        cls.fused_test = cls.pipeline.transform(cls.test_df, config_key="B4")

    def test_01_reliability_score_bounds(self):
        """Verify global and row reliability scores are strictly in [0, 1]."""
        global_scores = compute_global_reliability_scores(self.train_df)
        for mod, score in global_scores.items():
            self.assertGreaterEqual(score, 0.0, f"Global reliability for {mod} < 0")
            self.assertLessEqual(score, 1.0, f"Global reliability for {mod} > 1")

        rel_df = compute_context_aware_reliability(self.test_df, global_scores)
        for mod in ["modality_a", "modality_b", "modality_c", "modality_d"]:
            r_vals = rel_df[f"R_{mod}"].values
            self.assertTrue((r_vals >= 0.0).all(), f"Row reliability for {mod} contains negative values")
            self.assertTrue((r_vals <= 1.0).all(), f"Row reliability for {mod} exceeds 1.0")

    def test_02_weights_sum_to_one(self):
        """Verify normalized adaptive weights w_{i,m} sum to 1.0 for every sample."""
        weights_sum = (
            self.fused_test["w_modality_a"] +
            self.fused_test["w_modality_b"] +
            self.fused_test["w_modality_c"]
        )
        np.testing.assert_allclose(weights_sum.values, 1.0, rtol=1e-5, err_msg="Modality weights do not sum to 1.0!")

    def test_03_unavailable_modality_zero_weight(self):
        """Verify unavailable Modality D receives zero weight and zero reliability."""
        self.assertEqual(self.fused_test["w_modality_d"].sum(), 0.0, "Modality D received non-zero weight!")
        self.assertEqual(self.fused_test["R_modality_d"].sum(), 0.0, "Modality D received non-zero reliability!")

    def test_04_missingness_indicators_correctness(self):
        """Verify missingness indicators are binary and align with original missing values."""
        for col in ["Previous_WatLevel", "Previous2_WatLevel"]:
            indicator_col = f"Missing_{col}"
            self.assertIn(indicator_col, self.fused_test.columns)
            original_missing = self.test_df[col].isnull().astype(int).values
            indicator_vals = self.fused_test[indicator_col].values
            np.testing.assert_array_equal(original_missing, indicator_vals)

    def test_05_target_watlevel_excluded_from_predictors(self):
        """Verify target column WatLevel is strictly absent from predictor matrices."""
        for cfg in ["B0", "B1", "B2", "B3", "B4"]:
            X_tr, _ = extract_feature_matrix(self.fused_train, config_key=cfg)
            X_te, _ = extract_feature_matrix(self.fused_test, config_key=cfg)
            self.assertNotIn(TARGET_COLUMN, X_tr.columns)
            self.assertNotIn(TARGET_COLUMN, X_te.columns)

    def test_06_tiff_and_rasters_excluded(self):
        """Verify TIFF_Value and any target-derived rasters are excluded from all schemas."""
        for leak_col in LEAKAGE_COLUMNS:
            self.assertNotIn(leak_col, self.fused_train.columns)
            self.assertNotIn(leak_col, self.fused_test.columns)

    def test_07_weather_temporal_precedence(self):
        """Verify weather year used is strictly prior to observation year (Y_weather < Y_obs)."""
        diff = self.df_raw[YEAR_COLUMN] - self.df_raw["Weather_Year_Used"]
        self.assertTrue((diff == 1).all(), "Weather year used violates Y-1 lag rule!")

    def test_08_train_fitted_transformations(self):
        """Verify transforming test partition does not mutate fitted training statistics."""
        fitted_medians_before = dict(self.pipeline.imputer.training_medians)
        _ = self.pipeline.transform(self.test_df, config_key="B4")
        fitted_medians_after = dict(self.pipeline.imputer.training_medians)
        self.assertEqual(fitted_medians_before, fitted_medians_after)

    def test_09_deterministic_reproducibility(self):
        """Verify pipeline execution is 100% deterministic given identical inputs."""
        p1 = AMDFEPipeline().fit(self.train_df, save_artifacts=False)
        p2 = AMDFEPipeline().fit(self.train_df, save_artifacts=False)
        out1 = p1.transform(self.test_df, config_key="B4")
        out2 = p2.transform(self.test_df, config_key="B4")
        np.testing.assert_array_equal(out1["Fused_Previous_WatLevel"].values, out2["Fused_Previous_WatLevel"].values)

    def test_10_cold_start_vs_warm_start_availability(self):
        """Verify cold-start well observations receive lower groundwater availability than warm-start."""
        avail_df = compute_sample_availability(self.df_raw)
        cold_mask = (self.df_raw["Previous_Observation_Count"] == 0)
        warm_mask = (self.df_raw["Previous_Observation_Count"] > 3)
        mean_cold_avail = avail_df.loc[cold_mask, "A_modality_a"].mean()
        mean_warm_avail = avail_df.loc[warm_mask, "A_modality_a"].mean()
        self.assertLess(mean_cold_avail, mean_warm_avail, "Cold start availability should be lower than warm start!")

    def test_11_unseen_well_isolation(self):
        """Verify zero overlap between training and testing wells."""
        overlap = set(self.train_wells).intersection(set(self.test_wells))
        self.assertEqual(len(overlap), 0, "Train and test well IDs overlap!")


if __name__ == "__main__":
    unittest.main()
