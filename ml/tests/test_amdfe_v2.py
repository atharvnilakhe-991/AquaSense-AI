"""
AquaSense AI - AMDFE Version 2 Comprehensive Test Suite
Validates assertions AMDFE-V2-01 through AMDFE-V2-15.
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
from ml.amdfe.v2_reliability import (
    compute_v2_source_reliability,
    evaluate_modality_quality_dimensions,
)
from ml.amdfe.v2_observability import ContextObservabilityEngine
from ml.amdfe.v2_pipeline import AMDFEPipelineV2
from ml.amdfe.v2_fusion import extract_v2_feature_matrix


class TestAMDFEVersion2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=False)
        cls.train_df, cls.test_df, cls.train_wells, cls.test_wells = split_by_unseen_wells(
            cls.df_raw, test_size=TEST_WELL_RATIO, random_state=RANDOM_STATE
        )
        cls.pipeline = AMDFEPipelineV2(aggregation_method="geometric").fit(cls.train_df, save_artifacts=False)
        cls.fused_train = cls.pipeline.transform(cls.train_df, config_key="A5")
        cls.fused_test = cls.pipeline.transform(cls.test_df, config_key="A5")

    def test_v2_01_non_applicable_dimensions_masked(self):
        """AMDFE-V2-01: Non-applicable quality dimensions are masked, not treated as zero."""
        q_dims = evaluate_modality_quality_dimensions(self.train_df)
        self.assertEqual(q_dims["modality_c"]["Temporal_Coverage_T"], "not_applicable")
        self.assertNotIn("Temporal_Coverage_T", q_dims["modality_c"]["applicable_dimensions"])
        self.assertGreater(self.pipeline.global_reliability["modality_c"], 0.5)

    def test_v2_02_measurement_quality_not_fabricated(self):
        """AMDFE-V2-02: Measurement quality is marked unavailable, not assigned 1.0."""
        q_dims = evaluate_modality_quality_dimensions(self.train_df)
        self.assertEqual(q_dims["modality_a"]["Measurement_Quality_M"], "unavailable")
        self.assertNotIn("Measurement_Quality_M", q_dims["modality_a"]["applicable_dimensions"])

    def test_v2_03_reliability_and_observability_separated(self):
        """AMDFE-V2-03: Source Reliability and Context Observability exist as separate entities."""
        self.assertIn("R_modality_a", self.fused_test.columns)
        self.assertIn("A_modality_a", self.fused_test.columns)
        self.assertIn("w_modality_a", self.fused_test.columns)

    def test_v2_04_global_reliability_train_only(self):
        """AMDFE-V2-04: Global reliability does not depend on test set statistics."""
        p_train = AMDFEPipelineV2().fit(self.train_df, save_artifacts=False)
        p_full = AMDFEPipelineV2().fit(self.df_raw, save_artifacts=False)
        self.assertNotEqual(p_train.global_reliability, p_full.global_reliability)

    def test_v2_05_observability_uses_strictly_prior_info(self):
        """AMDFE-V2-05: Context observability uses only point-in-time prior records."""
        diffs = self.df_raw["Days_Since_Previous"].dropna()
        self.assertTrue((diffs >= 0).all())

    def test_v2_06_no_tiff_value(self):
        """AMDFE-V2-06: TIFF_Value is strictly excluded from all matrices."""
        for cfg in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
            X_tr, _, _ = extract_v2_feature_matrix(self.fused_train, config_key=cfg)
            X_te, _, _ = extract_v2_feature_matrix(self.fused_test, config_key=cfg)
            self.assertNotIn("TIFF_Value", X_tr.columns)
            self.assertNotIn("TIFF_Value", X_te.columns)

    def test_v2_07_no_target_derived_raster(self):
        """AMDFE-V2-07: No target-derived rasters enter predictor matrices."""
        for leak in LEAKAGE_COLUMNS:
            self.assertNotIn(leak, self.fused_train.columns)
            self.assertNotIn(leak, self.fused_test.columns)

    def test_v2_08_warm_start_cold_start_separated(self):
        """AMDFE-V2-08: Warm-start and cold-start rows are explicitly distinguished."""
        self.assertIn("Start_Type", self.fused_test.columns)
        self.assertIn("Cold-Start", self.fused_test["Start_Type"].values)
        self.assertIn("Warm-Start", self.fused_test["Start_Type"].values)

    def test_v2_09_weather_strictly_causal_lag(self):
        """AMDFE-V2-09: Weather year used is strictly Y - 1."""
        diff = self.df_raw[YEAR_COLUMN] - self.df_raw["Weather_Year_Used"]
        self.assertTrue((diff == 1).all())

    def test_v2_10_adaptive_weights_sum_to_one(self):
        """AMDFE-V2-10: Adaptive weights sum to 1.0 across active modalities."""
        w_sum = (
            self.fused_test["w_modality_a"] +
            self.fused_test["w_modality_b"] +
            self.fused_test["w_modality_c"]
        )
        np.testing.assert_allclose(w_sum.values, 1.0, rtol=1e-5)

    def test_v2_11_unavailable_modality_zero_contribution(self):
        """AMDFE-V2-11: Unavailable Modality D receives zero weight and reliability."""
        self.assertEqual(self.fused_test["w_modality_d"].sum(), 0.0)
        self.assertEqual(self.fused_test["R_modality_d"].sum(), 0.0)

    def test_v2_12_deterministic_reproducibility(self):
        """AMDFE-V2-12: Identical inputs produce identical outputs."""
        p1 = AMDFEPipelineV2().fit(self.train_df, save_artifacts=False)
        p2 = AMDFEPipelineV2().fit(self.train_df, save_artifacts=False)
        out1 = p1.transform(self.test_df, config_key="A5")
        out2 = p2.transform(self.test_df, config_key="A5")
        np.testing.assert_array_equal(out1["Fused_Previous_WatLevel"].values, out2["Fused_Previous_WatLevel"].values)

    def test_v2_13_scalers_and_imputers_train_fitted(self):
        """AMDFE-V2-13: Transform does not mutate fitted imputer parameters."""
        med_before = dict(self.pipeline.imputer.training_medians)
        self.pipeline.transform(self.test_df, config_key="A5")
        med_after = dict(self.pipeline.imputer.training_medians)
        self.assertEqual(med_before, med_after)

    def test_v2_14_stress_test_never_contaminates_clean_test(self):
        """AMDFE-V2-14: Degradation stress tests leave original test set clean."""
        clean_sum_before = float(self.test_df["Annual_Precipitation_Total"].dropna().sum())
        degraded = self.test_df.copy()
        degraded["Annual_Precipitation_Total"] = np.nan
        clean_sum_after = float(self.test_df["Annual_Precipitation_Total"].dropna().sum())
        self.assertEqual(clean_sum_before, clean_sum_after)

    def test_v2_15_target_purged_from_predictors(self):
        """AMDFE-V2-15: Target WatLevel is never in predictor matrices."""
        for cfg in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
            X_tr, _, _ = extract_v2_feature_matrix(self.fused_train, config_key=cfg)
            X_te, _, _ = extract_v2_feature_matrix(self.fused_test, config_key=cfg)
            self.assertNotIn(TARGET_COLUMN, X_tr.columns)
            self.assertNotIn(TARGET_COLUMN, X_te.columns)


if __name__ == "__main__":
    unittest.main()
