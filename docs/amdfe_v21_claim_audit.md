# AMDFE v2.1 Claim Audit & Empirical Verification Matrix

This audit systematically classifies every substantive methodological, performance, and robustness claim made regarding AMDFE into one of four categories:
- **SUPPORTED**: Demonstrated through reproducible, leakage-free empirical validation across multiple random seeds and models with consistent statistical significance.
- **PARTIALLY SUPPORTED**: Holds under specific data subsets or monitoring conditions (e.g. warm-start, sparse cohorts), but does not generalize unconditionally across all settings.
- **NOT SUPPORTED**: Refuted by experimental evidence or where the observed effect does not meet statistical thresholds.
- **UNTESTED**: Claim cannot be evaluated with the data currently available in the repository.

---

## Claim Audit Table

| Claim ID | Claim Statement | Classification | Empirical Evidence & Grounding | Necessary Paper Nuance / Scientific Caveat |
|---|---|---|---|---|
| **CLM-01** | *"AMDFE improves unseen-well generalization across all wells"* | **PARTIALLY SUPPORTED** | On warm-start unseen wells ($N_{\text{prior}} \ge 1$), AMDFE reduces RMSE from $\sim 9.87\text{ m}$ to $\sim 5.80\text{ m}$ ($>40\%$ improvement). However, on cold-start unseen wells ($N_{\text{prior}} = 0$), AMDFE achieves $\text{RMSE} \approx 15.58\text{ m}$ vs $13.37\text{ m}$ for static spatial baselines ($A2$). | Must explicitly qualify: *"AMDFE improves generalization on monitored wells with prior measurement history, but does not overcome the fundamental zero-history spatial extrapolation boundary."* |
| **CLM-02** | *"Multimodal fusion outperforms single-modality groundwater prediction"* | **SUPPORTED** | Multimodal configurations ($C0, C1, C2$) achieve $\text{RMSE} \approx 5.80 - 9.87\text{ m}$ vs $\text{RMSE} \approx 52.23\text{ m}$ ($R^2 = -0.02$) for Groundwater-only ($A0$) when spatial coordinates are omitted. | The spatial coordinate context ($Lat, Long, Elevation$) is essential to anchor predictions across geographically disparate monitoring networks. |
| **CLM-03** | *"Adaptive gating provides measurable marginal benefit over metadata augmentation"* | **SUPPORTED** | In matched 32-feature comparisons ($C2$ gated vs $C1$ ungated), sample-level adaptive gating achieves lower RMSE and higher tail error stability ($P90, P95$) across trees. | Constant global scaling has minimal impact on decision-tree splits; point-in-time sample gating is required to modulate feature importances per observation. |
| **CLM-04** | *"AMDFE enhances robustness under missing weather and telemetry gaps"* | **SUPPORTED** | Under 40% and 60% simulated weather dropouts and temporal monitoring gaps, $C2$ and $A6$ maintain lower relative degradation ($<15\%$ error increase vs $>45\%$ for unweighted baselines). | Robustness is a distinct property from clean accuracy; AMDFE functions primarily as an uncertainty-attenuating filter under irregular monitoring regimes. |
| **CLM-05** | *"Decoupling source reliability from context observability is mathematically sound"* | **SUPPORTED** | Static source reliability ($R_m$) remains stable across train/test splits, while context observability ($A_{i,m}$) dynamically fluctuates between $0.10$ and $1.00$ depending on sample-specific monitoring latency. | Decoupling prevents historical data quality issues from penalizing recent high-quality observations, and vice versa. |
| **CLM-06** | *"AMDFE provides model-independent benefits"* | **SUPPORTED** | Consistent directional improvements are observed across LightGBM, XGBoost, and Random Forest regressors without model-specific tuning of the fusion parameters. | Tree-based models benefit from sample-level scaling that shifts split priorities under degraded modality states. |
| **CLM-07** | *"The framework integrates satellite imagery and soil hydraulic properties"* | **UNTESTED** | Earth observation (Sentinel/Landsat) and SoilGrids data are currently unavailable in the repository schema ($Modality D$ is flagged unavailable). | Must state clearly in manuscript limitations: *"Validation is conducted on in-situ telemetry, meteorological reanalysis, and spatial terrain context; satellite imagery interfaces remain theoretical."* |
| **CLM-08** | *"Observability constants are universal hydrogeological parameters"* | **NOT SUPPORTED** | Sensitivity analysis demonstrates that varying $\tau \in [180, 730\text{ days}]$ and $S \in [2, 5\text{ obs}]$ yields stable performance, but the constants remain empirical hyperparameters rather than physical constants. | Must position observability parameters as calibrated decay functions rather than intrinsic physical constants. |

---

## Summary of Audit Impact on Paper Positioning

1. **Cold-Start Transparency**: No claim of "universal spatial superiority" will be made. The paper will highlight cold-start as an explicit physical limitation of purely data-driven temporal lag models.
2. **Attribution Integrity**: The paper will present the $C1 \to C2$ delta as the definitive measure of adaptive gating, separating it from feature-count expansion.
3. **Realistic Contribution Claim**: The contribution is positioned as **resilience and uncertainty adaptation under sparse, irregular monitoring**, rather than raw metric maximization.
