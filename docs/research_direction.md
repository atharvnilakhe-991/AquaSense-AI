# Research Direction & Scientific Methodology: AquaSense AI

**Title:** Observation-Aware Adaptive Multimodal Data Fusion for Hyperlocal Groundwater Prediction Under Irregular and Incomplete Monitoring

## 1. Problem Formulation
Groundwater monitoring wells in agricultural basins (such as Phelps County in the High Plains / Ogallala aquifer, Nebraska) exhibit:
- **Spatial Sparsity:** Well networks are distributed unevenly based on private irrigation permits and municipal installations.
- **Temporal Irregularity:** Measurements are not taken continuously; intervals between field readings range from quarterly to multi-year lapses.
- **Modality Asymmetry:** Weather and climate variables (precipitation, temperature, solar radiation via NASA POWER) are continuous and spatially gridded, whereas groundwater observations are sparse and point-based.

## 2. Adaptive Multimodal Data Fusion Engine (AMDFE)
The core research hypothesis examines whether an **Observation-Aware Adaptive Multimodal Fusion Engine** can dynamically balance data modalities based on source reliability:
- When well observation frequency is high and recency is low ($< 1$ year), historical groundwater features are allocated high predictive weight ($w_{gw} \approx 55\%$).
- When an observation gap occurs ($> 3$ years), well confidence decays exponentially, and AMDFE automatically transfers weight to continuous meteorological and topographic modalities ($w_{meteo} + w_{topo} \approx 75\%$).

## 3. Evaluation Protocol
### A. Unseen-Well GroupKFold Validation
Standard random row splitting introduces artificial data leakage because earlier and later readings from the same well appear in both train and test splits. AquaSense AI evaluates location generalization strictly via:
$$\text{GroupKFold}(K=5, \text{groups} = \text{well\_id})$$
Ensuring that 0% of test well identities exist in the training set.

### B. Baseline vs Model Progression
| Model Architecture | R² (Unseen Wells) | MAE (ft) | RMSE (ft) | Status |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest Baseline | 0.7848 | 14.53 | 22.97 | Baseline |
| XGBoost Regressor | 0.8046 | 12.75 | 21.89 | Evaluated |
| **LightGBM Regressor (AMDFE)** | **0.8107** | **12.75** | **21.55** | **Preferred Candidate** |

## 4. Explainable AI (XAI) with SHAP
To ensure transparency for water management districts, predictions are coupled with SHAP (SHapley Additive exPlanations):
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
Where $\phi_i$ decomposes local water depth into hydrometeorological surplus (negative drawdown) vs topographic and drought stress (positive drawdown).
