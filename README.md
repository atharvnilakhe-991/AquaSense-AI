# AquaSense AI

## Adaptive Multimodal AI for Hyperlocal Groundwater Prediction, Recharge Assessment & Decision Support

AquaSense AI is a research-oriented groundwater intelligence platform for hyperlocal groundwater-level prediction, recharge assessment, risk identification, and explainable decision support.

The system combines groundwater observations with hydro-meteorological and environmental information and investigates an **Adaptive Multimodal Data Fusion Engine (AMDFE)** that accounts for data quality, reliability, missingness, and irregular monitoring.

---

## 1. Problem Statement

Groundwater monitoring is often sparse, irregular, incomplete, and spatially uneven. Because wells are not continuously observed, local groundwater changes can be difficult to detect.

AquaSense AI addresses this challenge by combining groundwater and environmental information with machine learning and reliability-aware data fusion.

### Example

If a well has only a few measurements over several years while weather data is continuously available, AquaSense AI aims to:

1. Assess data quality.
2. Estimate source/observation reliability.
3. Adapt the contribution of different data modalities.
4. Handle missing information.
5. Predict groundwater level.
6. Estimate recharge potential.
7. Explain the prediction.
8. Identify wells requiring attention.

---

## 2. Objectives

- Predict groundwater level at hyperlocal well locations.
- Handle sparse, irregular, and incomplete observations.
- Fuse heterogeneous environmental information.
- Develop reliability-aware adaptive multimodal fusion.
- Evaluate generalization to unseen wells.
- Estimate recharge potential.
- Provide explainable AI.
- Generate risk and monitoring-priority indicators.
- Provide an interactive GIS decision-support dashboard.

---

## 3. System Architecture

```text
 Raw Groundwater + Weather + Environmental Data
                         |
                         v
              +----------------------+
              | Member 2 — AMDFE     |
              | Quality / Reliability|
              | Missing Data / Fusion|
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 4 — AI/ML     |
              | Prediction / XAI     |
              | Recharge / Risk      |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 3 — Backend   |
              | Database / REST APIs |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 1 — Frontend  |
              | GIS / Dashboard      |
              +----------------------+
```

---

## 4. Team Responsibilities

| Member | Responsibility | Main Deliverables |
|---|---|---|
| Member 1 | Frontend & GIS | Dashboard, maps, charts, filters, alerts, XAI UI |
| Member 2 | AMDFE & Data Engineering | Quality, reliability, adaptive fusion, missing-data recovery |
| Member 3 | Backend & Database | FastAPI, database, APIs, integration layer |
| Member 4 / Group Leader | AI/ML, XAI & Recharge | ML, validation, error analysis, XAI, recharge, risk |

See the four member Markdown files for detailed task plans.

---

## 5. Technology Stack

### Frontend
- React
- Tailwind CSS
- Leaflet / Mapbox
- Charting library

### Backend
- Python
- FastAPI
- PostgreSQL / Supabase
- Pydantic
- REST APIs

### AI/ML
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- XGBoost
- LightGBM
- SHAP

### Environmental / Geospatial
- Groundwater observations
- NASA POWER weather data
- Surface elevation
- Independently sourced satellite/environmental data where available
- GIS visualization

---

## 6. Current Dataset

The current leakage-safe ML dataset contains:

- 3,844 groundwater observations
- 170 wells
- 2000–2024 weather-aligned period
- Phelps County, Nebraska, USA

Primary predictors include latitude, longitude, surface elevation, annual temperature, annual precipitation, annual humidity, annual wind speed, and annual solar radiation.

### Data Integrity Rule

Groundwater-derived TIFF values must not be treated as independent ML predictors when they were generated from the same groundwater observations used as targets. Such use creates target leakage and can artificially inflate model performance.

---

## 7. Current ML Results

### Unseen-Well Evaluation

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest | 0.7848 | 14.53 | 22.97 |
| XGBoost | 0.8046 | 12.75 | 21.89 |
| LightGBM | **0.8107** | **12.75** | **21.55** |

LightGBM is currently the preferred candidate. These results come from one fixed 80/20 well split, so repeated/grouped validation is required before making a final generalization claim.

### Why Unseen-Well Testing Matters

Random row splits can place observations from the same well in both training and testing, which can make performance appear unrealistically strong. Grouping by well gives a more conservative test of location generalization.

---

## 8. Research Direction

### Proposed Direction

> **Observation-Aware Adaptive Multimodal Data Fusion for Groundwater Prediction Under Irregular and Incomplete Monitoring**

The research investigates whether observation history, reliability, missingness, and adaptive multimodal weighting can improve prediction under realistic monitoring conditions.

### Potential Contributions

1. Observation-aware reliability estimation.
2. Adaptive weighting of heterogeneous environmental modalities.
3. Leakage-safe handling of irregular observations.
4. Missing-data-aware multimodal fusion.
5. Rigorous temporal and unseen-well validation.
6. Explainable prediction linked to environmental factors and reliability.
7. Separate recharge-potential estimation and decision support.

These are research directions, not confirmed novelty claims. Formal literature and prior-art review is required before claiming novelty for publication or patent purposes.

---

## 9. Leakage Prevention

The project follows these safeguards:

- Do not use target-derived groundwater TIFF values as independent predictors.
- History features use only observations before the current target.
- Temporal validation respects time ordering.
- Unseen-well validation keeps test wells separate from training wells.
- Test data is not used for tuning.
- Missing-data recovery must not use future information in temporal prediction scenarios.

---

## 10. Final System Outputs

### Groundwater Prediction
- Predicted groundwater level
- Trend
- Error/uncertainty

### Recharge Assessment
- Recharge score/value
- Low / Moderate / High potential

### Risk Assessment
- Groundwater depletion risk
- Monitoring priority
- High-uncertainty warning

### Explainable AI
- Global feature importance
- Individual prediction explanation
- SHAP contributions

### GIS Dashboard
- Well locations
- Prediction visualization
- Risk visualization
- Recharge visualization
- Charts
- Filters
- Alerts

---

## 11. Repository Structure

```text
AquaSense-AI/
├── frontend/
├── backend/
├── ml/
├── amdf/
├── data/
│   └── README.md
├── docs/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

Do not commit large raw datasets, generated TIFFs, virtual environments, secrets, or unnecessary model binaries unless explicitly required.

---

## 12. Development Workflow

```text
Create feature branch
        ↓
Implement assigned module
        ↓
Test locally
        ↓
Commit
        ↓
Push branch
        ↓
Open Pull Request
        ↓
Group Leader review
        ↓
Merge to main
        ↓
Integration testing
```

Suggested branches:

```text
feature/member1-dashboard-gis
feature/member2-amdfe
feature/member3-backend
feature/member4-ml
feature/member4-xai
feature/member4-recharge
```

Suggested commit prefixes:

```text
feat:
fix:
docs:
test:
refactor:
```

---

## 13. Integration Contract

A typical prediction object:

```json
{
  "well_id": "12345",
  "latitude": 40.35,
  "longitude": -99.42,
  "observed_level": 167.4,
  "predicted_level": 170.2,
  "risk_level": "Moderate",
  "recharge_potential": "High",
  "confidence": 0.87
}
```

The final API contract must be agreed by Members 1–4 and maintained by Member 3.

---

## 14. Development Status

### Completed
- Groundwater dataset inspection
- Weather dataset inspection
- Leakage investigation
- Leakage-safe ML dataset
- Random Forest baseline
- XGBoost evaluation
- LightGBM evaluation
- Temporal evaluation
- Unseen-well evaluation
- Generalization/error analysis
- Initial irregular-observation feature engineering

### In Progress
- Observation-aware ML
- Adaptive multimodal fusion
- Repeated/grouped validation
- XAI
- Uncertainty
- Recharge prediction
- Risk/decision support
- Backend integration
- GIS integration

### Final Phase
- Full integration
- End-to-end testing
- Dashboard polishing
- Research documentation
- IEEE paper preparation
- Formal novelty/prior-art assessment

---

## 15. Team Integration Principle

```text
Member 2 → reliable/fused data → Member 4
Member 4 → prediction/recharge/XAI/risk → Member 3
Member 3 → APIs → Member 1
Member 1 → final decision-support dashboard
```

The Group Leader coordinates the interfaces and ensures each module is tested before final integration.

---

## 16. Disclaimer

AquaSense AI is a research/prototype system. Its predictions and recharge/risk indicators are decision-support outputs and should not replace professional hydrogeological assessment or official groundwater monitoring.
