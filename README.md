# AquaSense AI
### Adaptive Multimodal AI for Hyperlocal Groundwater Prediction, Recharge Assessment & Decision Support

AquaSense AI is a research-oriented groundwater intelligence platform for hyperlocal groundwater-level prediction, recharge assessment, risk identification, and explainable decision support (XAI).

The system combines groundwater observations with hydro-meteorological and environmental information and investigates an **Adaptive Multimodal Data Fusion Engine (AMDFE)** that accounts for data quality, reliability, missingness, and irregular monitoring.

---

## 1. System Architecture & Team Handoffs

```
 Raw Groundwater + Weather + Environmental Data
                         |
                         v
              +----------------------+
              | Member 2 — AMDFE     |  (amdf/quality.py, reliability.py, fusion.py)
              | Quality / Reliability|
              | Missing Data / Fusion|
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 4 — AI/ML     |  (ml/models.py, evaluate.py, recharge.py, xai.py)
              | Prediction / XAI     |
              | Recharge / Risk      |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 3 — Backend   |  (backend/main.py, routes/, database.py, schemas.py)
              | Database / REST APIs |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Member 1 — Frontend  |  (frontend/index.html, GIS Map, Hydrographs, XAI UI)
              | GIS / Dashboard      |
              +----------------------+
```

### Team Responsibilities & Module Map

| Member | Responsibility | Main Modules & Deliverables |
| :--- | :--- | :--- |
| **Member 1** | **Frontend & GIS** | `frontend/index.html`, `frontend/src/` — Interactive Leaflet GIS map, Hydrograph trends, SHAP attribution waterfall, filter panels, critical alerts |
| **Member 2** | **AMDFE & Data Engineering** | `amdf/quality.py`, `amdf/reliability.py`, `amdf/fusion.py` — Quality checks, anomaly filtering, observation irregularity metrics, adaptive modality weighting |
| **Member 3** | **Backend & Database** | `backend/main.py`, `backend/routes/`, `backend/schemas.py`, `backend/database.py` — FastAPI REST services, Pydantic v2 schemas, Section 13 contract |
| **Member 4 / Leader** | **AI/ML, XAI & Recharge** | `ml/dataset.py`, `ml/models.py`, `ml/evaluate.py`, `ml/recharge.py`, `ml/risk.py`, `ml/explainability.py` — LightGBM/XGB/RF models, GroupKFold unseen-well cross-validation, SHAP TreeExplainer, recharge score, depletion velocity |

---

## 2. Phelps County, Nebraska Dataset (ML Benchmark)

- **Total Observations:** 3,844 records
- **Spatial Coverage:** 170 wells in Phelps County, Nebraska, USA
- **Temporal Alignment:** 2000–2024
- **Primary Predictors:** Latitude, Longitude, Surface Elevation, NASA POWER Annual Temperature, Precipitation, Relative Humidity, Wind Speed, Solar Radiation, and AMDFE Reliability factors.

### ⚠️ Strict Data Integrity & Leakage Prevention Rules
1. **Zero Target-Derived TIFF Predictors:** Groundwater-derived interpolation TIFFs (IDW/Kriging) are strictly barred as independent ML predictors to prevent target leakage.
2. **Grouped Unseen-Well Validation:** Cross-validation is partitioned strictly by `well_id` (`GroupKFold`) so test wells have 0% exposure during training.
3. **Causal Lags Only:** Historical features strictly use observations prior to the target timestamp ($t_{prev} < t_{target}$).

---

## 3. Current ML Results (Unseen-Well Evaluation)

| Model | R² (Unseen Wells) | MAE (ft) | RMSE (ft) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Baseline** | 0.7848 | 14.53 | 22.97 | Baseline |
| **XGBoost** | 0.8046 | 12.75 | 21.89 | Evaluated |
| **LightGBM (AMDFE)** | **0.8107** | **12.75** | **21.55** | **Preferred Candidate** |

---

## 4. Quickstart & Running the Platform

### A. Environment Setup
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### B. Launch FastAPI Backend
```bash
uvicorn backend.main:app --reload --port 8000
```
- Interactive OpenAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### C. Launch Frontend GIS Dashboard
- Directly open `frontend/index.html` in any modern web browser, or:
- Navigate to [http://localhost:8000](http://localhost:8000) when the backend is running.

### D. Run Automated Test Suites
```bash
# Test AMDFE Data Fusion & Causality
pytest tests/test_amdf.py -v

# Test ML, Unseen-Well GroupKFold, Recharge & XAI
pytest tests/test_ml.py -v

# Test FastAPI Endpoints & Section 13 Contract
pytest tests/test_backend.py -v
```

---

## 5. Integration Contract (Section 13)

The backend and frontend communicate via standardized JSON objects:

```json
{
  "well_id": "WELL-NE-018",
  "latitude": 40.4812,
  "longitude": -99.3789,
  "observed_level": 122.8,
  "predicted_level": 124.6,
  "risk_level": "Moderate",
  "recharge_potential": "High",
  "confidence": 0.89,
  "surface_elevation_m": 714.0,
  "year": 2024,
  "depletion_velocity_ft_yr": 0.62,
  "monitoring_priority_score": 38.5,
  "high_uncertainty_warning": false,
  "source_reliability_score": 0.89
}
```

---

## 6. Repository Structure

```
AquaSense-AI/
├── frontend/             # Member 1: GIS Map, Hydrographs, XAI Waterfall & Dashboard
│   ├── index.html
│   └── package.json
├── backend/              # Member 3: FastAPI REST service & Pydantic schemas
│   ├── routes/
│   │   ├── wells.py
│   │   ├── predictions.py
│   │   └── xai.py
│   ├── database.py
│   ├── schemas.py
│   └── main.py
├── ml/                   # Member 4: Models, Grouped CV, Recharge, Risk & SHAP XAI
│   ├── dataset.py
│   ├── models.py
│   ├── evaluate.py
│   ├── recharge.py
│   ├── risk.py
│   └── explainability.py
├── amdf/                 # Member 2: Quality, Reliability & Adaptive Fusion
│   ├── quality.py
│   ├── reliability.py
│   └── fusion.py
├── data/                 # Dataset documentation & processed CSVs
│   └── README.md
├── docs/                 # Research notes, data contract & paper references
│   ├── data_contract.md
│   └── research_direction.md
├── tests/                # Automated test suites for all modules
│   ├── test_amdf.py
│   ├── test_ml.py
│   └── test_backend.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 7. Disclaimer
AquaSense AI is a research-grade decision-support prototype. Predictions and risk assessments assist groundwater management districts and should not replace professional field hydrogeological investigations.
