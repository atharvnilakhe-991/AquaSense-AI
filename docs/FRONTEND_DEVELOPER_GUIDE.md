# Member 1 — Complete Frontend Developer Guide & Architecture Manual
## AquaSense AI: Hyperlocal Groundwater Intelligence & Decision-Support System

**Role:** Member 1 — Frontend, GIS Visualization, Dashboard & Decision Support Interface  
**Study Area:** South-central Nebraska, USA  
**Geographic Bounding Box:**
- **Latitude:** 40.3508° to 40.6841° N
- **Longitude:** −99.6432° to −99.1795° W
- **Wells Monitored:** 170
- **Total Groundwater Observations:** 3,943
- **Monitoring Period:** 2000–2025

---

## 1. Technical Stack & Component Architecture

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Framework** | React.js / HTML5 / ES2023 | Core application structure and component lifecycle |
| **Styling** | Tailwind CSS (Cognify Theme) | Bento-Grid layout, rounded-2xl glassmorphism, responsive utilities |
| **3D Background** | Three.js (WebGL) | Edolus-inspired interactive subsurface fluid & particle terrain |
| **GIS Mapping** | Leaflet / React-Leaflet | 170-well interactive spatial visualization, multi-layer toggles |
| **Analytics Charts**| Chart.js | Hydrograph time-series, 90% prediction interval bands, scatter plots |
| **API Client** | `apiService.js` (Fetch API) | REST client with automatic fallback to South-central NE benchmark |

---

## 2. All 11 Pages & Routes (Member 1 Workload)

```
AquaSense AI
│
├── 1. Home / Landing Page
│   └── Edolus 3D WebGL background, animated stats counter, hero CTA
├── 2. Dashboard ⭐
│   └── 6 Core KPI cards (170 Wells, 3,943 Obs, Avg Level, Predicted Level, High-Risk Wells, Recharge),
│       micro-sparklines, Aquifer Stability radial gauge, and quick-inspect panel
├── 3. Groundwater Map ⭐
│   └── Leaflet GIS map with 7 multimodal layer toggles:
│       ☑ Groundwater Wells (🟢 Stable, 🟡 Moderate, 🟠 High Risk, 🔴 Critical)
│       ☑ Groundwater Level (depth contours)
│       ☑ Depletion Trend Map (Increasing, Stable, Declining, Rapid Drawdown)
│       ☐ Recharge Potential Overlay (Very High, Moderate, Low, Very Low)
│       ☐ Risk Layer
│       ☐ Satellite (Sentinel-1/2)
│       ☐ Elevation (DEM)
│       and the interactive Time Slider (2000–2025)
├── 4. Well Details & Uncertainty ⭐
│   └── Complete metadata panel, predicted level, 90% Prediction Interval (121.8 – 131.2 m),
│       confidence score (84%), and 25-year hydrograph with shaded error bands
├── 5. Prediction Analysis
│   └── Predicted vs Actual scatter/line chart (● Actual points, ─ Predicted line)
├── 6. Recharge Analysis
│   └── Infiltration capacity, rainfall surplus, soil condition, and elevation response
├── 7. Risk Assessment ⭐
│   └── Depletion velocity matrix, priority ranking table, and action recommendations
├── 8. Well Comparison Tool ⭐
│   └── Side-by-side comparative table analyzing WELL A vs WELL B
├── 9. Data Explorer
│   └── Filterable, searchable table of all 3,943 observations with CSV export
├── 10. Model Performance
│   └── Candidate leaderboard (RF, XGBoost, LightGBM, AMDFE) and Validation Comparison (Temporal, Unseen Wells, Spatial holdout)
├── 11. About / Methodology
│   └── Formal study area definition (South-central Nebraska) and research formulation
└── 12. Help / Documentation
    └── User instructions, map layer guide, and legend explanation
```

---

## 3. Team Handoffs & How Member 1 Interfaces

### A. Interface with Member 3 (Backend API)
Member 1 consumes REST endpoints via `frontend/src/services/apiService.js`:
- `GET /api/v1/wells` → Populates GIS markers and well tables.
- `GET /api/v1/wells/{id}` → Populates well details and historical hydrographs.
- `GET /api/v1/predictions/summary` → Populates executive dashboard cards.
- `GET /api/v1/xai/local/{id}` → Populates SHAP waterfall attribution bars.

*Note for Frontend Developer:* If Member 3's backend is not running, `apiService.js` automatically uses the built-in 170-well South-central Nebraska benchmark so you can develop and test the entire UI offline.

### B. Interface with Member 4 (AI/ML & Research Team)
- **Risk Thresholds:** Scientifically defined by Member 4 (Critical: drawdown $> 0.35$ m/yr or depth $> 46$m).
- **Prediction Uncertainty:** Member 1 visualizes Member 4's 90% prediction intervals (`interval_lower_m` and `interval_upper_m`) as shaded confidence bands on hydrograph charts.
- **XAI:** Member 1 receives SHAP values from Member 4 and renders them as horizontal contribution bars.

### C. Interface with Member 2 (AMDFE)
- **Reliability:** Member 1 renders data quality meters (Groundwater 94%, Weather 98%, Satellite 87%, Soil 91%, DEM 99%) and observation reliability tags (High, Medium, Low).

---

## 4. How to Run and Test as Frontend Developer

### Instant Zero-Install Browser View:
Simply double-click or open [frontend/index.html](file:///c:/Users/Lenovo/OneDrive/Desktop/AQUA/frontend/index.html) in Chrome, Edge, or Firefox.
- **Edolus 3D Animation:** Move your cursor on the Home tab to verify the 3D WebGL wave physics.
- **Cognify Dashboard:** Click into the Dashboard tab to inspect the sparklines, radial gauge, and GIS map.
- **Time Slider:** Move the slider between 2000 and 2025 to observe dynamic marker color updates.
- **Well Comparison:** Switch to the Well Comparison tab and compare two wells side-by-side.

### React Modular Workflow:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to develop and preview the React components.
