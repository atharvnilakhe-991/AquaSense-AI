# Member 1 — Complete Frontend, GIS Visualization & Decision Support Architecture
## With Edolus 3D WebGL Background & Cognify Project Intelligence Dashboard

**Owner:** Member 1  
**Project:** AquaSense AI: Adaptive Multimodal AI for Hyperlocal Groundwater Prediction, Recharge Assessment & Decision Support  
**Study Area:** South-central Nebraska, USA (`40.3508° to 40.6841° N`, `−99.6432° to −99.1795° W`, 170 wells, 3,943 observations, 2000–2025).

---

## 1. Visual & Architectural Design Systems

### A. Homepage: Edolus-Inspired 3D WebGL Subsurface Topology
- **Inspiration Source:** [Edolus on Awwwards](https://www.awwwards.com/sites/edolus)
- **Technology:** Three.js GPU-accelerated WebGL (`PlaneGeometry` + additive blending particle system).
- **Aesthetic & Behavior:**
  - Undulating 3D fluid and topographic wave geometry representing deep underground aquifer strata.
  - Interactive mouse tracking: cursor coordinates smoothly ripple radial waves across the 3D particle surface and dynamically tilt the 3D camera.
  - Gradient colors: Luminous cyan (`#0EA5E9`), vibrant emerald (`#10B981`), and deep subterranean indigo.
  - Floating glassmorphic hero presentation with animated counters (`170 Wells`, `3,943 Observations`, `R²: 0.8107`).

### B. Decision Studio: Cognify Project Intelligence Dashboard
- **Inspiration Source:** [Cognify Project Intelligence Dashboard by Outcrowd on Dribbble](https://dribbble.com/shots/27061220-Cognify-Project-Intelligence-Dashboard)
- **Aesthetic & Layout:**
  - Modern **Bento-Grid architecture** with `rounded-2xl` and `rounded-3xl` translucent glassmorphic cards (`rgba(16, 22, 38, 0.75)` with `backdrop-blur-xl`).
  - **Micro-Sparklines:** SVG path sparklines integrated directly inside metric cards showing 5-year drawdown velocity and recharge infiltration surplus.
  - **Aquifer Stability Radial Meter:** Circular SVG progress ring tracking the 76% health status of South-central Nebraska wells.
  - **Two-Column Decision Workspace:**
    - Left column: Full interactive Leaflet GIS map with modern CartoDB Dark Matter tiles, well markers colored by scientifically defined risk tiers, and time slider (`2000–2025`).
    - Right column: Real-time well intelligence, **90% prediction intervals** (`36.6 – 39.4 m`), mini-trend hydrograph, and **SHAP Explainability decomposition bars**.

---

## 2. All 11 Functional Views & Workflows

1. **Home (3D WebGL Terrain):** Edolus-style real-time 3D particle landscape with hero CTA.
2. **Cognify Dashboard ⭐:** Bento-Grid executive intelligence studio with sparklines and radial gauge.
3. **Interactive GIS Map System:** Leaflet map with 7 multimodal layers (Wells, Depletion, Recharge, Risk, Satellite/DEM).
4. **Well Details & 90% Prediction Interval ⭐:** Inspection card with shaded uncertainty interval bands (`121.8 – 131.2 m`) and 25-year hydrographs.
5. **Well Comparison Tool ⭐:** Side-by-side comparative analysis of `Well A` vs `Well B`.
6. **Recharge Potential Analysis:** Infiltration capacity, precipitation surplus factor, and soil conditions.
7. **Decision-Support Risk Assessment:** Depletion velocity matrix and water district priority ranking.
8. **Explainable AI (SHAP Waterfall):** Global feature importance and per-well SHAP decomposition.
9. **Data Quality & AMDFE Fusion Dashboard:** Modality completeness indicators (Groundwater 94%, Weather 98%, Satellite 87%, Soil 91%, DEM 99%).
10. **Model Performance & Validation Comparison:** Model leaderboard (RF, XGBoost, LightGBM) and GroupKFold unseen-well cross-validation.
11. **Data Explorer:** Searchable, filterable table of all 3,943 observations with CSV export.

---

## 3. Geographic Study Area Citation Notice
In all documentation and publications:
> **Study Area:** South-central Nebraska, USA (spanning Phelps, Kearney, and Gosper county monitoring zones, 40.3508° to 40.6841° N, −99.6432° to −99.1795° W).
