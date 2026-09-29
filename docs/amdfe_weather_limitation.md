# Meteorological Temporal Aggregation: Limitations & Extensibility Interface
**AquaSense-AI Research Pipeline**
**Target**: Peer-Reviewed Hydroinformatics Literature
**Date**: September 2026

---

## 1. Context & Temporal Leakage Issue

In groundwater modeling, meteorological forcing (precipitation, temperature, evapotranspiration) drives dynamic recharge. However, when using annual climate summaries, standard practices frequently introduce **temporal lookahead leakage**:
- An in-situ groundwater measurement taken on **April 15, 2022** is matched with the calendar year 2022 total precipitation.
- This leaks precipitation from **May through December 2022** into a model predicting an April water level.

To strictly eliminate lookahead leakage without fabricating daily data, AMDFE enforces the **Completed Prior Period Rule**:
$$\text{Weather\_Year\_Used} = Y_{\text{observation}} - 1$$

---

## 2. Scientific Limitations of the $Y-1$ Annual Baseline

While the $Y-1$ annual weather policy provides a 100% leakage-safe guarantee, researchers must acknowledge its physical limitations:

1. **Hydrological Lag vs. Event Recharge**:
   - The $Y-1$ aggregate reflects the *macro-climatic antecedent background state* (e.g. multi-year drought vs wet cycle), but cannot resolve short-term event recharge from storms occurring 1–30 days prior to measurement.
2. **Seasonal Attenuation**:
   - Intra-annual seasonal variability (e.g. spring snowmelt vs summer convective rain) is integrated over the 365-day period, dampening high-frequency hydraulic head fluctuations.
3. **Boundary Truncation**:
   - Observations in year 2000 require 1999 annual weather, which is absent from the current localized dataset (2000–2024), resulting in missing weather values for year 2000 records.

---

## 3. Extensible Multi-Scale Weather Interface (Future Integration)

AMDFE is designed with a plug-in interface to support high-frequency reanalysis (e.g., ERA5-Land, Daymet, PRISM) when localized into the repository:

```python
class WeatherAdapterInterface:
    """
    Standardized abstract adapter for meteorological data streams.
    """
    def get_antecedent_weather(
        self,
        lat: float,
        lon: float,
        obs_date: str,
        lag_windows_days: list = [30, 90, 180, 365]
    ) -> dict:
        """
        Extracts antecedent precipitation, PET, and temperature over rolling windows
        strictly preceding obs_date.
        """
        raise NotImplementedError("Requires high-frequency ERA5-Land ingestion.")
```

Until daily/monthly gridded meteorological products are ingested, AMDFE v2 transparently reports annual weather as an **antecedent annual climatic background covariate**, rather than short-term hydrologic forcing.
