/**
 * AquaSense AI — Frontend API Service Layer
 * Member 1: Frontend + GIS Dashboard ONLY
 * 
 * Provides clean frontend service placeholders ready for future backend integration:
 * - getWells()
 * - getGroundwaterData()
 * - getPredictions()
 * - getRechargeData()
 * - getRiskData()
 * - getAMDFEData()
 * - getXAIData()
 * 
 * Centralized fallback to local South-Central Nebraska benchmark dataset:
 * Phelps County, NE (170 Wells, 3,844 Observations, 2000–2024).
 */

import { MOCK_WELLS, STUDY_AREA_INFO } from '../data/wellData';
import { GROUNDWATER_FACTS, KPI_CARDS_DATA, GROUNDWATER_TREND_SERIES } from '../data/groundwaterData';
import { RISK_DISTRIBUTION_DATA, GROUNDWATER_ALERTS } from '../data/riskData';
import { RECHARGE_CATEGORIES, RECHARGE_FACTORS } from '../data/rechargeData';
import { AMDFE_MODALITIES, AMDFE_SYSTEM_STATS } from '../data/amdfData';
import { ML_MODELS_DATA, AI_DECISION_SUPPORT_ITEMS } from '../data/modelData';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const apiService = {
  /**
   * GET /wells — Fetches all 170 monitored wells with risk and recharge attributes
   */
  async getWells(filters = {}) {
    try {
      const queryParams = new URLSearchParams(filters).toString();
      const res = await fetch(`${API_BASE_URL}/wells?${queryParams}`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Expected: Backend will be connected by Member 3/4 later
    }
    return MOCK_WELLS;
  },

  /**
   * GET /wells/{well_id} — Fetches detail for a single well
   */
  async getWellDetail(wellId) {
    try {
      const res = await fetch(`${API_BASE_URL}/wells/${wellId}`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback to local well data
    }
    return MOCK_WELLS.find(w => w.id === wellId) || MOCK_WELLS[0];
  },

  /**
   * GET /groundwater — Fetches core groundwater statistics & KPI cards
   */
  async getGroundwaterData() {
    try {
      const res = await fetch(`${API_BASE_URL}/groundwater`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      facts: GROUNDWATER_FACTS,
      kpis: KPI_CARDS_DATA,
      studyArea: STUDY_AREA_INFO
    };
  },

  /**
   * GET /predictions — Fetches ML prediction horizons and historical trend series
   */
  async getPredictions(timeframe = 'all') {
    try {
      const res = await fetch(`${API_BASE_URL}/predictions?timeframe=${timeframe}`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      trend: GROUNDWATER_TREND_SERIES[timeframe] || GROUNDWATER_TREND_SERIES.all,
      models: ML_MODELS_DATA
    };
  },

  /**
   * GET /recharge — Fetches recharge categories and environmental factor scores
   */
  async getRechargeData() {
    try {
      const res = await fetch(`${API_BASE_URL}/recharge`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      categories: RECHARGE_CATEGORIES,
      factors: RECHARGE_FACTORS
    };
  },

  /**
   * GET /risk — Fetches risk distribution and real-time groundwater alerts
   */
  async getRiskData() {
    try {
      const res = await fetch(`${API_BASE_URL}/risk`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      distribution: RISK_DISTRIBUTION_DATA,
      alerts: GROUNDWATER_ALERTS
    };
  },

  /**
   * GET /amdfe — Fetches multimodal data quality and sensor reliability metrics
   */
  async getAMDFEData() {
    try {
      const res = await fetch(`${API_BASE_URL}/amdfe`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      modalities: AMDFE_MODALITIES,
      stats: AMDFE_SYSTEM_STATS
    };
  },

  /**
   * GET /xai — Fetches explainable AI directives and decision support recommendations
   */
  async getXAIData() {
    try {
      const res = await fetch(`${API_BASE_URL}/xai`, {
        signal: AbortSignal.timeout(2000)
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      directives: AI_DECISION_SUPPORT_ITEMS
    };
  }
};
