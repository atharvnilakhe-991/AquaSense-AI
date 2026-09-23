/**
 * AquaSense AI — ML Models & Decision Support Module
 * Models comparison & Prescriptive Decision Directives (Frontend Prototype)
 */

export const ML_MODELS_DATA = [
  {
    id: 'rf',
    name: 'Random Forest Regressor',
    architecture: 'Baseline Ensemble Trees',
    status: 'Available Model',
    readiness: 'Prediction Ready',
    relativePerformance: 88,
    r2: 0.784,
    rmse: '1.42 m',
    mae: '1.08 m',
    badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    description: 'Lagged water level + meteorological feature baselines with 200 estimators.'
  },
  {
    id: 'xgb',
    name: 'XGBoost Gradient Boosting',
    architecture: 'Extreme Gradient Boosting',
    status: 'Available Model',
    readiness: 'Prediction Ready',
    relativePerformance: 91,
    r2: 0.812,
    rmse: '1.28 m',
    mae: '0.95 m',
    badgeClass: 'bg-sky-50 text-sky-700 border-sky-200',
    description: 'Fast non-linear feature split with L1/L2 regularization and early stopping.'
  },
  {
    id: 'lgb',
    name: 'LightGBM Regressor',
    architecture: 'Light Gradient Boost Machine',
    status: 'Available Model',
    readiness: 'Prediction Ready',
    relativePerformance: 90,
    r2: 0.805,
    rmse: '1.31 m',
    mae: '0.98 m',
    badgeClass: 'bg-teal-50 text-teal-700 border-teal-200',
    description: 'Leaf-wise tree growth with histogram-based binning optimized for tabular features.'
  },
  {
    id: 'amdfe-ensemble',
    name: 'AMDFE Multimodal Ensemble',
    architecture: 'Reliability-Weighted Fusion',
    status: 'Available Model',
    readiness: 'Backend Integration Pending',
    relativePerformance: 95,
    r2: 0.862,
    rmse: '1.12 m',
    mae: '0.82 m',
    badgeClass: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    description: 'Dynamic SNR weighted multimodal predictor (90% PI) combining in-situ, radar, and satellite.'
  }
];

export const AI_DECISION_SUPPORT_ITEMS = [
  {
    id: 'dec-1',
    directive: 'High-risk well requires monitoring',
    priority: 'Priority 1 — Attention',
    priorityColor: 'rose',
    description: 'Drawdown velocity exceeding regional replenishment in the Holdrege sector. Prioritize continuous piezometric observation and extraction review.',
    category: 'Extraction Management',
    tag: 'Demo Decision Support'
  },
  {
    id: 'dec-2',
    directive: 'Recharge potential detected',
    priority: 'Intervention Opportunity',
    priorityColor: 'teal',
    description: 'High infiltration conductivity identified along the Platte canal corridor. Suitable for Managed Aquifer Recharge (MAR) surface diversion.',
    category: 'Recharge Enhancement',
    tag: 'Demo Decision Support'
  },
  {
    id: 'dec-3',
    directive: 'Groundwater trend requires observation',
    priority: 'Trend Advisory',
    priorityColor: 'amber',
    description: 'Seasonal vadose drying observed in center-pivot agricultural wells. Recommend ongoing tracking of water table stabilization during winter cycles.',
    category: 'Vadose Monitoring',
    tag: 'Demo Decision Support'
  }
];

export const QUICK_ACCESS_MODULES = [
  {
    id: 'mod-gis',
    title: 'GIS Monitoring',
    description: 'Explore groundwater wells across Nebraska with interactive spatial layers.',
    icon: 'MapPin',
    actionText: 'Open GIS Map',
    path: '/groundwater-map',
    badge: '170 Wells'
  },
  {
    id: 'mod-predictions',
    title: 'Prediction Analysis',
    description: 'View groundwater prediction outputs, 90% uncertainty intervals, and horizon forecasts.',
    icon: 'TrendingUp',
    actionText: 'View Predictions',
    path: '/prediction-analysis',
    badge: 'ML Prototype'
  },
  {
    id: 'mod-recharge',
    title: 'Recharge Assessment',
    description: 'Explore recharge potential, soil vadose conductivity, and canal infiltration zones.',
    icon: 'Droplets',
    actionText: 'Assess Recharge',
    path: '/recharge-analysis',
    badge: 'Multi-Modal'
  },
  {
    id: 'mod-risk',
    title: 'Risk Assessment',
    description: 'Identify groundwater risk zones, over-drafted sectors, and depletion hotspots.',
    icon: 'AlertTriangle',
    actionText: 'Review Risk Zones',
    path: '/risk-assessment',
    badge: '24 Critical'
  }
];

export const FAQ_DOCUMENTATION_ITEMS = [
  {
    id: 'faq-1',
    question: 'How do I interpret the Groundwater Map and risk classifications?',
    answer: 'The map classifies 170 monitored wells in Phelps County into 4 tiers: Green (<20m) indicates stable water levels, Yellow (20-38m) indicates moderate depth, Orange (38-55m) denotes high drawdown stress, and Red (>55m) represents critical over-extraction requiring immediate priority management.'
  },
  {
    id: 'faq-2',
    question: 'What are the 90% Prediction Intervals (PI) in Prediction Analysis?',
    answer: 'AquaSense AI ML models generate an uncertainty envelope representing the 90% confidence range (e.g. ±0.9m). This accounts for missing telemetry soundings, sensor drift, and seasonal rainfall volatility.'
  },
  {
    id: 'faq-3',
    question: 'How does the Recharge Potential assessment work?',
    answer: 'Recharge potential evaluates multi-factor vadose conductivity combining GPM radar rainfall (540 mm/yr), SMAP soil moisture (0.28 m³/m³), SRTM topographic slope, and Sentinel-2 NDVI surface cover across alluvial canal corridors.'
  },
  {
    id: 'faq-4',
    question: 'What is the role of the AMDFE engine in data fusion?',
    answer: 'The Adaptive Multimodal Data Fusion Engine dynamically estimates the Signal-to-Noise Ratio (SNR) for each of the 5 sensor modalities (piezometers, radar, NDVI, SMAP, DEM) to prevent low-quality or cloud-occluded measurements from biasing predictions.'
  },
  {
    id: 'faq-5',
    question: 'Can I export observational records from the Data Explorer?',
    answer: 'Yes. Use the Data Explorer page to filter by well ID, township, risk tier, or date range, and click "Export CSV" to download the records for offline research analysis.'
  }
];
