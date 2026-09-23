/**
 * AquaSense AI — Groundwater Data Module
 * Dataset facts: 170 Monitoring Wells, 3,844 Observations, 2000–2024 Dataset
 * Study Area: Phelps County, South-Central Nebraska
 */

export const GROUNDWATER_FACTS = {
  totalWells: 170,
  totalObservations: 3844,
  period: '2000–2024',
  studyArea: 'Phelps County, Nebraska, USA',
  region: 'South-Central Nebraska',
  meanWaterDepthM: 50.7,
  predictedWaterDepthM: 52.3,
  rechargeStatus: 'Optimal / Moderate',
  rechargeScore: 76,
  criticalWellsCount: 24,
  drawdownRate: '+0.38 m/yr'
};

export const KPI_CARDS_DATA = [
  {
    id: 'kpi-wells',
    title: 'Groundwater Monitoring',
    value: '170',
    unit: '',
    description: 'Monitored Wells',
    status: '100% Active',
    statusType: 'success',
    trend: 'Phelps County Network',
    badgeText: 'Active Network',
    isDemo: false
  },
  {
    id: 'kpi-obs',
    title: 'Groundwater Observations',
    value: '3,844',
    unit: '',
    description: 'Observations',
    status: 'Verified',
    statusType: 'success',
    trend: '2000–2024 Dataset',
    badgeText: 'Curated Dataset',
    isDemo: false
  },
  {
    id: 'kpi-mean-depth',
    title: 'Average Water Depth',
    value: '50.7',
    unit: 'm',
    description: 'Mean Piezometric Depth',
    status: 'Notice',
    statusType: 'warning',
    trend: '+0.38 m/yr Drawdown',
    badgeText: 'Demo Value',
    isDemo: true
  },
  {
    id: 'kpi-predicted-depth',
    title: '30-Day Prediction',
    value: '52.3',
    unit: 'm',
    description: 'ML Forecast Horizon',
    status: 'Predictive',
    statusType: 'info',
    trend: '90% PI: 50.1–54.5m',
    badgeText: 'ML Prediction — Demo',
    isDemo: true
  },
  {
    id: 'kpi-recharge',
    title: 'Recharge Potential',
    value: '76',
    unit: '/100',
    description: 'Platte Corridor Capacity',
    status: 'Favorable',
    statusType: 'success',
    trend: 'Optimal Conductive Zone',
    badgeText: 'Demo Score',
    isDemo: true
  },
  {
    id: 'kpi-risk',
    title: 'High-Risk Wells',
    value: '24',
    unit: 'Critical',
    description: 'Over-Drafted Stations',
    status: 'Attention',
    statusType: 'critical',
    trend: '14.1% Critical Ratio',
    badgeText: 'Demo Risk',
    isDemo: true
  }
];

export const GROUNDWATER_TREND_SERIES = {
  all: {
    labels: ['2000', '2003', '2006', '2009', '2012', '2015', '2018', '2021', '2024 (Now)', '2025 (Pred)', '2026 (Pred)'],
    observed: [43.2, 44.5, 45.9, 47.4, 49.3, 49.8, 50.1, 50.5, 50.7, null, null],
    predicted: [null, null, null, null, null, null, null, null, 50.7, 51.5, 52.3],
    criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
  },
  fiveYears: {
    labels: ['2020', '2021', '2022', '2023', '2024 (Now)', '2025 (Pred)'],
    observed: [49.8, 50.1, 50.4, 50.6, 50.7, null],
    predicted: [null, null, null, null, 50.7, 51.5],
    criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
  },
  oneYear: {
    labels: ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024 (Now)', 'Q1 2025 (Pred)'],
    observed: [50.4, 50.5, 50.6, 50.7, null],
    predicted: [null, null, null, 50.7, 51.1],
    criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0]
  },
  thirtyDays: {
    labels: ['Day 1', 'Day 7', 'Day 14', 'Day 21', 'Day 30 (Now)', 'Day +15 (Pred)'],
    observed: [50.62, 50.64, 50.66, 50.68, 50.70, null],
    predicted: [null, null, null, null, 50.70, 50.78],
    criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
  },
  sevenDays: {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun (Now)', 'Next (Pred)'],
    observed: [50.67, 50.68, 50.68, 50.69, 50.69, 50.70, 50.70, null],
    predicted: [null, null, null, null, null, null, 50.70, 50.74],
    criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
  }
};

// 20 Sample Observations for Data Explorer
export const MOCK_EXPLORER_OBSERVATIONS = [
  { id: 'OBS-001', wellId: 'NE-PH-001', date: '2024-11-28', lat: 40.4124, lon: -99.4521, level: 66.8, temp: '11.2 °C', rain: '520 mm', soil: '0.27 m³/m³', quality: '94% Continuous Telemetry' },
  { id: 'OBS-002', wellId: 'NE-PH-002', date: '2024-11-28', lat: 40.4518, lon: -99.3812, level: 67.9, temp: '11.0 °C', rain: '515 mm', soil: '0.26 m³/m³', quality: '96% Continuous Telemetry' },
  { id: 'OBS-003', wellId: 'NE-PH-003', date: '2024-11-27', lat: 40.3892, lon: -99.4930, level: 68.2, temp: '10.8 °C', rain: '530 mm', soil: '0.25 m³/m³', quality: '92% Manual Sounding' },
  { id: 'OBS-004', wellId: 'NE-PH-004', date: '2024-11-27', lat: 40.4412, lon: -99.3721, level: 69.1, temp: '11.4 °C', rain: '510 mm', soil: '0.24 m³/m³', quality: '95% Continuous Telemetry' },
  { id: 'OBS-005', wellId: 'NE-PH-018', date: '2024-11-26', lat: 40.4631, lon: -99.3418, level: 43.8, temp: '11.1 °C', rain: '542 mm', soil: '0.29 m³/m³', quality: '93% Piezometer QA' },
  { id: 'OBS-006', wellId: 'NE-PH-025', date: '2024-11-26', lat: 40.5120, lon: -99.4105, level: 25.1, temp: '10.9 °C', rain: '550 mm', soil: '0.31 m³/m³', quality: '98% Continuous Telemetry' },
  { id: 'OBS-007', wellId: 'NE-PH-042', date: '2024-11-25', lat: 40.5524, lon: -99.4891, level: 26.4, temp: '10.7 °C', rain: '545 mm', soil: '0.30 m³/m³', quality: '91% Piezometer QA' },
  { id: 'OBS-008', wellId: 'NE-PH-062', date: '2024-11-25', lat: 40.6128, lon: -99.2314, level: 16.2, temp: '10.5 °C', rain: '560 mm', soil: '0.34 m³/m³', quality: '99% Alluvial Gauge' },
  { id: 'OBS-009', wellId: 'NE-PH-078', date: '2024-11-24', lat: 40.5891, lon: -99.2789, level: 17.5, temp: '10.6 °C', rain: '555 mm', soil: '0.32 m³/m³', quality: '97% Alluvial Gauge' },
  { id: 'OBS-010', wellId: 'NE-PH-089', date: '2024-11-24', lat: 40.5284, lon: -99.5512, level: 41.2, temp: '11.0 °C', rain: '535 mm', soil: '0.28 m³/m³', quality: '90% Continuous Telemetry' },
  { id: 'OBS-011', wellId: 'NE-PH-102', date: '2024-11-23', lat: 40.4812, lon: -99.3905, level: 23.8, temp: '11.3 °C', rain: '528 mm', soil: '0.29 m³/m³', quality: '95% Continuous Telemetry' },
  { id: 'OBS-012', wellId: 'NE-PH-115', date: '2024-11-23', lat: 40.4190, lon: -99.4682, level: 65.4, temp: '11.1 °C', rain: '518 mm', soil: '0.25 m³/m³', quality: '93% Piezometer QA' },
  { id: 'OBS-013', wellId: 'NE-PH-128', date: '2024-11-22', lat: 40.6421, lon: -99.2104, level: 15.9, temp: '10.4 °C', rain: '565 mm', soil: '0.35 m³/m³', quality: '98% Alluvial Gauge' },
  { id: 'OBS-014', wellId: 'NE-PH-142', date: '2024-11-22', lat: 40.3685, lon: -99.5109, level: 67.1, temp: '11.5 °C', rain: '512 mm', soil: '0.24 m³/m³', quality: '94% Continuous Telemetry' },
  { id: 'OBS-015', wellId: 'NE-PH-156', date: '2024-11-21', lat: 40.5401, lon: -99.3120, level: 24.6, temp: '10.8 °C', rain: '540 mm', soil: '0.30 m³/m³', quality: '96% Continuous Telemetry' }
];
