/**
 * AquaSense AI — Centralized Frontend Mock Data
 * Member 1: Frontend + GIS Dashboard ONLY
 * 
 * Study Area: South-Central Nebraska, USA
 * Counties: Phelps, Kearney, Gosper, Hallam, Dawson, Buffalo, Adams
 * Coordinates: Latitude 40.3508° to 40.6841° N, Longitude −99.6432° to −99.1795° W
 * Total Wells: 170 | Total Observations: 3,943 (2000–2025)
 */

export const MOCK_SYSTEM_METRICS = {
  telemetry: '100% Online',
  totalObservations: '3,943 Records',
  satelliteSync: 'Sentinel-2 & SMAP Synced',
  amdfeQuality: '92.1% SNR Mean',
  basinHealth: '76% Operational Stability',
  drawdownRate: '+0.38 m/yr',
  modelConfidence: '88.4%'
};

export const MOCK_KPI_CARDS = [
  {
    id: 'stations',
    title: 'Monitored Stations',
    value: '170',
    unit: '',
    status: 'Healthy',
    statusColor: 'emerald',
    badgeText: 'Healthy',
    trend: '+0 new this cycle',
    sparkline: [160, 162, 165, 168, 170, 170, 170]
  },
  {
    id: 'observations',
    title: 'Observations',
    value: '3,943',
    unit: '',
    status: 'Verified',
    statusColor: 'sky',
    badgeText: 'Verified',
    trend: '100% QA validated',
    sparkline: [3200, 3400, 3550, 3700, 3850, 3943]
  },
  {
    id: 'meanDepth',
    title: 'Mean Water Depth',
    value: '50.7',
    unit: 'm',
    status: 'Warning',
    statusColor: 'amber',
    badgeText: 'Warning',
    trend: '+0.38 m/yr drawdown',
    sparkline: [44.2, 45.8, 47.1, 48.9, 49.8, 50.7]
  },
  {
    id: 'forecast',
    title: 'ML 30d Forecast',
    value: '52.3',
    unit: 'm',
    status: 'Predictive',
    statusColor: 'purple',
    badgeText: 'Predictive',
    trend: '90% PI: 50.1–54.5m',
    sparkline: [49.8, 50.7, 51.2, 51.8, 52.3]
  },
  {
    id: 'highRisk',
    title: 'High-Risk Wells',
    value: '27',
    unit: '',
    status: 'Critical',
    statusColor: 'rose',
    badgeText: 'Critical',
    trend: '15.8% of basin network',
    sparkline: [14, 18, 20, 22, 25, 27]
  },
  {
    id: 'recharge',
    title: 'Recharge Potential',
    value: '76',
    unit: '/100',
    status: 'Optimal',
    statusColor: 'emerald',
    badgeText: 'Optimal',
    trend: 'Platte valley corridor',
    sparkline: [68, 70, 72, 75, 76]
  }
];

export const MOCK_HISTORICAL_TREND_DATA = {
  years: ['2000', '2004', '2008', '2012', '2016', '2020', '2024', '2025 (Now)', '2026 (Pred)', '2027 (Pred)'],
  basinAverageDepth: [43.2, 44.5, 45.9, 48.8, 49.3, 50.1, 50.5, 50.7, null, null],
  mlForecast: [null, null, null, null, null, null, 50.5, 50.7, 51.5, 52.3],
  criticalThreshold: [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
};

export const MOCK_PROJECT_RISK_ITEMS = [
  {
    id: 'risk-1',
    region: 'Gosper Uplands Over-Extraction',
    status: 'Critical Level',
    statusType: 'critical',
    description: 'Pumping stress ratio reached 1.68. Rapid drawdown rate detected in center-pivot agricultural wells.',
    actionLabel: 'Review Zone →'
  },
  {
    id: 'risk-2',
    region: 'Phelps Central Vadose Drying',
    status: 'Warning',
    statusType: 'warning',
    description: 'SMAP soil moisture decline observed. Seasonal recharge lag detected.',
    actionLabel: 'Check Recharge →'
  },
  {
    id: 'risk-3',
    region: 'Platte River Alluvial Infiltration',
    status: 'Healthy',
    statusType: 'healthy',
    description: 'High recharge conductivity maintaining groundwater stability.',
    actionLabel: null
  }
];

export const MOCK_AMDFE_QUALITY_BARS = [
  {
    id: 'piezo',
    label: 'In-Situ Piezometers (Continuous)',
    percentage: 94,
    color: 'emerald',
    detail: 'Continuous telemetry • SNR 24.2 dB'
  },
  {
    id: 'radar',
    label: 'Weather Radar (IMD/GPM Precipitation)',
    percentage: 98,
    color: 'sky',
    detail: 'Gridded 0.1° GPM IMERG Daily'
  },
  {
    id: 'sentinel',
    label: 'Sentinel-2 Optical NDVI (Cloud-Filtered)',
    percentage: 87,
    color: 'purple',
    detail: '10m Surface Reflectance QA masked'
  },
  {
    id: 'smap',
    label: 'SMAP Soil Moisture Radiometer (1km Grid)',
    percentage: 91,
    color: 'amber',
    detail: 'L4 Surface & Root Zone 9km/1km'
  },
  {
    id: 'srtm',
    label: 'SRTM Digital Elevation Model (Topography)',
    percentage: 99,
    color: 'cyan',
    detail: 'USGS 30m Hydro-enforced DEM'
  }
];

export const MOCK_DECISION_DIRECTIVES = [
  {
    id: 'dir-1',
    title: 'Gosper Uplands Pumping Curtailment',
    tag: 'Priority 1',
    tagColor: 'rose',
    description: 'Drawdown velocity exceeding regional replenishment. Review pumping reduction for summer irrigation cycles.'
  },
  {
    id: 'dir-2',
    title: 'Managed Aquifer Recharge (MAR) Diversion',
    tag: 'Intervention',
    tagColor: 'cyan',
    description: 'Potential recharge intervention identified based on recharge suitability in Platte canal corridor.'
  },
  {
    id: 'dir-3',
    title: 'Telemetry Piezometer Calibration',
    tag: 'Maintenance',
    tagColor: 'amber',
    description: 'Sensor drift identified in 3 monitoring nodes. Physical calibration inspection recommended.'
  }
];

export const MOCK_RISK_CATEGORIES = [
  {
    label: 'Stable / Low Risk',
    threshold: '<18m',
    count: 82,
    percent: '48.2%',
    color: 'emerald',
    dotColor: '#10B981'
  },
  {
    label: 'Moderate Risk',
    threshold: '18–38m',
    count: 41,
    percent: '24.1%',
    color: 'amber',
    dotColor: '#F59E0B'
  },
  {
    label: 'High Risk',
    threshold: '38–65m',
    count: 20,
    percent: '11.8%',
    color: 'orange',
    dotColor: '#F97316'
  },
  {
    label: 'Critical Over-exploited',
    threshold: '>65m',
    count: 27,
    percent: '15.9%',
    color: 'rose',
    dotColor: '#EF4444'
  }
];

// Generator for the 170 Mock Geolocated Monitoring Stations
export function generateMockWells() {
  const wells = [];
  const counties = ['Phelps', 'Kearney', 'Gosper', 'Dawson', 'Buffalo', 'Adams', 'Hallam'];
  const minLat = 40.3508, maxLat = 40.6841;
  const minLon = -99.6432, maxLon = -99.1795;

  for (let i = 1; i <= 170; i++) {
    const id = `WELL-NE-${String(i).padStart(3, '0')}`;
    const lat = minLat + (Math.sin(i * 997) * 0.5 + 0.5) * (maxLat - minLat);
    const lon = minLon + (Math.cos(i * 773) * 0.5 + 0.5) * (maxLon - minLon);
    const county = counties[i % counties.length];

    // Distribute risk categories strictly: 27 Critical, 20 High, 41 Moderate, 82 Low
    let riskLevel = 'Stable / Low Risk';
    let depth = 14.2 + (i % 15) * 0.25;
    let color = '#10B981';

    if (i <= 27) {
      riskLevel = 'Critical Over-exploited';
      depth = 66.5 + (i % 12) * 1.2;
      color = '#EF4444';
    } else if (i <= 47) {
      riskLevel = 'High Risk';
      depth = 42.0 + (i % 18) * 1.1;
      color = '#F97316';
    } else if (i <= 88) {
      riskLevel = 'Moderate Risk';
      depth = 22.0 + (i % 14) * 0.9;
      color = '#F59E0B';
    }

    wells.push({
      well_id: id,
      county: `${county} County`,
      latitude: parseFloat(lat.toFixed(4)),
      longitude: parseFloat(lon.toFixed(4)),
      elevation_m: 685 + (i % 55) * 1.5,
      water_depth_m: parseFloat(depth.toFixed(1)),
      risk_level: riskLevel,
      color: color,
      recharge_potential: (i % 3 === 0) ? 'High' : ((i % 3 === 1) ? 'Moderate' : 'Low'),
      last_observation: '2025-03-15',
      confidence: `${(82 + (i % 16)).toFixed(0)}%`
    });
  }
  return wells;
}
