/**
 * AquaSense AI — Risk Distribution & Alerts Module
 * Centralized Demo/Prototype data for Risk Classification & Alerts
 */

export const RISK_DISTRIBUTION_DATA = [
  {
    category: 'Stable / Low Risk',
    threshold: '< 20m Depth',
    count: 85,
    percentage: 50.0,
    color: '#10B981', // green
    badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    barClass: 'bg-emerald-500'
  },
  {
    category: 'Moderate Risk',
    threshold: '20m – 38m Depth',
    count: 42,
    percentage: 24.7,
    color: '#EAB308', // yellow
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    barClass: 'bg-amber-500'
  },
  {
    category: 'High Risk',
    threshold: '38m – 55m Depth',
    count: 19,
    percentage: 11.2,
    color: '#F97316', // orange
    badgeClass: 'bg-orange-50 text-orange-700 border-orange-200',
    barClass: 'bg-orange-500'
  },
  {
    category: 'Critical',
    threshold: '> 55m Over-drawn',
    count: 24,
    percentage: 14.1,
    color: '#EF4444', // red
    badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
    barClass: 'bg-rose-500'
  }
];

export const GROUNDWATER_ALERTS = [
  {
    id: 'alt-01',
    wellId: 'NE-PH-004',
    location: 'Holdrege Township, Phelps Co.',
    alertType: 'Critical Depletion',
    severity: 'Critical',
    severityColor: 'rose',
    time: '12m ago',
    message: 'Drawdown velocity exceeded +0.48 m/yr. Water table reached 68.2m depth.',
    action: 'Review Well'
  },
  {
    id: 'alt-02',
    wellId: 'NE-PH-018',
    location: 'Funk Agricultural Area, Phelps Co.',
    alertType: 'High Drawdown Stress',
    severity: 'High Risk',
    severityColor: 'orange',
    time: '45m ago',
    message: 'Continuous seasonal extraction rate detected near center-pivot wells.',
    action: 'Inspect Zone'
  },
  {
    id: 'alt-03',
    wellId: 'NE-PH-062',
    location: 'Platte Alluvial Margin, Phelps Co.',
    alertType: 'Recharge Infiltration Window',
    severity: 'Recharge Alert',
    severityColor: 'emerald',
    time: '2h ago',
    message: 'Positive hydraulic gradient detected following canal surface flow.',
    action: 'Assess Infiltration'
  },
  {
    id: 'alt-04',
    wellId: 'NE-PH-089',
    location: 'Bertrand District, Phelps Co.',
    alertType: 'Telemetry Uncertainty Drift',
    severity: 'Data Quality Alert',
    severityColor: 'purple',
    time: '5h ago',
    message: 'AMDFE detected SNR variance > 15% across piezometer telemetry node.',
    action: 'Calibrate Sensor'
  },
  {
    id: 'alt-05',
    wellId: 'NE-PH-112',
    location: 'Westmark Sector, Phelps Co.',
    alertType: 'Observation Latency Notice',
    severity: 'Sensor/Observation Alert',
    severityColor: 'sky',
    time: '8h ago',
    message: 'Station awaiting scheduled bi-weekly manual piezometric sounding.',
    action: 'Check Schedule'
  }
];
