/**
 * AquaSense AI — AMDFE Data Quality & Reliability Module
 * Adaptive Multimodal Data Fusion Engine (Frontend Visualization Prototype)
 */

export const AMDFE_MODALITIES = [
  {
    name: 'Groundwater (In-Situ)',
    type: 'Piezometers',
    quality: 94,
    reliability: 'Good',
    status: 'Good',
    color: '#0D9488', // teal
    badgeClass: 'bg-teal-50 text-teal-700 border-teal-200',
    detail: 'Continuous telemetry • SNR 24.2 dB • QA filtered'
  },
  {
    name: 'Weather (Precipitation)',
    type: 'Radar / IMD / GPM',
    quality: 98,
    reliability: 'Excellent',
    status: 'Excellent',
    color: '#0284C7', // blue
    badgeClass: 'bg-sky-50 text-sky-700 border-sky-200',
    detail: 'Gridded 0.1° GPM IMERG daily precipitation'
  },
  {
    name: 'Satellite (Optical NDVI)',
    type: 'Sentinel-2 MSI',
    quality: 87,
    reliability: 'Good',
    status: 'Good',
    color: '#10B981', // green
    badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    detail: '10m surface reflectance • Cloud-screened QA'
  },
  {
    name: 'Soil (Moisture Radiometer)',
    type: 'SMAP L4 Grid',
    quality: 91,
    reliability: 'Good',
    status: 'Good',
    color: '#F59E0B', // amber
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    detail: 'L4 surface & root-zone moisture (1km downscaled)'
  },
  {
    name: 'DEM / Elevation (Topography)',
    type: 'SRTM 30m',
    quality: 99,
    reliability: 'Excellent',
    status: 'Excellent',
    color: '#6366F1', // indigo
    badgeClass: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    detail: 'Hydro-enforced USGS 30m digital elevation model'
  }
];

export const AMDFE_SYSTEM_STATS = {
  meanQuality: '93.8%',
  errorReduction: '-38.2% RMSE vs single source',
  fusionEngineVersion: 'AMDFE v2.4 (Frontend Prototype)',
  statusText: 'AMDFE dynamically evaluates data quality and reliability before multimodal fusion.'
};
