/**
 * AquaSense AI — Recharge Assessment Data Module
 * Frontend visualization prototype for groundwater recharge potential
 */

export const RECHARGE_CATEGORIES = [
  {
    category: 'High Recharge Potential',
    percentage: 34,
    description: 'Platte alluvial corridor & sandy loam depressions',
    color: '#0D9488', // teal
    badgeClass: 'bg-teal-50 text-teal-700 border-teal-200',
    barClass: 'bg-teal-600'
  },
  {
    category: 'Moderate Recharge Potential',
    percentage: 46,
    description: 'Central agricultural loess plain with seasonal infiltration',
    color: '#0284C7', // blue
    badgeClass: 'bg-sky-50 text-sky-700 border-sky-200',
    barClass: 'bg-sky-500'
  },
  {
    category: 'Low Recharge Potential',
    percentage: 20,
    description: 'Dense clay subsoils with low vadose conductivity',
    color: '#F59E0B', // amber
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    barClass: 'bg-amber-500'
  }
];

export const RECHARGE_FACTORS = [
  {
    name: 'Rainfall & Precipitation',
    value: '540 mm/yr',
    score: 78,
    status: 'Favorable',
    description: 'GPM IMERG gridded radar precipitation input'
  },
  {
    name: 'Soil Permeability & Moisture',
    value: '0.28 m³/m³',
    score: 72,
    status: 'Moderate',
    description: 'SMAP L4 surface and root-zone soil moisture'
  },
  {
    name: 'Topographic Elevation (DEM)',
    value: '685–740 m',
    score: 84,
    status: 'Optimal',
    description: 'SRTM 30m digital elevation model drainage slope'
  },
  {
    name: 'Groundwater Hydraulic Gradient',
    value: '-0.38 m/yr',
    score: 61,
    status: 'Deficit Warning',
    description: 'Historical in-situ piezometric drawdown trajectory'
  },
  {
    name: 'Surface Land Cover Conditions',
    value: 'Cropland & Canals',
    score: 80,
    status: 'Conductive',
    description: 'Sentinel-2 optical vegetation index (NDVI)'
  }
];
