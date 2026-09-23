/**
 * AquaSense AI — Well Data Module
 * Centralized Demo/Prototype data for 170 Monitoring Wells in South-Central Nebraska
 * Study Area: Phelps County & adjacent South-Central Nebraska
 * Total Wells: 170 | Period: 2000–2024
 */

export const STUDY_AREA_INFO = {
  county: 'Phelps County, Nebraska, USA',
  region: 'South-Central Nebraska',
  totalWells: 170,
  totalObservations: 3844,
  period: '2000–2024',
  latRange: '40.3508° N – 40.6841° N',
  lonRange: '99.6432° W – 99.1795° W',
  aquifer: 'High Plains (Ogallala) Aquifer'
};

export function generateWellList() {
  const wells = [];
  const minLat = 40.3508, maxLat = 40.6841;
  const minLon = -99.6432, maxLon = -99.1795;
  const townships = ['Holdrege', 'Funk', 'Bertrand', 'Atlanta', 'Loomis', 'Westmark', 'Prairie'];

  // Categories distribution:
  // Stable/Low: 85 wells
  // Moderate: 42 wells
  // High: 19 wells
  // Critical: 24 wells
  // Total: 170 wells

  for (let i = 1; i <= 170; i++) {
    const id = `NE-PH-${String(i).padStart(3, '0')}`;
    const lat = minLat + (Math.sin(i * 1.618) * 0.5 + 0.5) * (maxLat - minLat);
    const lon = minLon + (Math.cos(i * 2.718) * 0.5 + 0.5) * (maxLon - minLon);
    const township = townships[i % townships.length];

    let riskLevel = 'Stable / Low Risk';
    let riskCategory = 'Stable';
    let depth = 16.4 + (i % 12) * 0.8;
    let predDepth = depth + 0.35;
    let color = '#10B981'; // Green
    let badgeColor = 'emerald';

    if (i <= 24) {
      riskLevel = 'Critical';
      riskCategory = 'Critical';
      depth = 66.8 + (i % 14) * 1.1;
      predDepth = depth + 0.85;
      color = '#EF4444'; // Red
      badgeColor = 'rose';
    } else if (i <= 43) {
      riskLevel = 'High Risk';
      riskCategory = 'High';
      depth = 42.5 + (i % 15) * 1.2;
      predDepth = depth + 0.65;
      color = '#F97316'; // Orange
      badgeColor = 'orange';
    } else if (i <= 85) {
      riskLevel = 'Moderate Risk';
      riskCategory = 'Moderate';
      depth = 24.2 + (i % 13) * 0.9;
      predDepth = depth + 0.42;
      color = '#EAB308'; // Yellow
      badgeColor = 'amber';
    }

    const rechargePotential = (i % 4 === 0) ? 'High' : ((i % 4 === 1 || i % 4 === 2) ? 'Moderate' : 'Low');

    wells.push({
      id: id,
      wellId: id,
      name: `Monitoring Station ${id}`,
      county: 'Phelps County',
      location: `${township}, Phelps Co., NE`,
      lat: parseFloat(lat.toFixed(4)),
      lon: parseFloat(lon.toFixed(4)),
      elevationM: Math.round(685 + (i % 40) * 1.8),
      waterDepthM: parseFloat(depth.toFixed(1)),
      predictedDepthM: parseFloat(predDepth.toFixed(1)),
      riskLevel: riskLevel,
      riskCategory: riskCategory,
      color: color,
      badgeColor: badgeColor,
      rechargePotential: rechargePotential,
      predictionStatus: 'Prediction Ready (Demo)',
      dataQuality: `${(90 + (i % 9)).toFixed(0)}% Good`,
      lastObservation: '2024-11-28',
      observationsCount: Math.round(20 + (i % 15) * 2),
      confidenceInterval: '90% PI: ±0.9m',
      historicalTrend: [
        depth - 2.8, depth - 2.2, depth - 1.7, depth - 1.1, depth - 0.4, depth
      ]
    });
  }

  return wells;
}

export const MOCK_WELLS = generateWellList();
