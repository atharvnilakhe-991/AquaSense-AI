import React, { useState, useEffect, useRef } from 'react';
import { MOCK_WELLS, STUDY_AREA_INFO } from './data/wellData';
import { GROUNDWATER_FACTS, KPI_CARDS_DATA, GROUNDWATER_TREND_SERIES, MOCK_EXPLORER_OBSERVATIONS } from './data/groundwaterData';
import { RISK_DISTRIBUTION_DATA, GROUNDWATER_ALERTS } from './data/riskData';
import { RECHARGE_CATEGORIES, RECHARGE_FACTORS } from './data/rechargeData';
import { AMDFE_MODALITIES, AMDFE_SYSTEM_STATS } from './data/amdfData';
import { ML_MODELS_DATA, AI_DECISION_SUPPORT_ITEMS, QUICK_ACCESS_MODULES, FAQ_DOCUMENTATION_ITEMS } from './data/modelData';

export default function App() {
  const [currentRoute, setCurrentRoute] = useState('/');
  const [selectedWell, setSelectedWell] = useState(MOCK_WELLS[3]); // Default NE-PH-004
  const [activeRiskFilter, setActiveRiskFilter] = useState('all');
  const [predTimeframe, setPredTimeframe] = useState('all');
  const [explorerSearch, setExplorerSearch] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifDropdownOpen, setNotifDropdownOpen] = useState(false);

  // Map refs
  const mapContainerRef = useRef(null);
  const leafletMapRef = useRef(null);
  const markersLayerRef = useRef(null);

  // Chart refs
  const dashChartCanvasRef = useRef(null);
  const dashChartInstance = useRef(null);
  const predChartCanvasRef = useRef(null);
  const predChartInstance = useRef(null);

  // Handle URL Hash navigation
  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.replace('#', '') || '/';
      setCurrentRoute(hash);
    };
    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  const navigate = (route) => {
    setCurrentRoute(route);
    window.location.hash = route === '/' ? '#/' : `#${route}`;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Initialize Map for /groundwater-map
  useEffect(() => {
    if (currentRoute === '/groundwater-map' && window.L && mapContainerRef.current) {
      if (!leafletMapRef.current) {
        const map = window.L.map(mapContainerRef.current, {
          zoomControl: true,
          attributionControl: false
        }).setView([40.5175, -99.4114], 11);

        window.L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
          maxZoom: 18,
          subdomains: 'abcd',
        }).addTo(map);

        window.L.rectangle([[40.3508, -99.6432], [40.6841, -99.1795]], {
          color: '#0284C7',
          weight: 1.5,
          dashArray: '4, 4',
          fillColor: '#0284C7',
          fillOpacity: 0.04
        }).addTo(map);

        const markersGroup = window.L.layerGroup().addTo(map);
        markersLayerRef.current = markersGroup;
        leafletMapRef.current = map;

        renderMapMarkers(MOCK_WELLS, markersGroup);
      } else {
        leafletMapRef.current.invalidateSize();
      }
    }
  }, [currentRoute]);

  // Update map markers when filter changes
  useEffect(() => {
    if (markersLayerRef.current) {
      if (activeRiskFilter === 'all') {
        renderMapMarkers(MOCK_WELLS, markersLayerRef.current);
      } else {
        const filtered = MOCK_WELLS.filter(w => w.riskCategory.toLowerCase() === activeRiskFilter.toLowerCase());
        renderMapMarkers(filtered, markersLayerRef.current);
      }
    }
  }, [activeRiskFilter]);

  const renderMapMarkers = (wellsList, layerGroup) => {
    layerGroup.clearLayers();
    wellsList.forEach(well => {
      const isCritical = well.riskCategory === 'Critical';
      const icon = window.L.divIcon({
        className: 'custom-well-marker',
        html: `<div style="width:14px; height:14px; border-radius:50%; background-color:${well.color}; border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,0.3);" class="${isCritical ? 'well-pulse' : ''}"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });

      const marker = window.L.marker([well.lat, well.lon], { icon });
      marker.on('click', () => setSelectedWell(well));
      marker.bindPopup(`
        <div style="font-family:Inter,sans-serif; font-size:11px; padding:2px;">
          <strong>${well.id}</strong> — <span style="color:${well.color}">${well.riskLevel}</span><br/>
          <span>${well.location}</span><br/>
          <span>Depth: <strong>${well.waterDepthM} m</strong></span><br/>
          <span>Forecast: <strong style="color:#7C3AED">${well.predictedDepthM} m</strong></span>
        </div>
      `);
      layerGroup.addLayer(marker);
    });
  };

  // Dashboard Chart
  useEffect(() => {
    if (currentRoute === '/dashboard' && window.Chart && dashChartCanvasRef.current) {
      if (dashChartInstance.current) dashChartInstance.current.destroy();
      const ctx = dashChartCanvasRef.current.getContext('2d');
      dashChartInstance.current = new window.Chart(ctx, {
        type: 'line',
        data: {
          labels: ['2000', '2004', '2008', '2012', '2016', '2020', '2024 (Now)', '2025 (Pred)'],
          datasets: [
            {
              label: 'Observed Depth (m)',
              data: [43.2, 44.5, 45.9, 47.4, 49.3, 50.1, 50.7, null],
              borderColor: '#0284C7',
              backgroundColor: 'rgba(2, 132, 199, 0.08)',
              borderWidth: 2.5,
              tension: 0.3,
              fill: true
            },
            {
              label: 'ML Forecast',
              data: [null, null, null, null, null, null, 50.7, 52.3],
              borderColor: '#7C3AED',
              borderDash: [5, 5],
              borderWidth: 2.5,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } },
            y: { min: 40, max: 58, grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } }
          }
        }
      });
    }
  }, [currentRoute]);

  // Prediction Analysis Chart
  useEffect(() => {
    if (currentRoute === '/prediction-analysis' && window.Chart && predChartCanvasRef.current) {
      if (predChartInstance.current) predChartInstance.current.destroy();
      const ctx = predChartCanvasRef.current.getContext('2d');
      const series = GROUNDWATER_TREND_SERIES[predTimeframe] || GROUNDWATER_TREND_SERIES.all;

      predChartInstance.current = new window.Chart(ctx, {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            {
              label: 'Observed Depth (m)',
              data: series.observed,
              borderColor: '#0284C7',
              backgroundColor: 'rgba(2, 132, 199, 0.08)',
              borderWidth: 2.5,
              tension: 0.3,
              pointRadius: 4,
              fill: true
            },
            {
              label: 'ML Forecast Horizon (90% PI)',
              data: series.predicted,
              borderColor: '#7C3AED',
              borderDash: [6, 4],
              borderWidth: 2.5,
              pointRadius: 5,
              fill: false
            },
            {
              label: 'Critical Threshold (>55m)',
              data: series.criticalThreshold,
              borderColor: 'rgba(239, 68, 68, 0.4)',
              borderDash: [4, 4],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } },
            y: { min: 40, max: 58, grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } }
          }
        }
      });
    }
  }, [currentRoute, predTimeframe]);

  const handleExportCSV = (records, filename) => {
    const csv = "data:text/csv;charset=utf-8," + records.map(r => Object.values(r).join(",")).join("\n");
    const encodedUri = encodeURI(csv);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const navItems = [
    { label: 'Homepage', route: '/' },
    { label: 'Dashboard', route: '/dashboard' },
    { label: 'Groundwater Map', route: '/groundwater-map' },
    { label: 'Prediction Analysis', route: '/prediction-analysis' },
    { label: 'Recharge Analysis', route: '/recharge-analysis' },
    { label: 'Risk Assessment', route: '/risk-assessment' },
    { label: 'Data Explorer', route: '/data-explorer' },
    { label: 'Model Performance', route: '/model-performance' },
    { label: 'About', route: '/about' },
    { label: 'Help', route: '/help' },
  ];

  const isHomepage = currentRoute === '/';

  return (
    <div className={`min-h-screen antialiased font-sans ${isHomepage ? 'bg-[#050914] text-white' : 'bg-slate-50 text-slate-800'}`}>
      
      {/* ======================================================== */}
      {/*GLOBAL HEADER WITH 10 ROUTES                             */}
      {/* ======================================================== */}
      <header className={`sticky top-0 z-50 transition-all duration-200 ${
        isHomepage ? 'bg-[#050914]/90 backdrop-blur-md border-b border-white/10 text-white' : 'bg-white/95 backdrop-blur-md border-b border-slate-200 text-slate-800'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            
            {/* Brand Logo */}
            <div className="flex items-center space-x-3 cursor-pointer shrink-0" onClick={() => navigate('/')}>
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-600 to-teal-500 flex items-center justify-center text-white shadow-xs">
                💧
              </div>
              <div>
                <span className={`text-base font-bold tracking-tight ${isHomepage ? 'text-white' : 'text-slate-900'}`}>
                  AquaSense<span className="text-sky-500">.AI</span>
                </span>
                <p className="text-[10px] text-slate-400 font-medium leading-none">Groundwater Intelligence</p>
              </div>
            </div>

            {/* Desktop Horizontal Navigation (All 10 Routes) */}
            <nav className="hidden xl:flex items-center space-x-1 overflow-x-auto py-1">
              {navItems.map(item => (
                <button
                  key={item.route}
                  onClick={() => navigate(item.route)}
                  className={`px-3 py-1.5 text-xs rounded-lg transition-colors whitespace-nowrap ${
                    currentRoute === item.route
                      ? (isHomepage ? 'bg-white/10 text-white font-bold border border-white/20' : 'bg-sky-50 text-sky-700 font-semibold border border-sky-200')
                      : (isHomepage ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100')
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>

            {/* Right Icons */}
            <div className="flex items-center space-x-2 shrink-0">
              <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 bg-emerald-50 border border-emerald-200 rounded-full text-emerald-700 text-[11px] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>LIVE (170 WELLS)</span>
              </div>

              <button onClick={() => setNotifDropdownOpen(!notifDropdownOpen)} className="p-2 text-slate-400 hover:text-slate-600 rounded-lg relative">
                🔔
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full"></span>
              </button>

              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-sky-600 to-teal-500 text-white flex items-center justify-center font-bold text-xs">
                EV
              </div>

              <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="xl:hidden p-2 text-slate-400 hover:text-white">
                ☰
              </button>
            </div>

          </div>
        </div>

        {/* Mobile Navigation Dropdown */}
        {mobileMenuOpen && (
          <div className="xl:hidden bg-white text-slate-800 border-b border-slate-200 px-4 pt-2 pb-4 space-y-1 text-xs font-medium">
            {navItems.map(item => (
              <button
                key={item.route}
                onClick={() => { navigate(item.route); setMobileMenuOpen(false); }}
                className={`block w-full text-left py-2 px-3 rounded-lg ${
                  currentRoute === item.route ? 'bg-sky-50 text-sky-700 font-bold' : 'hover:bg-slate-50 text-slate-700'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </header>

      {/* Notifications Dropdown */}
      {notifDropdownOpen && (
        <div className="fixed top-16 right-4 sm:right-8 z-50 w-80 bg-white border border-slate-200 rounded-xl shadow-xl p-4 text-xs text-slate-800">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="font-bold text-slate-800 text-sm">System Alerts</span>
            <span className="px-1.5 py-0.5 bg-rose-100 text-rose-700 font-semibold rounded text-[10px]">3 Active</span>
          </div>
          <div className="mt-3 space-y-2.5">
            <div className="p-2 bg-rose-50 border border-rose-100 rounded-lg">
              <p className="font-semibold text-rose-900">Gosper Uplands Over-Extraction</p>
              <p className="text-slate-600 text-[11px]">Drawdown stress ratio 1.68 detected.</p>
            </div>
            <div className="p-2 bg-amber-50 border border-amber-100 rounded-lg">
              <p className="font-semibold text-amber-900">Phelps Central Vadose Drying</p>
              <p className="text-slate-600 text-[11px]">SMAP soil moisture lag detected.</p>
            </div>
          </div>
        </div>
      )}

      {/* ======================================================== */}
      {/* 1. HOMEPAGE (KEEP IT VERY SIMPLE & CINEMATIC)            */}
      {/* ======================================================== */}
      {currentRoute === '/' && (
        <div className="relative min-h-[calc(100vh-4rem)] flex flex-col justify-between py-16 px-4 sm:px-6 lg:px-8 bg-cover bg-right" style={{
          backgroundImage: `radial-gradient(circle at 75% 45%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
            linear-gradient(to right, rgba(5, 9, 20, 0.95) 0%, rgba(5, 9, 20, 0.70) 50%, rgba(5, 9, 20, 0.40) 80%, rgba(5, 9, 20, 0.85) 100%),
            linear-gradient(to top, rgba(5, 9, 20, 1) 0%, transparent 20%),
            url('file:///C:/Users/Lenovo/.gemini/antigravity-ide/brain/fb617959-419c-408d-971c-56b7b4b8cbdb/.user_uploaded/media_1790189505246.jpg'),
            url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')`
        }}>
          <div className="max-w-4xl mx-auto my-auto text-left w-full">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono tracking-widest uppercase mb-6">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Hyperlocal Groundwater Intelligence</span>
            </div>

            <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight text-white leading-none">
              AquaSense<span className="text-[#00E5FF]">AI</span>
            </h1>

            <p className="mt-4 text-xl sm:text-2xl font-semibold text-slate-200 max-w-2xl">
              Adaptive Multimodal AI for Hyperlocal Groundwater Prediction, Recharge Assessment & Decision Support
            </p>

            <p className="mt-4 text-sm sm:text-base text-slate-300/80 max-w-xl leading-relaxed">
              Combining in-situ piezometers, GPM weather radar, Sentinel-2 optical imagery, and SMAP soil moisture grids across the High Plains Aquifer in Phelps County, Nebraska.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-4">
              <button 
                onClick={() => navigate('/dashboard')} 
                className="px-7 py-3.5 rounded-full bg-gradient-to-r from-[#00E5FF] to-sky-500 hover:from-cyan-300 hover:to-sky-400 text-slate-950 font-bold text-sm tracking-wide shadow-lg shadow-cyan-500/30 transition-all flex items-center space-x-2"
              >
                <span>Explore Dashboard</span>
                <span>→</span>
              </button>
              
              <button 
                onClick={() => navigate('/groundwater-map')} 
                className="px-6 py-3.5 rounded-full bg-white/10 hover:bg-white/15 border border-white/20 text-white font-medium text-sm transition-all"
              >
                Spatial GIS Map
              </button>
            </div>
          </div>

          <div className="max-w-4xl mx-auto w-full pt-8 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-4">
            <span>South-Central Nebraska • 170 Monitored Stations • 2000–2024</span>
            <span className="italic font-serif">"Data from Earth. Insights for a Sustainable Tomorrow."</span>
          </div>
        </div>
      )}

      {/* ======================================================== */}
      {/* 2. DASHBOARD VIEW (WHITE SAAS THEME)                     */}
      {/* ======================================================== */}
      {currentRoute === '/dashboard' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Groundwater Intelligence Overview</h1>
            <p className="text-sm text-slate-500 mt-1">Hyperlocal groundwater monitoring, prediction and decision support across Phelps County, Nebraska</p>
          </div>

          {/* 6 Large Rounded Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
            {KPI_CARDS_DATA.map(kpi => (
              <div key={kpi.id} className="saas-card p-5 bg-white border border-slate-200 rounded-2xl shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">{kpi.title}</span>
                  <span className={`px-2 py-0.5 text-[10px] font-bold rounded-md border ${
                    kpi.statusType === 'critical' ? 'bg-rose-50 text-rose-700 border-rose-200' :
                    kpi.statusType === 'warning' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                    kpi.statusType === 'info' ? 'bg-purple-50 text-purple-700 border-purple-200' :
                    'bg-emerald-50 text-emerald-700 border-emerald-200'
                  }`}>
                    {kpi.badgeText}
                  </span>
                </div>
                <p className="text-3xl font-bold text-slate-900 mt-3">{kpi.value} <span className="text-xs font-normal text-slate-500">{kpi.unit}</span></p>
                <p className="text-xs text-slate-500 mt-1 font-medium">{kpi.description}</p>
              </div>
            ))}
          </div>

          {/* Overview Sections (Trend Chart + AMDFE Quality) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-900">Groundwater Level Trend & Decadal Variance</h2>
                  <p className="text-xs text-slate-500">25-year historical drawdown with ML forward forecast envelope</p>
                </div>
                <button onClick={() => navigate('/prediction-analysis')} className="text-xs font-semibold text-sky-600 hover:text-sky-700">Detailed View →</button>
              </div>
              <div className="h-64 w-full">
                <canvas ref={dashChartCanvasRef}></canvas>
              </div>
            </div>

            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <h2 className="text-base font-bold text-slate-900">AMDFE Multimodal Data Quality & SNR Reliability</h2>
                  <span className="px-2.5 py-0.5 bg-teal-50 text-teal-700 text-xs font-bold rounded-full">93.8% Mean Quality</span>
                </div>
                <p className="text-xs text-slate-500 mt-1">Dynamic reliability weighting across 5 heterogeneous sensor modalities</p>

                <div className="mt-5 space-y-3.5 text-xs">
                  {AMDFE_MODALITIES.map(mod => (
                    <div key={mod.name}>
                      <div className="flex justify-between mb-1"><span class="font-medium text-slate-700">{mod.name}</span><span class="font-bold text-slate-900">{mod.quality}%</span></div>
                      <div className="w-full bg-slate-100 rounded-full h-2"><div className="h-2 rounded-full" style={{ backgroundColor: mod.color, width: `${mod.quality}%` }}></div></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 3. GROUNDWATER MAP VIEW (DEDICATED GIS MAP)              */}
      {/* ======================================================== */}
      {currentRoute === '/groundwater-map' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Groundwater Map</h1>
            <p className="text-sm text-slate-500 mt-1">Interactive spatial monitoring of 170 wells across Phelps County, South-Central Nebraska</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 saas-card p-4 bg-white border border-slate-200 rounded-2xl space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-700">Filter By Risk:</span>
                  {['all', 'critical', 'high', 'moderate', 'stable'].map(cat => (
                    <button
                      key={cat}
                      onClick={() => setActiveRiskFilter(cat)}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold capitalize ${
                        activeRiskFilter === cat ? 'bg-sky-600 text-white' : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-xl overflow-hidden border border-slate-200">
                <div ref={mapContainerRef} className="w-full h-[520px] bg-slate-100"></div>
              </div>
            </div>

            {/* Right Information Panel */}
            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                  <h2 className="text-base font-bold text-slate-900">Well Information Panel</h2>
                  <span className="px-2 py-0.5 text-xs font-bold rounded-md" style={{ backgroundColor: `${selectedWell.color}20`, color: selectedWell.color }}>
                    {selectedWell.riskLevel}
                  </span>
                </div>

                <div className="mt-4 space-y-4 text-xs">
                  <div><span className="text-slate-400">Station ID:</span><p className="text-lg font-bold text-slate-900">{selectedWell.id}</p></div>
                  <div><span className="text-slate-400">Location:</span><p className="font-medium text-slate-800">{selectedWell.location}</p></div>
                  <div><span className="text-slate-400">Coordinates:</span><p className="font-mono text-slate-800">{selectedWell.lat}° N, {Math.abs(selectedWell.lon)}° W</p></div>
                  <div><span className="text-slate-400">Depth to Groundwater:</span><p className="text-xl font-bold text-slate-900">{selectedWell.waterDepthM} m</p></div>
                  <div><span className="text-slate-400">30d ML Prediction:</span><p className="text-base font-bold text-purple-700">{selectedWell.predictedDepthM} m</p></div>
                  <div><span className="text-slate-400">Recharge Capacity:</span><p className="font-semibold text-teal-700">{selectedWell.rechargePotential}</p></div>
                </div>
              </div>

              <button onClick={() => navigate('/prediction-analysis')} className="w-full py-2 bg-sky-50 hover:bg-sky-100 text-sky-700 font-bold rounded-xl text-xs transition-colors mt-6">
                Analyze Hydrograph →
              </button>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 4. PREDICTION ANALYSIS VIEW                              */}
      {/* ======================================================== */}
      {currentRoute === '/prediction-analysis' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Prediction Analysis</h1>
            <p className="text-sm text-slate-500 mt-1">AI-based groundwater level prediction and trend analysis across multiple forecast horizons</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl">
              <span className="text-xs font-semibold text-slate-500">Observed Water Depth</span>
              <p className="text-3xl font-bold text-slate-900 mt-2">50.7 m</p>
              <p className="text-xs text-slate-500 mt-1">2024 Observed Baseline</p>
            </div>
            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl">
              <span className="text-xs font-semibold text-slate-500">30-Day ML Prediction</span>
              <p className="text-3xl font-bold text-purple-700 mt-2">52.3 m</p>
              <p className="text-xs text-purple-600 mt-1">Projected Drawdown</p>
            </div>
            <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl">
              <span className="text-xs font-semibold text-slate-500">90% Prediction Interval</span>
              <p className="text-3xl font-bold text-sky-700 mt-2">±0.9 m</p>
              <p className="text-xs text-slate-500 mt-1">Uncertainty Envelope: 50.1 – 54.5 m</p>
            </div>
          </div>

          <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900">Observed vs. Predicted Groundwater Level</h2>
                <p className="text-xs text-slate-500">Historical decadal observations vs. Forward ML trajectory</p>
              </div>
              <div className="flex space-x-1 bg-slate-100 p-1 rounded-xl text-xs">
                {['all', 'fiveYears', 'oneYear'].map(tf => (
                  <button
                    key={tf}
                    onClick={() => setPredTimeframe(tf)}
                    className={`px-3 py-1 rounded-lg ${predTimeframe === tf ? 'bg-white font-bold text-sky-700 shadow-xs' : 'text-slate-600'}`}
                  >
                    {tf === 'all' ? 'All (2000–2024)' : tf === 'fiveYears' ? '5 Years' : '1 Year'}
                  </button>
                ))}
              </div>
            </div>
            <div className="h-72 w-full">
              <canvas ref={predChartCanvasRef}></canvas>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 5. RECHARGE ANALYSIS VIEW                                */}
      {/* ======================================================== */}
      {currentRoute === '/recharge-analysis' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Recharge Analysis</h1>
            <p className="text-sm text-slate-500 mt-1">Assessment of groundwater recharge potential after rainfall events across the vadose zone</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="saas-card p-5 bg-white border border-slate-200 rounded-2xl border-l-4 border-l-teal-600">
              <span className="text-xs font-semibold text-slate-500">Recharge Potential</span>
              <p className="text-3xl font-bold text-teal-700 mt-2">76 / 100</p>
              <p className="text-xs text-teal-600 font-medium mt-1">Optimal Infiltration Corridor</p>
            </div>
            <div className="saas-card p-5 bg-white border border-slate-200 rounded-2xl border-l-4 border-l-sky-500">
              <span className="text-xs font-semibold text-slate-500">Annual Rainfall</span>
              <p className="text-3xl font-bold text-slate-900 mt-2">540 mm</p>
              <p className="text-xs text-slate-500 mt-1">GPM IMERG Radar</p>
            </div>
            <div className="saas-card p-5 bg-white border border-slate-200 rounded-2xl border-l-4 border-l-amber-500">
              <span className="text-xs font-semibold text-slate-500">Vadose Soil Moisture</span>
              <p className="text-3xl font-bold text-slate-900 mt-2">0.28 m³/m³</p>
              <p className="text-xs text-slate-500 mt-1">SMAP L4 Radiometer</p>
            </div>
            <div className="saas-card p-5 bg-white border border-slate-200 rounded-2xl border-l-4 border-l-emerald-500">
              <span className="text-xs font-semibold text-slate-500">Conductivity Score</span>
              <p className="text-3xl font-bold text-emerald-700 mt-2">84%</p>
              <p className="text-xs text-emerald-600 mt-1">Platte Canal Alluvial Margins</p>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 6. RISK ASSESSMENT VIEW                                  */}
      {/* ======================================================== */}
      {currentRoute === '/risk-assessment' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Risk Assessment</h1>
            <p className="text-sm text-slate-500 mt-1">Identify groundwater depletion risk, over-extracted wells, and monitoring priorities</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {RISK_DISTRIBUTION_DATA.map(item => (
              <div key={item.category} className="saas-card p-5 bg-white border border-slate-200 rounded-2xl">
                <span className="text-xs font-semibold text-slate-500">{item.category}</span>
                <p className="text-3xl font-bold text-slate-900 mt-2">{item.count} Wells</p>
                <p className="text-xs text-slate-400 mt-1">{item.threshold} • {item.percentage}%</p>
              </div>
            ))}
          </div>

          <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-4">
            <h2 className="text-base font-bold text-slate-900">Priority Monitoring & Depletion Risk Registry</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 uppercase text-[10px]">
                    <th className="py-2.5">Well ID</th>
                    <th className="py-2.5">Location</th>
                    <th className="py-2.5">Groundwater Level</th>
                    <th className="py-2.5">Risk Tier</th>
                    <th className="py-2.5">Monitoring Priority</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {MOCK_WELLS.slice(0, 8).map(w => (
                    <tr key={w.id}>
                      <td className="py-3 font-bold text-slate-900">{w.id}</td>
                      <td>{w.location}</td>
                      <td className="font-bold text-slate-900">{w.waterDepthM} m</td>
                      <td><span className="px-2 py-0.5 rounded font-bold text-[10px]" style={{ backgroundColor: `${w.color}20`, color: w.color }}>{w.riskLevel}</span></td>
                      <td className="font-medium text-slate-700">{w.riskCategory === 'Critical' ? 'Priority 1 (Urgent)' : 'Priority 3 (Periodic)'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 7. DATA EXPLORER VIEW                                    */}
      {/* ======================================================== */}
      {currentRoute === '/data-explorer' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Data Explorer</h1>
              <p className="text-sm text-slate-500 mt-1">Explore 3,844 curated groundwater and hydro-meteorological observations (2000–2024)</p>
            </div>
            <button 
              onClick={() => handleExportCSV(MOCK_EXPLORER_OBSERVATIONS, "aquasense_explorer_observations.csv")}
              className="px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white font-bold text-xs rounded-xl"
            >
              Export Filtered Records (CSV)
            </button>
          </div>

          <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-4">
            <input 
              type="text" 
              placeholder="Search by Well ID..." 
              value={explorerSearch} 
              onChange={(e) => setExplorerSearch(e.target.value)}
              className="w-full sm:w-80 px-3 py-1.5 border border-slate-200 rounded-lg text-xs"
            />
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 uppercase text-[10px]">
                    <th className="py-2.5">Obs ID</th>
                    <th className="py-2.5">Well ID</th>
                    <th className="py-2.5">Date</th>
                    <th className="py-2.5">Water Depth</th>
                    <th className="py-2.5">Temp</th>
                    <th className="py-2.5">Rain</th>
                    <th className="py-2.5">Soil</th>
                    <th className="py-2.5">Quality</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {MOCK_EXPLORER_OBSERVATIONS.filter(o => !explorerSearch || o.wellId.toLowerCase().includes(explorerSearch.toLowerCase())).map(r => (
                    <tr key={r.id}>
                      <td className="py-3 font-mono text-slate-600">{r.id}</td>
                      <td className="font-bold text-sky-700">{r.wellId}</td>
                      <td>{r.date}</td>
                      <td className="font-bold text-slate-900">{r.level} m</td>
                      <td>{r.temp}</td>
                      <td>{r.rain}</td>
                      <td>{r.soil}</td>
                      <td><span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 text-[10px]">{r.quality}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 8. MODEL PERFORMANCE VIEW                                */}
      {/* ======================================================== */}
      {currentRoute === '/model-performance' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Model Performance</h1>
            <p className="text-sm text-slate-500 mt-1">Evaluation of machine learning groundwater level prediction models</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {ML_MODELS_DATA.map(model => (
              <div key={model.id} className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-3">
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${model.badgeClass}`}>{model.name}</span>
                <h3 className="font-bold text-slate-900">{model.architecture}</h3>
                <div className="space-y-1 text-xs pt-2 border-t border-slate-100">
                  <div className="flex justify-between"><span>R² Score:</span><strong className="text-slate-900">{model.r2}</strong></div>
                  <div className="flex justify-between"><span>RMSE:</span><strong className="text-slate-900">{model.rmse}</strong></div>
                  <div className="flex justify-between"><span>MAE:</span><strong className="text-slate-900">{model.mae}</strong></div>
                </div>
              </div>
            ))}
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 9. ABOUT VIEW                                            */}
      {/* ======================================================== */}
      {currentRoute === '/about' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">About AquaSense AI</h1>
            <p className="text-sm text-slate-500 mt-1">Hyperlocal Groundwater Intelligence Platform</p>
          </div>

          <div className="saas-card p-8 bg-white border border-slate-200 rounded-2xl space-y-4">
            <h2 className="text-lg font-bold text-slate-900">Core Research & Operational Concept</h2>
            <p className="text-slate-600 text-sm leading-relaxed">
              AquaSense AI combines groundwater observations with environmental and geospatial information to support hyperlocal groundwater monitoring, prediction, recharge assessment, risk identification, and explainable decision support across the High Plains Aquifer in Phelps County, Nebraska.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {QUICK_ACCESS_MODULES.map(m => (
              <div key={m.id} onClick={() => navigate(m.path)} className="saas-card p-6 bg-white border border-slate-200 rounded-2xl cursor-pointer hover:border-sky-300">
                <span className="text-2xl">📌</span>
                <h3 className="font-bold text-slate-900 text-sm mt-3">{m.title}</h3>
                <p className="text-xs text-slate-500 mt-2">{m.description}</p>
              </div>
            ))}
          </div>
        </main>
      )}

      {/* ======================================================== */}
      {/* 10. HELP VIEW                                            */}
      {/* ======================================================== */}
      {currentRoute === '/help' && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Help & Documentation</h1>
            <p className="text-sm text-slate-500 mt-1">User guides, platform methodologies, and frequently asked questions</p>
          </div>

          <div className="saas-card p-6 bg-white border border-slate-200 rounded-2xl space-y-4">
            <h2 className="text-base font-bold text-slate-900">Frequently Asked Questions</h2>
            <div className="space-y-3 text-xs">
              {FAQ_DOCUMENTATION_ITEMS.map(faq => (
                <div key={faq.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                  <h3 className="font-bold text-slate-900 text-sm">{faq.question}</h3>
                  <p className="mt-2 text-slate-600 leading-relaxed">{faq.answer}</p>
                </div>
              ))}
            </div>
          </div>
        </main>
      )}

    </div>
  );
}
