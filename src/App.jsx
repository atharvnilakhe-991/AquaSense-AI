import React, { useState, useEffect, useRef } from "react";
import { MOCK_WELLS, STUDY_AREA_INFO } from "./data/wellData.js";
import { GROUNDWATER_FACTS, KPI_CARDS_DATA, GROUNDWATER_TREND_SERIES, MOCK_EXPLORER_OBSERVATIONS } from "./data/groundwaterData.js";
import { RISK_DISTRIBUTION_DATA, GROUNDWATER_ALERTS } from "./data/riskData.js";
import { RECHARGE_CATEGORIES, RECHARGE_FACTORS } from "./data/rechargeData.js";
import { AMDFE_MODALITIES, AMDFE_SYSTEM_STATS } from "./data/amdfData.js";
import { ML_MODELS_DATA, AI_DECISION_SUPPORT_ITEMS, QUICK_ACCESS_MODULES, FAQ_DOCUMENTATION_ITEMS } from "./data/modelData.js";

export default function App() {
  const [currentRoute, setCurrentRoute] = useState("/");
  const [selectedWell, setSelectedWell] = useState(MOCK_WELLS[3]);
  const [activeRiskFilter, setActiveRiskFilter] = useState("all");
  const [predTimeframe, setPredTimeframe] = useState("all");
  const [explorerSearch, setExplorerSearch] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifModalOpen, setNotifModalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  const mapContainerRef = useRef(null);
  const leafletMapRef = useRef(null);
  const markersLayerRef = useRef(null);
  const dashChartCanvasRef = useRef(null);
  const dashChartInstance = useRef(null);
  const predChartCanvasRef = useRef(null);
  const predChartInstance = useRef(null);

  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.replace("#", "") || "/";
      setCurrentRoute(hash);
    };
    handleHash();
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  const navigate = (route) => {
    setCurrentRoute(route);
    window.location.hash = route === "/" ? "#/" : `#${route}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 antialiased font-sans flex flex-col justify-between selection:bg-sky-500/20 selection:text-sky-900">

      {currentRoute === "/" && (
        <section
          style={{
            position: "relative",
            width: "100vw",
            height: "100vh",
            minHeight: "700px",
            overflow: "hidden",
            backgroundColor: "#030712",
            color: "#ffffff",
            fontFamily: "'Inter', sans-serif",
            userSelect: "none"
          }}
        >


          {/* LAYER 2 — NAVBAR */}
          <nav
            style={{
              position: "absolute", top: 0, left: 0, right: 0,
              padding: "22px 40px 12px 40px",
              display: "flex", alignItems: "center", justifyContent: "space-between",
              zIndex: 30
            }}
          >
            {/* Logo */}
            <div
              style={{ display: "flex", alignItems: "center", gap: "12px", cursor: "pointer", flexShrink: 0 }}
              onClick={() => navigate("/")}
            >
              <div
                style={{
                  width: "40px", height: "40px", borderRadius: "50%",
                  background: "rgba(3,7,18,0.80)",
                  border: "1px solid rgba(0,229,255,0.6)",
                  boxShadow: "0 0 15px rgba(0,229,255,0.40)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  padding: "8px", flexShrink: 0
                }}
              >
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00e5ff">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8"
                    d="M12 2.5C8.5 7.5 6 11 6 15a6 6 0 0012 0c0-4-2.5-7.5-6-12.5z"
                    fill="#00e5ff" fillOpacity="0.2" />
                  <circle cx="12" cy="14" r="2.5" fill="#00e5ff" />
                </svg>
              </div>
              <div>
                <div style={{ display: "flex", alignItems: "baseline", lineHeight: 1 }}>
                  <span style={{ fontSize: "22px", fontWeight: 900, letterSpacing: "-0.02em", color: "#ffffff", fontFamily: "'Inter', sans-serif" }}>AquaSense</span>
                  <span style={{ fontSize: "22px", fontWeight: 900, letterSpacing: "-0.02em", color: "#00e5ff", fontFamily: "'Inter', sans-serif" }}>AI</span>
                </div>
                <p style={{ fontSize: "11px", fontWeight: 500, color: "#94a3b8", letterSpacing: "0.10em", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>
                  Groundwater Intelligence
                </p>
              </div>
            </div>

            {/* Center nav links */}
            <div style={{ display: "flex", alignItems: "center", gap: "28px" }}>
              <button
                onClick={() => navigate("/")}
                style={{
                  fontSize: "14px", fontWeight: 600, color: "#00e5ff",
                  background: "none", border: "none", borderBottom: "2px solid #00e5ff",
                  cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                  fontFamily: "'Inter', sans-serif"
                }}
              >
                Home
              </button>
              {[
                { label: "Dashboard", action: () => navigate("/dashboard") },
                { label: "About",     action: () => navigate("/about") },
                { label: "Features",  action: () => navigate("/prediction-analysis") },
                { label: "Team",      action: () => setSettingsModalOpen(true) },
                { label: "Contact",   action: () => setNotifModalOpen(true) }
              ].map(({ label, action }) => (
                <button
                  key={label}
                  onClick={action}
                  style={{
                    fontSize: "14px", fontWeight: 400, color: "#cbd5e1",
                    background: "none", border: "none", cursor: "pointer",
                    lineHeight: 1, fontFamily: "'Inter', sans-serif", transition: "color 0.2s"
                  }}
                  onMouseEnter={e => e.currentTarget.style.color = "#ffffff"}
                  onMouseLeave={e => e.currentTarget.style.color = "#cbd5e1"}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Right: Live Monitoring + Get Started */}
            <div style={{ display: "flex", alignItems: "center", gap: "22px", flexShrink: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div
                  style={{
                    width: "12px", height: "12px", borderRadius: "50%",
                    border: "2px solid #10b981",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    boxShadow: "0 0 8px rgba(16,185,129,0.7)", flexShrink: 0
                  }}
                >
                  <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: "#10b981", display: "block" }} />
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{
                    fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 700, letterSpacing: "0.16em", color: "#10b981",
                    lineHeight: 1, textShadow: "0 0 8px rgba(16,185,129,0.5)", textTransform: "uppercase"
                  }}>LIVE MONITORING</span>
                  <span style={{ fontSize: "11px", color: "#cbd5e1", marginTop: "3px", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>
                    South-Central Nebraska
                  </span>
                </div>
              </div>
              <button
                onClick={() => navigate("/dashboard")}
                className="hero-btn-primary"
                style={{
                  padding: "10px 24px", borderRadius: "9999px",
                  fontWeight: 700, fontSize: "14px", letterSpacing: "0.02em",
                  color: "#ffffff", display: "flex", alignItems: "center",
                  gap: "8px", cursor: "pointer", border: "none",
                  fontFamily: "'Inter', sans-serif", flexShrink: 0
                }}
              >
                <span>Get Started</span>
                <span style={{ fontSize: "15px" }}>→</span>
              </button>
            </div>
          </nav>

          {/* LAYERS 4–6 — HERO TEXT + BUTTONS + STATISTICS */}
          <div
            style={{
              position: "absolute", top: "50%", left: "40px",
              transform: "translateY(-52%)", zIndex: 20,
              maxWidth: "500px", pointerEvents: "auto"
            }}
          >
            {/* Eyebrow */}
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "10.5px", fontWeight: 600, letterSpacing: "0.26em",
              color: "rgba(148,163,184,0.75)", textTransform: "uppercase", marginBottom: "20px"
            }}>
              MONITOR &nbsp; PREDICT &nbsp; RECHARGE &nbsp; SUSTAIN
            </div>

            {/* Main heading */}
            <h1 style={{
              fontSize: "clamp(52px,7vw,78px)", fontWeight: 900,
              letterSpacing: "-0.025em", lineHeight: 0.95,
              margin: "0 0 14px 0", fontFamily: "'Inter', sans-serif"
            }}>
              <span style={{ color: "#ffffff" }}>AquaSense</span>
              <span style={{ color: "#00e5ff" }}>AI</span>
            </h1>

            {/* Subtitle */}
            <h2 style={{
              fontSize: "clamp(19px,2.6vw,29px)", fontWeight: 700,
              color: "#ffffff", letterSpacing: "-0.01em", lineHeight: 1.22,
              margin: "0 0 18px 0", fontFamily: "'Inter', sans-serif"
            }}>
              Hyperlocal Groundwater<br />Intelligence Platform
            </h2>

            {/* Description */}
            <p style={{
              fontSize: "14.5px", lineHeight: 1.70,
              color: "rgba(148,163,184,0.90)", maxWidth: "415px",
              margin: "0 0 30px 0", fontFamily: "'Inter', sans-serif"
            }}>
              AI-powered groundwater monitoring, prediction, recharge<br />
              assessment and explainable decision support for a more<br />
              sustainable tomorrow.
            </p>

            {/* CTA buttons */}
            <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "34px" }}>
              <button
                onClick={() => navigate("/dashboard")}
                className="hero-btn-primary"
                style={{
                  padding: "13px 30px", borderRadius: "9999px",
                  fontSize: "14px", fontWeight: 700, letterSpacing: "0.02em",
                  display: "flex", alignItems: "center", gap: "8px",
                  cursor: "pointer", border: "none", color: "#ffffff",
                  fontFamily: "'Inter', sans-serif"
                }}
              >
                <span>Explore Dashboard</span>
                <span style={{ fontSize: "16px", lineHeight: 1 }}>→</span>
              </button>
              <button
                onClick={() => setSettingsModalOpen(true)}
                className="hero-btn-secondary"
                style={{
                  padding: "12px 24px", borderRadius: "9999px",
                  fontSize: "14px", fontWeight: 600,
                  display: "flex", alignItems: "center", gap: "10px",
                  cursor: "pointer", fontFamily: "'Inter', sans-serif"
                }}
              >
                <span style={{ fontSize: "13px", lineHeight: 1 }}>▶</span>
                <span>Watch Overview</span>
              </button>
            </div>

            {/* Statistics row */}
            <div style={{ display: "flex", alignItems: "center", flexWrap: "nowrap", gap: "18px" }}>

              <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
                <svg width="20" height="20" fill="none" stroke="#00e5ff" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <div>
                  <p style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: "22px", color: "#ffffff", lineHeight: 1, margin: 0 }}>170</p>
                  <p style={{ fontSize: "10px", color: "#94a3b8", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>Monitoring Wells</p>
                </div>
              </div>

              <div style={{ width: "1px", height: "34px", background: "rgba(255,255,255,0.15)", flexShrink: 0 }} />

              <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
                <svg width="20" height="20" fill="none" stroke="#00e5ff" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
                <div>
                  <p style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: "22px", color: "#ffffff", lineHeight: 1, margin: 0 }}>3,844</p>
                  <p style={{ fontSize: "10px", color: "#94a3b8", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>Observations</p>
                </div>
              </div>

              <div style={{ width: "1px", height: "34px", background: "rgba(255,255,255,0.15)", flexShrink: 0 }} />

              <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
                <svg width="20" height="20" fill="none" stroke="#00e5ff" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <div>
                  <p style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: "22px", color: "#ffffff", lineHeight: 1, margin: 0 }}>2000 – 2024</p>
                  <p style={{ fontSize: "10px", color: "#94a3b8", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>Dataset Period</p>
                </div>
              </div>

              <div style={{ width: "1px", height: "34px", background: "rgba(255,255,255,0.15)", flexShrink: 0 }} />

              <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
                <svg width="20" height="20" fill="none" stroke="#00e5ff" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                  <rect x="9" y="3" width="6" height="5" rx="1" strokeWidth="2" />
                  <rect x="3" y="16" width="6" height="5" rx="1" strokeWidth="2" />
                  <rect x="15" y="16" width="6" height="5" rx="1" strokeWidth="2" />
                  <path d="M12 8v4M6 16v-4h12v4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div>
                  <p style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: "22px", color: "#ffffff", lineHeight: 1, margin: 0 }}>AMDFE</p>
                  <p style={{ fontSize: "10px", color: "#94a3b8", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>Enabled</p>
                </div>
              </div>

            </div>
          </div>

          {/* LAYER 7a — SCROLL DOWN indicator */}
          <div
            onClick={() => navigate("/dashboard")}
            style={{
              position: "absolute", bottom: "28px", left: "50%",
              transform: "translateX(-50%)", zIndex: 20,
              cursor: "pointer", display: "flex", flexDirection: "column",
              alignItems: "center", gap: "6px", opacity: 0.72,
              userSelect: "none", pointerEvents: "auto"
            }}
          >
            <div style={{
              width: "18px", height: "30px", borderRadius: "9999px",
              border: "1.5px solid rgba(148,163,184,0.45)",
              display: "flex", alignItems: "flex-start", justifyContent: "center", padding: "4px"
            }}>
              <span className="animate-bounce" style={{
                width: "4px", height: "7px", borderRadius: "9999px",
                background: "#22d3ee", display: "block"
              }} />
            </div>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace", fontSize: "9px",
              fontWeight: 600, letterSpacing: "0.22em",
              color: "#94a3b8", textTransform: "uppercase"
            }}>SCROLL DOWN</span>
            <span style={{ color: "#64748b", fontSize: "12px", lineHeight: 1 }}>⌄</span>
          </div>

          {/* LAYER 7b — BOTTOM-RIGHT QUOTE */}
          <div style={{
            position: "absolute", bottom: "28px", right: "40px",
            zIndex: 20, pointerEvents: "none"
          }}>
            <div style={{ borderLeft: "2px solid rgba(100,116,139,0.55)", paddingLeft: "14px" }}>
              <p style={{
                fontStyle: "italic", color: "rgba(203,213,225,0.75)",
                lineHeight: 1.55, fontSize: "12px", fontFamily: "Georgia, serif", margin: 0
              }}>
                &#8220;Data from Earth.<br />
                Insights for a Sustainable Tomorrow.&#8221;
              </p>
            </div>
          </div>

        </section>
      )}
    </div>
  );
}
