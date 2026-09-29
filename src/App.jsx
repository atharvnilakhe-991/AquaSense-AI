import React, { useState, useEffect, useRef } from "react";
import { MOCK_WELLS, STUDY_AREA_INFO } from "./data/wellData.js";
import { GROUNDWATER_FACTS, KPI_CARDS_DATA, GROUNDWATER_TREND_SERIES, MOCK_EXPLORER_OBSERVATIONS } from "./data/groundwaterData.js";
import { RISK_DISTRIBUTION_DATA, GROUNDWATER_ALERTS } from "./data/riskData.js";
import { RECHARGE_CATEGORIES, RECHARGE_FACTORS } from "./data/rechargeData.js";
import { AMDFE_MODALITIES, AMDFE_SYSTEM_STATS } from "./data/amdfData.js";
import { ML_MODELS_DATA, AI_DECISION_SUPPORT_ITEMS, QUICK_ACCESS_MODULES, FAQ_DOCUMENTATION_ITEMS } from "./data/modelData.js";

export default function App() {
  const [currentRoute, setCurrentRoute] = useState("/");
  const [isAuthenticated, setIsAuthenticated] = useState(() => localStorage.getItem("aquasense_auth") === "true");
  const [loginModalOpen, setLoginModalOpen] = useState(false);
  const [selectedWell, setSelectedWell] = useState(MOCK_WELLS[3]);
  const [activeRiskFilter, setActiveRiskFilter] = useState("all");
  const [predTimeframe, setPredTimeframe] = useState("all");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifModalOpen, setNotifModalOpen] = useState(false);

  const protectedRoutes = ["/dashboard", "/groundwater-map", "/prediction-analysis", "/recharge-analysis", "/risk-assessment"];

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
      if (protectedRoutes.includes(hash) && !isAuthenticated) {
        setLoginModalOpen(true);
        setCurrentRoute("/");
        window.location.hash = "#/";
      } else {
        setCurrentRoute(hash);
      }
    };
    handleHash();
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, [isAuthenticated]);

  const navigate = (route) => {
    if (protectedRoutes.includes(route) && !isAuthenticated) {
      setLoginModalOpen(true);
      setCurrentRoute("/");
      window.location.hash = "#/";
      return;
    }
    setCurrentRoute(route);
    window.location.hash = route === "/" ? "#/" : `#${route}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
    localStorage.setItem("aquasense_auth", "true");
    setLoginModalOpen(false);
    navigate("/dashboard");
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem("aquasense_auth");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 antialiased font-sans flex flex-col justify-between selection:bg-sky-500/20 selection:text-sky-900">

      {currentRoute !== "/" && (
        <header
          style={{
            position: "sticky",
            top: 0,
            zIndex: 50,
            width: "100%",
            backgroundColor: "#ffffff",
            borderBottom: "1px solid #e2e8f0",
            boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.04)"
          }}
        >
          <div style={{ maxWidth: "1536px", margin: "0 auto", padding: "0 24px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "80px", gap: "24px" }}>

              {/* Left: Branding */}
              <div
                style={{ display: "flex", alignItems: "center", gap: "12px", cursor: "pointer", flexShrink: 0 }}
                onClick={() => navigate("/")}
              >
                <div
                  style={{
                    width: "40px", height: "40px", borderRadius: "50%",
                    background: "#f0f9ff", border: "1px solid #7dd3fc",
                    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    padding: "8px", flexShrink: 0
                  }}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0284c7">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8"
                      d="M12 2.5C8.5 7.5 6 11 6 15a6 6 0 0012 0c0-4-2.5-7.5-6-12.5z"
                      fill="#0284c7" fillOpacity="0.15" />
                    <circle cx="12" cy="14" r="2.5" fill="#0284c7" />
                  </svg>
                </div>
                <div>
                  <div style={{ display: "flex", alignItems: "baseline", lineHeight: 1 }}>
                    <span style={{ fontSize: "22px", fontWeight: 900, letterSpacing: "-0.02em", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>AquaSense</span>
                    <span style={{ fontSize: "22px", fontWeight: 900, letterSpacing: "-0.02em", color: "#0284c7", fontFamily: "'Inter', sans-serif" }}>AI</span>
                  </div>
                  <p style={{ fontSize: "11px", fontWeight: 500, color: "#64748b", letterSpacing: "0.10em", margin: "3px 0 0 0", lineHeight: 1, fontFamily: "'Inter', sans-serif" }}>
                    Groundwater Intelligence
                  </p>
                </div>
              </div>

              {/* Center: Single-Line Nav Links */}
              <div style={{ display: "flex", alignItems: "center", gap: "24px", whiteSpace: "nowrap", flex: 1, justifyContent: "center" }}>
                <button
                  onClick={() => navigate("/")}
                  style={{
                    fontSize: "14px", fontWeight: currentRoute === "/" ? 700 : 500,
                    color: currentRoute === "/" ? "#0284c7" : "#475569",
                    background: "none", border: "none",
                    borderBottom: currentRoute === "/" ? "2.5px solid #0284c7" : "none",
                    cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                    whiteSpace: "nowrap", fontFamily: "'Inter', sans-serif"
                  }}
                >
                  Home
                </button>
                {[
                  { label: "Dashboard",           route: "/dashboard" },
                  { label: "Groundwater Map",     route: "/groundwater-map" },
                  { label: "Prediction Analysis", route: "/prediction-analysis" },
                  { label: "Recharge Analysis",   route: "/recharge-analysis" },
                  { label: "Risk Assessment",     route: "/risk-assessment" }
                ].map(({ label, route }) => (
                  <button
                    key={label}
                    onClick={() => navigate(route)}
                    style={{
                      fontSize: "14px", fontWeight: currentRoute === route ? 700 : 500,
                      color: currentRoute === route ? "#0284c7" : "#475569",
                      background: "none", border: "none",
                      borderBottom: currentRoute === route ? "2.5px solid #0284c7" : "none",
                      cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                      whiteSpace: "nowrap", fontFamily: "'Inter', sans-serif", transition: "color 0.2s"
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = "#0284c7"}
                    onMouseLeave={e => e.currentTarget.style.color = currentRoute === route ? "#0284c7" : "#475569"}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {/* Right: Live Monitoring + Analyst + Logout */}
              <div style={{ display: "flex", alignItems: "center", gap: "18px", flexShrink: 0 }}>
                {/* Live Monitoring */}
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <div
                    style={{
                      width: "12px", height: "12px", borderRadius: "50%",
                      border: "2px solid #10b981",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      boxShadow: "0 0 5px rgba(16,185,129,0.4)", flexShrink: 0
                    }}
                  >
                    <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: "#10b981", display: "block" }} />
                  </div>
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <span style={{
                      fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
                      fontWeight: 700, letterSpacing: "0.16em", color: "#059669",
                      lineHeight: 1, textTransform: "uppercase"
                    }}>LIVE MONITORING</span>
                    <span style={{ fontSize: "11px", color: "#64748b", marginTop: "3px", lineHeight: 1, fontFamily: "'Inter', sans-serif", fontWeight: 500 }}>
                      South-Central Nebraska
                    </span>
                  </div>
                </div>

                {/* Analyst Badge */}
                {isAuthenticated && (
                  <div style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "6px 12px", borderRadius: "9999px",
                    background: "#f1f5f9", border: "1px solid #e2e8f0",
                    fontSize: "12px", color: "#334155"
                  }}>
                    <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981" }} />
                    <span style={{ fontFamily: "monospace", fontSize: "11px", fontWeight: 600 }}>Analyst</span>
                  </div>
                )}

                {/* Logout Button */}
                {isAuthenticated ? (
                  <button
                    onClick={handleLogout}
                    style={{
                      padding: "8px 16px", borderRadius: "10px",
                      fontWeight: 700, fontSize: "12px", letterSpacing: "0.02em",
                      color: "#334155", background: "#f1f5f9",
                      display: "flex", alignItems: "center", gap: "6px",
                      cursor: "pointer", border: "1px solid #e2e8f0",
                      fontFamily: "'Inter', sans-serif", flexShrink: 0
                    }}
                  >
                    <span>Logout</span>
                    <span style={{ fontSize: "13px" }}>⎋</span>
                  </button>
                ) : (
                  <button
                    onClick={() => setLoginModalOpen(true)}
                    className="hero-btn-primary"
                    style={{
                      padding: "9px 20px", borderRadius: "10px",
                      fontWeight: 700, fontSize: "13px", color: "#ffffff",
                      cursor: "pointer", border: "none"
                    }}
                  >
                    Login →
                  </button>
                )}
              </div>

            </div>
          </div>
        </header>
      )}

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

            {/* Homepage Nav Links: Positioned closer to AquaSenseAI branding with a balanced professional gap */}
            <div style={{ display: "flex", alignItems: "center", gap: "24px", flex: 1, justifyContent: "flex-start", marginLeft: "40px" }}>
              <button
                onClick={() => navigate("/")}
                style={{
                  fontSize: "14px", fontWeight: 600, color: "#00e5ff",
                  background: "none", border: "none", borderBottom: currentRoute === "/" ? "2px solid #00e5ff" : "none",
                  cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                  fontFamily: "'Inter', sans-serif"
                }}
              >
                Home
              </button>
              {isAuthenticated && [
                { label: "Dashboard",           route: "/dashboard" },
                { label: "Groundwater Map",     route: "/groundwater-map" },
                { label: "Prediction Analysis", route: "/prediction-analysis" },
                { label: "Recharge Analysis",   route: "/recharge-analysis" },
                { label: "Risk Assessment",     route: "/risk-assessment" }
              ].map(({ label, route }) => (
                <button
                  key={label}
                  onClick={() => navigate(route)}
                  style={{
                    fontSize: "14px", fontWeight: currentRoute === route ? 600 : 400,
                    color: currentRoute === route ? "#00e5ff" : "#cbd5e1",
                    background: "none", border: "none",
                    borderBottom: currentRoute === route ? "2px solid #00e5ff" : "none",
                    cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                    fontFamily: "'Inter', sans-serif", transition: "color 0.2s"
                  }}
                  onMouseEnter={e => e.currentTarget.style.color = "#ffffff"}
                  onMouseLeave={e => e.currentTarget.style.color = currentRoute === route ? "#00e5ff" : "#cbd5e1"}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Right: Live Monitoring + Login/Logout CTA */}
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

              {isAuthenticated ? (
                <button
                  onClick={handleLogout}
                  className="hero-btn-secondary"
                  style={{
                    padding: "8px 20px", borderRadius: "9999px",
                    fontWeight: 600, fontSize: "13px", letterSpacing: "0.02em",
                    color: "#ffffff", display: "flex", alignItems: "center",
                    gap: "6px", cursor: "pointer", border: "1px solid rgba(255,255,255,0.25)",
                    fontFamily: "'Inter', sans-serif", flexShrink: 0
                  }}
                >
                  <span>Logout</span>
                  <span style={{ fontSize: "14px" }}>⎋</span>
                </button>
              ) : (
                <button
                  onClick={() => setLoginModalOpen(true)}
                  className="hero-btn-primary"
                  style={{
                    padding: "10px 24px", borderRadius: "9999px",
                    fontWeight: 700, fontSize: "14px", letterSpacing: "0.02em",
                    color: "#ffffff", display: "flex", alignItems: "center",
                    gap: "8px", cursor: "pointer", border: "none",
                    fontFamily: "'Inter', sans-serif", flexShrink: 0
                  }}
                >
                  <span>Login</span>
                  <span style={{ fontSize: "15px" }}>→</span>
                </button>
              )}
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

          {/* LAYER 8 — HERO BOTTOM FADE (PREMIUM BLUE + AQUA + WHITE) */}
          <div
            aria-hidden="true"
            className="hero-bottom-fade"
            style={{
              position: "absolute",
              bottom: "-1px",
              left: 0,
              right: 0,
              height: "140px",
              pointerEvents: "none",
              zIndex: 15,
              userSelect: "none",
              background: "radial-gradient(ellipse 85% 65% at 50% 68%, rgba(56, 189, 248, 0.16) 0%, rgba(14, 116, 180, 0.08) 50%, transparent 80%), linear-gradient(to bottom, rgba(3, 7, 18, 0) 0%, rgba(4, 18, 42, 0.35) 18%, rgba(8, 36, 76, 0.65) 34%, rgba(12, 56, 112, 0.75) 48%, rgba(16, 88, 154, 0.72) 60%, rgba(34, 138, 202, 0.65) 72%, rgba(110, 198, 242, 0.68) 82%, rgba(195, 230, 250, 0.85) 90%, rgba(235, 245, 252, 0.96) 96%, #f8fafc 100%)"
            }}
          />

        </section>
      )}

      {/* LOGIN MODAL */}
      {loginModalOpen && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 100,
          background: "rgba(3, 7, 18, 0.85)", backdropFilter: "blur(12px)",
          display: "flex", alignItems: "center", justifyContent: "center", padding: "16px"
        }}>
          <div style={{
            background: "#0f172a", border: "1px solid rgba(0, 229, 255, 0.3)",
            borderRadius: "20px", maxWidth: "420px", width: "100%", padding: "28px",
            color: "#ffffff", boxShadow: "0 0 50px rgba(0, 229, 255, 0.15)", position: "relative"
          }}>
            <button
              onClick={() => setLoginModalOpen(false)}
              style={{
                position: "absolute", top: "16px", right: "16px",
                background: "transparent", border: "none", color: "#94a3b8",
                fontSize: "18px", cursor: "pointer"
              }}
            >
              ✕
            </button>

            <div style={{ marginBottom: "20px" }}>
              <span style={{
                display: "inline-block", padding: "3px 10px", borderRadius: "9999px",
                background: "rgba(0, 229, 255, 0.1)", border: "1px solid rgba(0, 229, 255, 0.3)",
                color: "#00e5ff", fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "8px"
              }}>
                SECURE ACCESS PORTAL
              </span>
              <h2 style={{ fontSize: "22px", fontWeight: 800, margin: "4px 0", color: "#ffffff" }}>
                Sign In to AquaSense<span style={{ color: "#00e5ff" }}>AI</span>
              </h2>
              <p style={{ fontSize: "12px", color: "#94a3b8", margin: 0, lineHeight: 1.5 }}>
                Enter your credentials or click instant demo login to access models and telemetry.
              </p>
            </div>

            <div style={{
              background: "rgba(30, 41, 59, 0.7)", border: "1px solid rgba(51, 65, 85, 0.6)",
              borderRadius: "12px", padding: "12px", marginBottom: "18px"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>
                <span>Demo Account:</span>
                <span style={{ color: "#00e5ff", fontFamily: "monospace" }}>analyst@aquasense.ai</span>
              </div>
              <button
                onClick={handleLogin}
                type="button"
                style={{
                  width: "100%", padding: "8px", borderRadius: "8px",
                  background: "rgba(0, 229, 255, 0.15)", border: "1px solid rgba(0, 229, 255, 0.4)",
                  color: "#38bdf8", fontWeight: 700, fontSize: "12px", cursor: "pointer"
                }}
              >
                ⚡ Instant 1-Click Demo Login
              </button>
            </div>

            <form onSubmit={(e) => { e.preventDefault(); handleLogin(); }} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <div>
                <label style={{ display: "block", fontSize: "11px", color: "#cbd5e1", marginBottom: "4px" }}>Email</label>
                <input
                  type="email" defaultValue="analyst@aquasense.ai" required
                  style={{
                    width: "100%", padding: "10px 12px", borderRadius: "10px",
                    background: "#030712", border: "1px solid #334155", color: "#ffffff",
                    fontSize: "13px", outline: "none", boxSizing: "border-box"
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "11px", color: "#cbd5e1", marginBottom: "4px" }}>Password</label>
                <input
                  type="password" defaultValue="aquasense2026" required
                  style={{
                    width: "100%", padding: "10px 12px", borderRadius: "10px",
                    background: "#030712", border: "1px solid #334155", color: "#ffffff",
                    fontSize: "13px", outline: "none", boxSizing: "border-box"
                  }}
                />
              </div>
              <button
                type="submit"
                className="hero-btn-primary"
                style={{
                  width: "100%", padding: "12px", borderRadius: "12px",
                  fontWeight: 700, fontSize: "14px", color: "#ffffff",
                  cursor: "pointer", border: "none", marginTop: "6px"
                }}
              >
                Sign In & Unlock Platform →
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
