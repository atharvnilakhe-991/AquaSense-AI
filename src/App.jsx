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
  const [overviewModalOpen, setOverviewModalOpen] = useState(false);
  const [pendingRoute, setPendingRoute] = useState(null);
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
        setPendingRoute(hash);
        setLoginModalOpen(true);
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
      setPendingRoute(route);
      setLoginModalOpen(true);
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
    const next = pendingRoute || "/dashboard";
    setPendingRoute(null);
    setCurrentRoute(next);
    window.location.hash = next === "/" ? "#/" : `#${next}`;
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem("aquasense_auth");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-[#080D18] text-slate-200 antialiased font-sans flex flex-col justify-between selection:bg-cyan-500/20 selection:text-cyan-200">

      {/* SINGLE UNIFIED MASTER NAVBAR - ALWAYS VISIBLE */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          width: "100%",
          backgroundColor: "rgba(8, 13, 24, 0.94)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.35)"
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

            {/* Center: Complete Single-Line Navigation Links (ALWAYS VISIBLE) */}
            <div style={{ display: "flex", alignItems: "center", gap: "24px", whiteSpace: "nowrap", flex: 1, justifyContent: "flex-start", marginLeft: "36px" }}>
              {[
                { label: "Home",                route: "/" },
                { label: "Dashboard",           route: "/dashboard" },
                { label: "Groundwater Map",     route: "/groundwater-map" },
                { label: "Prediction Analysis", route: "/prediction-analysis" },
                { label: "Recharge Analysis",   route: "/recharge-analysis" },
                { label: "Risk Assessment",     route: "/risk-assessment" }
              ].map(({ label, route }) => {
                const isActive = currentRoute === route;
                return (
                  <button
                    key={label}
                    onClick={() => navigate(route)}
                    style={{
                      fontSize: "14px", fontWeight: isActive ? 700 : 500,
                      color: isActive ? "#00e5ff" : "#cbd5e1",
                      background: "none", border: "none",
                      borderBottom: isActive ? "2.5px solid #00e5ff" : "none",
                      boxShadow: isActive ? "0 2px 10px rgba(0,229,255,0.4)" : "none",
                      cursor: "pointer", paddingBottom: "4px", lineHeight: 1,
                      whiteSpace: "nowrap", fontFamily: "'Inter', sans-serif", transition: "color 0.15s"
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = "#00e5ff"}
                    onMouseLeave={e => e.currentTarget.style.color = isActive ? "#00e5ff" : "#cbd5e1"}
                  >
                    {label}
                  </button>
                );
              })}
            </div>

            {/* Right: Live Monitoring + Analyst / Login Actions */}
            <div style={{ display: "flex", alignItems: "center", gap: "18px", flexShrink: 0 }}>
              {/* Live Monitoring */}
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

              {/* Analyst Badge & Auth Controls */}
              {isAuthenticated ? (
                <>
                  <div style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "6px 12px", borderRadius: "9999px",
                    background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(51, 65, 85, 0.8)",
                    fontSize: "12px", color: "#38bdf8"
                  }}>
                    <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981" }} />
                    <span style={{ fontFamily: "monospace", fontSize: "11px", fontWeight: 600 }}>Analyst</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="hero-btn-secondary"
                    style={{
                      padding: "8px 16px", borderRadius: "9999px",
                      fontWeight: 600, fontSize: "12px", letterSpacing: "0.02em",
                      color: "#ffffff", display: "flex", alignItems: "center",
                      gap: "6px", cursor: "pointer", border: "1px solid rgba(255,255,255,0.2)",
                      fontFamily: "'Inter', sans-serif", flexShrink: 0
                    }}
                  >
                    <span>Logout</span>
                    <span style={{ fontSize: "13px" }}>⎋</span>
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setLoginModalOpen(true)}
                  className="hero-btn-primary"
                  style={{
                    padding: "9px 22px", borderRadius: "9999px",
                    fontWeight: 700, fontSize: "13px", color: "#ffffff",
                    cursor: "pointer", border: "none", display: "flex", alignItems: "center", gap: "6px"
                  }}
                >
                  <span>Login</span>
                  <span style={{ fontSize: "14px" }}>→</span>
                </button>
              )}
            </div>

          </div>
        </div>
      </header>

      {currentRoute === "/" && (
        <section
          style={{
            position: "relative",
            width: "100vw",
            height: "calc(100vh - 80px)",
            minHeight: "640px",
            overflow: "hidden",
            backgroundColor: "#030712",
            color: "#ffffff",
            fontFamily: "'Inter', sans-serif",
            userSelect: "none"
          }}
        >
          {/* Static Hero Satellite & Earth Background */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 0,
              backgroundImage: "url('/hero_bg.jpg')",
              backgroundSize: "cover",
              backgroundPosition: "center center",
              backgroundRepeat: "no-repeat",
              pointerEvents: "none"
            }}
          />

          {/* Left Readability Overlay */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 1,
              background: "linear-gradient(to right, rgba(3,7,18,0.85) 0%, rgba(3,7,18,0.65) 25%, rgba(3,7,18,0.22) 48%, transparent 65%)",
              pointerEvents: "none"
            }}
          />

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
                onClick={() => setOverviewModalOpen(true)}
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

          {/* LAYER 8 — HERO BOTTOM FADE (DARK NAVY TRANSITION) */}
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
              background: "radial-gradient(ellipse 85% 65% at 50% 70%, rgba(0, 229, 255, 0.08) 0%, rgba(14, 116, 180, 0.04) 50%, transparent 80%), linear-gradient(to bottom, rgba(3, 7, 18, 0) 0%, rgba(5, 10, 22, 0.35) 25%, rgba(8, 13, 24, 0.75) 60%, #080D18 100%)"
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
            background: "#111A2A", border: "1px solid rgba(0, 229, 255, 0.3)",
            borderRadius: "20px", maxWidth: "420px", width: "100%", padding: "28px",
            color: "#ffffff", boxShadow: "0 0 50px rgba(0, 229, 255, 0.15)", position: "relative"
          }}>
            <button
              onClick={() => { setLoginModalOpen(false); setPendingRoute(null); }}
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
                Sign in to access AquaSense AI groundwater intelligence.
              </p>
            </div>

            {pendingRoute && (
              <div style={{
                background: "rgba(0, 229, 255, 0.08)", border: "1px solid rgba(0, 229, 255, 0.3)",
                borderRadius: "10px", padding: "10px 14px", marginBottom: "16px",
                fontSize: "12px", fontFamily: "'JetBrains Mono', monospace", color: "#38bdf8",
                display: "flex", alignItems: "center", gap: "8px"
              }}>
                <span style={{ color: "#94a3b8" }}>Target Module:</span>
                <span style={{ color: "#ffffff", fontWeight: 700 }}>
                  {pendingRoute === "/dashboard" ? "Dashboard" :
                   pendingRoute === "/groundwater-map" ? "Groundwater Map" :
                   pendingRoute === "/prediction-analysis" ? "Prediction Analysis" :
                   pendingRoute === "/recharge-analysis" ? "Recharge Analysis" :
                   pendingRoute === "/risk-assessment" ? "Risk Assessment" : pendingRoute}
                </span>
              </div>
            )}

            <div style={{
              background: "#151F31", border: "1px solid rgba(51, 65, 85, 0.6)",
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
                    background: "#0D1422", border: "1px solid #334155", color: "#ffffff",
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
                    background: "#0D1422", border: "1px solid #334155", color: "#ffffff",
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

      {/* OVERVIEW MODAL */}
      {overviewModalOpen && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 100,
          background: "rgba(3, 7, 18, 0.85)", backdropFilter: "blur(12px)",
          display: "flex", alignItems: "center", justifyContent: "center", padding: "16px"
        }}>
          <div style={{
            background: "#111A2A", border: "1px solid rgba(0, 229, 255, 0.3)",
            borderRadius: "20px", maxWidth: "600px", width: "100%", padding: "28px",
            color: "#ffffff", boxShadow: "0 0 50px rgba(0, 229, 255, 0.15)", position: "relative"
          }}>
            <button
              onClick={() => setOverviewModalOpen(false)}
              style={{
                position: "absolute", top: "16px", right: "16px",
                background: "transparent", border: "none", color: "#94a3b8",
                fontSize: "18px", cursor: "pointer"
              }}
            >
              ✕
            </button>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "16px", marginBottom: "16px" }}>
              <div style={{ width: "36px", height: "36px", borderRadius: "50%", background: "rgba(0,229,255,0.15)", border: "1px solid rgba(0,229,255,0.4)", display: "flex", alignItems: "center", justifyContent: "center", color: "#00e5ff" }}>
                ▷
              </div>
              <div>
                <h3 style={{ fontSize: "18px", fontWeight: 700, margin: 0, color: "#ffffff" }}>AquaSense AI Platform Overview</h3>
                <p style={{ fontSize: "12px", color: "#94a3b8", margin: "2px 0 0 0" }}>Groundwater Intelligence & Orbital Earth Observation</p>
              </div>
            </div>
            <div style={{ fontSize: "13px", color: "#cbd5e1", lineHeight: 1.6, marginBottom: "20px" }}>
              AquaSense AI harmonizes satellite observations, environmental variables, and 24 years of piezometric soundings to forecast groundwater depletion and recharge dynamics with rigorous scientific certainty.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "24px" }}>
              <div style={{ background: "#151F31", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ fontSize: "10px", fontFamily: "monospace", color: "#38bdf8", fontWeight: 700, display: "block", marginBottom: "4px" }}>01 / SENSING INGESTION</span>
                <span style={{ fontSize: "12px", color: "#ffffff" }}>NASA POWER, Sentinel-2 Multispectral & SRTM 30m DEM</span>
              </div>
              <div style={{ background: "#151F31", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ fontSize: "10px", fontFamily: "monospace", color: "#38bdf8", fontWeight: 700, display: "block", marginBottom: "4px" }}>02 / AMDFE FUSION</span>
                <span style={{ fontSize: "12px", color: "#ffffff" }}>Dynamic Signal-to-Noise Ratio (SNR) Reliability Weighting</span>
              </div>
              <div style={{ background: "#151F31", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ fontSize: "10px", fontFamily: "monospace", color: "#38bdf8", fontWeight: 700, display: "block", marginBottom: "4px" }}>03 / QUANTILE ENSEMBLE</span>
                <span style={{ fontSize: "12px", color: "#ffffff" }}>LightGBM & XGBoost with 90% Prediction Uncertainty Bands</span>
              </div>
              <div style={{ background: "#151F31", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)" }}>
                <span style={{ fontSize: "10px", fontFamily: "monospace", color: "#38bdf8", fontWeight: 700, display: "block", marginBottom: "4px" }}>04 / DECISION SUPPORT</span>
                <span style={{ fontSize: "12px", color: "#ffffff" }}>TreeSHAP Game-Theoretic Feature Attributions & GIS Maps</span>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "16px" }}>
              <button
                onClick={() => setOverviewModalOpen(false)}
                className="hero-btn-secondary"
                style={{ padding: "8px 16px", borderRadius: "8px", fontSize: "12px", cursor: "pointer" }}
              >
                Close
              </button>
              <button
                onClick={() => { setOverviewModalOpen(false); navigate("/dashboard"); }}
                className="hero-btn-primary"
                style={{ padding: "8px 20px", borderRadius: "8px", fontSize: "12px", cursor: "pointer", border: "none" }}
              >
                Explore Dashboard →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
