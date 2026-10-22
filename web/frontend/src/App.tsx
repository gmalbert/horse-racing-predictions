import { useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import InternationalPage from "./pages/InternationalPage";
import UsRacingPage from "./pages/UsRacingPage";
import DataExplorerPage from "./pages/DataExplorerPage";
import TrainerStatsPage from "./pages/TrainerStatsPage";
import JockeyStatsPage from "./pages/JockeyStatsPage";
import BettingManagerPage from "./pages/BettingManagerPage";
import BacktestResultsPage from "./pages/BacktestResultsPage";
const NAV = [
  { to: "/", label: "International", icon: "🏇", end: true },
  { to: "/us-racing", label: "US Racing", icon: "🇺🇸" },
  { to: "/data-explorer", label: "Data Explorer", icon: "📊" },
  { to: "/trainer-stats", label: "Trainer Stats", icon: "🏋️" },
  { to: "/jockey-stats", label: "Jockey Stats", icon: "👤" },
  { to: "/betting-manager", label: "Betting Manager", icon: "💰" },
  { to: "/backtest", label: "Backtest Results", icon: "📈" },
];

function Footer() {
  return (
    <div className="footer">
      <div>
        Powered by{" "}
        <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">
          Betting Oracle
        </a>
      </div>
      <div>Sports Prediction Analytics</div>
      <img
        src="https://raw.githubusercontent.com/gmalbert/betting-oracle/main/data_files/logo.png"
        alt="Betting Oracle"
        loading="lazy"
      />
    </div>
  );
}

export default function App() {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <>
      <div className="mobile-topbar">
        <button className="btn secondary" onClick={() => setNavOpen((o) => !o)} aria-label="Menu" aria-expanded={navOpen} aria-controls="main-navigation">
          ☰
        </button>
        <span className="brand">🏇 Equine Edge</span>
      </div>
      <div className="layout">
        <nav id="main-navigation" aria-label="Main navigation" className={`sidebar ${navOpen ? "open" : ""}`}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end as boolean | undefined}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
              onClick={() => setNavOpen(false)}
            >
              <span className="icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
          <div className="spacer" />
          <div className="side-logo">
            <img src={`${import.meta.env.BASE_URL}logo.png`} alt="" />
          </div>
        </nav>
        <main className="main">
          <Routes>
            <Route path="/" element={<InternationalPage />} />
            <Route path="/us-racing" element={<UsRacingPage />} />
            <Route path="/data-explorer" element={<DataExplorerPage />} />
            <Route path="/trainer-stats" element={<TrainerStatsPage />} />
            <Route path="/jockey-stats" element={<JockeyStatsPage />} />
            <Route path="/betting-manager" element={<BettingManagerPage />} />
            <Route path="/backtest" element={<BacktestResultsPage />} />
            <Route path="*" element={<div className="page-header"><h1>Page not found</h1><NavLink to="/">Return to International Racing</NavLink></div>} />
          </Routes>
          <Footer />
        </main>
      </div>
    </>
  );
}
