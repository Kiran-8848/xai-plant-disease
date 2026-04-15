import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Analyze from "./pages/Analyze";
import Compare from "./pages/Compare";
import About from "./pages/About";
import Sidebar from "./components/Sidebar";
import "./index.css";

export default function App() {
  const [page, setPage] = useState("dashboard");

  const pages = {
    dashboard: <Dashboard />,
    analyze:   <Analyze />,
    compare:   <Compare />,
    about:     <About />,
  };

  return (
    <div className="app-shell">
      <Sidebar currentPage={page} onNavigate={setPage} />
      <main className="main-content">
        {pages[page]}
      </main>
    </div>
  );
}
