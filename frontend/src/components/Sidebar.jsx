import { LayoutDashboard, Microscope, GitCompare, BookOpen, Leaf } from "lucide-react";

const NAV = [
  { id: "dashboard", label: "Dashboard",     icon: LayoutDashboard },
  { id: "analyze",   label: "Analyze Image", icon: Microscope },
  { id: "compare",   label: "Compare XAI",   icon: GitCompare },
  { id: "about",     label: "Methodology",   icon: BookOpen },
];

export default function Sidebar({ currentPage, onNavigate }) {
  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <Leaf size={16} />
        XAI<span>Plant</span>
      </div>

      <div className="sidebar-section-label">Navigation</div>

      {NAV.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          className={`sidebar-nav-item ${currentPage === id ? "active" : ""}`}
          onClick={() => onNavigate(id)}
        >
          <Icon size={15} />
          {label}
        </button>
      ))}

      <div style={{ flex: 1 }} />

      <div style={{
        padding: "12px",
        borderRadius: "var(--radius-sm)",
        background: "var(--green-dim)",
        marginTop: "auto",
      }}>
        <div style={{ fontSize: 11, color: "var(--green)", fontWeight: 500, marginBottom: 4 }}>
          ResNet18 + XAI
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          Grad-CAM · LIME · SHAP
        </div>
      </div>
    </nav>
  );
}
