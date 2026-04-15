import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from "recharts";

const TRAINING_CURVE = [
  { epoch: 1,  train: 0.42, val: 0.38 },
  { epoch: 3,  train: 0.61, val: 0.57 },
  { epoch: 5,  train: 0.72, val: 0.68 },
  { epoch: 8,  train: 0.80, val: 0.76 },
  { epoch: 10, train: 0.85, val: 0.81 },
  { epoch: 13, train: 0.88, val: 0.84 },
  { epoch: 15, train: 0.90, val: 0.86 },
  { epoch: 18, train: 0.91, val: 0.88 },
  { epoch: 20, train: 0.926, val: 0.891 },
];

const CLASSWISE_F1 = [
  { name: "Tomato Blight",   score: 0.94 },
  { name: "Potato Healthy",  score: 0.97 },
  { name: "Apple Scab",      score: 0.91 },
  { name: "Grape Black Rot", score: 0.89 },
  { name: "Corn Rust",       score: 0.93 },
  { name: "Pepper Healthy",  score: 0.96 },
].sort((a, b) => b.score - a.score);

const XAI_RADAR = [
  { metric: "Faithfulness",  GradCAM: 82, LIME: 71, SHAP: 76 },
  { metric: "Robustness",    GradCAM: 88, LIME: 64, SHAP: 79 },
  { metric: "Localization",  GradCAM: 91, LIME: 69, SHAP: 73 },
  { metric: "Speed",         GradCAM: 99, LIME: 28, SHAP: 42 },
  { metric: "Consistency",   GradCAM: 85, LIME: 72, SHAP: 80 },
];

const FAITHFULNESS_CURVE = [
  { pct: "0%",   GradCAM: 0.89, LIME: 0.89, SHAP: 0.89 },
  { pct: "5%",   GradCAM: 0.72, LIME: 0.80, SHAP: 0.76 },
  { pct: "10%",  GradCAM: 0.54, LIME: 0.71, SHAP: 0.63 },
  { pct: "20%",  GradCAM: 0.38, LIME: 0.59, SHAP: 0.48 },
  { pct: "30%",  GradCAM: 0.24, LIME: 0.48, SHAP: 0.35 },
  { pct: "50%",  GradCAM: 0.12, LIME: 0.31, SHAP: 0.22 },
];

const TOOLTIP_STYLE = {
  backgroundColor: "#111318",
  border: "1px solid #1e2530",
  borderRadius: 8,
  fontSize: 12,
  color: "#e8ecf2",
};

export default function Dashboard() {
  return (
    <div>
      <div className="page-title">Research Dashboard</div>
      <div className="page-subtitle">
        Quantitative analysis of XAI methods for plant disease classification
      </div>

      {/* ── Stat tiles ─────────────────────────────── */}
      <div className="stat-grid">
        {[
          { label: "Model Accuracy",    value: "96.93%", color: "green" },
          { label: "F1 Score (macro)",  value: "0.970",  color: "green" },
          { label: "Classes",           value: "16",     color: "blue"  },
          { label: "Best XAI Method",   value: "Grad-CAM", color: "amber" },
          { label: "Faithfulness Gap",  value: "11.2%",  color: "coral" },
          { label: "SHAP SSIM",         value: "0.79",   color: "blue"  },
          
        ].map(s => (
          <div className="stat-tile" key={s.label}>
            <div className="stat-label">{s.label}</div>
            <div className={`stat-value ${s.color}`}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* ── Charts row 1 ───────────────────────────── */}
      <div className="grid-2 mb-4">
        {/* Training curve */}
        <div className="card">
          <div className="section-title">Training & Validation Accuracy</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={TRAINING_CURVE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="gTrain" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#22d48a" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22d48a" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="gVal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#4d9ef5" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#4d9ef5" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2530" />
              <XAxis dataKey="epoch" tick={{ fill: "#4d5668", fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#4d5668", fontSize: 11 }} />
              <YAxis domain={[0.3, 1]} tick={{ fill: "#4d5668", fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => `${(v*100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#8b95a6" }} />
              <Area type="monotone" dataKey="train" name="Train" stroke="#22d48a" fill="url(#gTrain)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="val"   name="Val"   stroke="#4d9ef5" fill="url(#gVal)"   strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* XAI Radar */}
        <div className="card">
          <div className="section-title">XAI Method Comparison (Radar)</div>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={XAI_RADAR}>
              <PolarGrid stroke="#1e2530" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#8b95a6", fontSize: 11 }} />
              <Radar name="Grad-CAM" dataKey="GradCAM" stroke="#f0a832" fill="#f0a832" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="LIME"     dataKey="LIME"    stroke="#4d9ef5" fill="#4d9ef5" fillOpacity={0.10} strokeWidth={2} />
              <Radar name="SHAP"     dataKey="SHAP"    stroke="#f06060" fill="#f06060" fillOpacity={0.10} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#8b95a6" }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Charts row 2 ───────────────────────────── */}
      <div className="grid-2 mb-4">
        {/* Faithfulness curve */}
        <div className="card">
          <div className="section-title">Faithfulness — Deletion Test</div>
          <div className="text-secondary text-sm mb-3">
            Confidence drop as top-K% pixels removed. Steeper = more faithful.
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={FAITHFULNESS_CURVE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2530" />
              <XAxis dataKey="pct" tick={{ fill: "#4d5668", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#4d5668", fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => `${(v*100).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#8b95a6" }} />
              <Area type="monotone" dataKey="GradCAM" stroke="#f0a832" fill="none" strokeWidth={2.5} dot={false} />
              <Area type="monotone" dataKey="LIME"    stroke="#4d9ef5" fill="none" strokeWidth={2} dot={false} strokeDasharray="4 2" />
              <Area type="monotone" dataKey="SHAP"    stroke="#f06060" fill="none" strokeWidth={2} dot={false} strokeDasharray="2 2" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Per-class F1 */}
        <div className="card">
          <div className="section-title">Per-class F1 Score (sample)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={CLASSWISE_F1} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2530" horizontal={false} />
              <XAxis type="number" domain={[0.8, 1.0]} tick={{ fill: "#4d5668", fontSize: 10 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#8b95a6", fontSize: 11 }} width={110} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => `${(v*100).toFixed(1)}%`} />
              <Bar dataKey="score" fill="#22d48a" radius={[0, 4, 4, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Method comparison table ─────────────────── */}
      <div className="card">
        <div className="section-title">XAI Method Summary Table</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Method", "Type", "Faithfulness ↓", "Robustness (SSIM)", "Time", "Pros", "Cons"].map(h => (
                <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--text-muted)", fontWeight: 500, fontSize: 11 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              {
                method: "Grad-CAM", type: "Gradient", faith: "0.31", ssim: "0.88",
                time: "~0.05s", color: "badge-gradcam",
                pros: "Fast, spatial", cons: "Low resolution",
              },
              {
                method: "LIME", type: "Perturbation", faith: "0.42", ssim: "0.64",
                time: "~8s", color: "badge-lime",
                pros: "Model-agnostic", cons: "Slow, noisy",
              },
              {
                method: "SHAP", type: "Game theory", faith: "0.38", ssim: "0.79",
                time: "~12s", color: "badge-shap",
                pros: "Signed contrib.", cons: "Slow, memory",
              },
            ].map((r, i) => (
              <tr key={r.method} style={{ borderBottom: "1px solid var(--border)", background: i % 2 ? "var(--bg-elevated)" : "transparent" }}>
                <td style={{ padding: "10px 12px" }}><span className={`badge ${r.color}`}>{r.method}</span></td>
                <td style={{ padding: "10px 12px", color: "var(--text-secondary)" }}>{r.type}</td>
                <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", color: "var(--green)" }}>{r.faith}</td>
                <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", color: "var(--blue)" }}>{r.ssim}</td>
                <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", color: "var(--amber)" }}>{r.time}</td>
                <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontSize: 12 }}>{r.pros}</td>
                <td style={{ padding: "10px 12px", color: "var(--text-muted)", fontSize: 12 }}>{r.cons}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
