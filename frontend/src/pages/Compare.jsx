import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, GitCompare } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar,
} from "recharts";
import axios from "axios";

const API = "http://localhost:8000/api";

const METHOD_COLORS = {
  GradCAM: "#f0a832",
  LIME:    "#4d9ef5",
};
const METHOD_BADGE = {
  GradCAM: "badge-gradcam",
  LIME:    "badge-lime",
};

const TOOLTIP_STYLE = {
  backgroundColor: "#111318",
  border:          "1px solid #1e2530",
  borderRadius:    8,
  fontSize:        12,
  color:           "#e8ecf2",
};

export default function Compare() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);

  const onDrop = useCallback(([f]) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:   { "image/*": [] },
    maxFiles: 1,
  });

  const handleCompare = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const { data } = await axios.post(
        `${API}/explain/compare?run_faithfulness=true`, fd
      );
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const faithChartData = result
    ? (() => {
        const methods = Object.keys(result.comparison);
        const xPoints = result.comparison[methods[0]]
          ?.confidence_curve?.x || [];
        return xPoints.map((x, i) => {
          const row = { pct: `${Math.round(x * 100)}%` };
          methods.forEach(m => {
            row[m] = result.comparison[m]
              ?.confidence_curve?.y?.[i] ?? null;
          });
          return row;
        });
      })()
    : [];

  const summaryBarData = result
    ? Object.entries(result.comparison).map(([name, r]) => ({
        name,
        time: r.computation_time_s,
      }))
    : [];

  const formatClass = s =>
    (s || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div>
      <div className="page-title">Compare XAI Methods</div>
      <div className="page-subtitle">
        Run Grad-CAM and LIME on the same image and compare faithfulness.
      </div>

      {/* Upload row */}
      <div
        className="card mb-4"
        style={{ display: "flex", alignItems: "center", gap: 24 }}
      >
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "active" : ""}`}
          style={{ flex: 1, padding: 24, minHeight: "unset" }}
        >
          <input {...getInputProps()} />
          {preview ? (
            <img
              src={preview}
              alt="preview"
              style={{
                height:       80,
                borderRadius: "var(--radius-sm)",
                margin:       "0 auto",
                display:      "block",
              }}
            />
          ) : (
            <div style={{
              display:    "flex",
              alignItems: "center",
              gap:        12,
              color:      "var(--text-secondary)",
            }}>
              <Upload size={20} />
              <span>Drop leaf image here or click to upload</span>
            </div>
          )}
        </div>

        <button
          className="btn btn-primary"
          onClick={handleCompare}
          disabled={!file || loading}
          style={{ padding: "12px 24px", whiteSpace: "nowrap" }}
        >
          {loading ? (
            <>
              <div className="spinner" style={{ width: 16, height: 16 }} />
              Comparing methods...
            </>
          ) : (
            <><GitCompare size={15} /> Run Comparison</>
          )}
        </button>
      </div>

      {error && (
        <div style={{
          padding:      12,
          borderRadius: "var(--radius-sm)",
          background:   "var(--coral-dim)",
          color:        "var(--coral)",
          fontSize:     13,
          marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Prediction summary */}
          <div className="card-sm mb-4">
            <span
              className="badge badge-green"
              style={{ marginRight: 10 }}
            >
              Prediction
            </span>
            <span style={{ fontSize: 15, fontWeight: 600 }}>
              {formatClass(result.prediction?.pred_label)}
            </span>
            <span style={{
              marginLeft: 12,
              fontSize:   13,
              color:      "var(--text-secondary)",
            }}>
              Confidence:{" "}
              <span style={{
                color:      "var(--green)",
                fontFamily: "var(--font-mono)",
              }}>
                {(
                  (result.prediction?.confidence || 0) * 100
                ).toFixed(1)}%
              </span>
            </span>
          </div>

          {/* Side by side explanations */}
          <div className="grid-2 mb-4">
            {Object.entries(result.comparison).map(([method, r]) => (
              <div className="card" key={method}>
                <div className="flex items-center gap-2 mb-3">
                  <span className={`badge ${METHOD_BADGE[method]}`}>
                    {method}
                  </span>
                  <span className="text-sm text-muted">
                    {r.computation_time_s}s
                  </span>
                </div>
                <img
                  src={`data:image/png;base64,${r.explanation_b64}`}
                  alt={method}
                  className="explanation-image mb-3"
                />
                <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span style={{ color: "var(--text-muted)" }}>
                      Faithfulness score:
                    </span>
                    <span style={{
                      color:      METHOD_COLORS[method],
                      fontFamily: "var(--font-mono)",
                    }}>
                      {r.faithfulness_score?.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span style={{ color: "var(--text-muted)" }}>AUC:</span>
                    <span style={{ fontFamily: "var(--font-mono)" }}>
                      {r.faithfulness_auc?.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Charts */}
          {faithChartData.length > 0 && (
            <div className="grid-2 mb-4">
              <div className="card">
                <div className="section-title">
                  Deletion Test — Confidence Curves
                </div>
                <div className="text-secondary text-sm mb-3">
                  Steeper drop = explanation identified truly important regions.
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart
                    data={faithChartData}
                    margin={{ top: 4, right: 4, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e2530"
                    />
                    <XAxis
                      dataKey="pct"
                      tick={{ fill: "#4d5668", fontSize: 11 }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fill: "#4d5668", fontSize: 11 }}
                      tickFormatter={v =>
                        `${(v * 100).toFixed(0)}%`
                      }
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={v =>
                        v ? `${(v * 100).toFixed(1)}%` : "N/A"
                      }
                    />
                    <Legend
                      wrapperStyle={{
                        fontSize: 12,
                        color:    "#8b95a6",
                      }}
                    />
                    {Object.keys(result.comparison).map(m => (
                      <Line
                        key={m}
                        type="monotone"
                        dataKey={m}
                        stroke={METHOD_COLORS[m]}
                        strokeWidth={2.5}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <div className="section-title">Computation Time</div>
                <div className="text-secondary text-sm mb-3">
                  Real inference time per image on your hardware.
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={summaryBarData}
                    margin={{ top: 4, right: 4, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e2530"
                    />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: "#8b95a6", fontSize: 12 }}
                    />
                    <YAxis
                      tick={{ fill: "#4d5668", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={v => `${v.toFixed(2)}s`}
                    />
                    <Bar
                      dataKey="time"
                      fill="#4d9ef5"
                      radius={[4, 4, 0, 0]}
                      barSize={60}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Finding */}
          <div className="card" style={{
            borderColor: "var(--green-dim)"
          }}>
            <div className="section-title text-green mb-2">
              Research Findings
            </div>
            <div style={{
              fontSize:   13,
              color:      "var(--text-secondary)",
              lineHeight: 1.8,
            }}>
              {(() => {
                const sorted = Object.entries(result.comparison)
                  .sort((a, b) =>
                    (a[1].faithfulness_score || 0) -
                    (b[1].faithfulness_score || 0)
                  );
                const best    = sorted[0]?.[0];
                const fastest = Object.entries(result.comparison)
                  .sort((a, b) =>
                    a[1].computation_time_s - b[1].computation_time_s
                  )[0]?.[0];
                return (
                  <>
                    <strong style={{ color: "var(--green)" }}>
                      {best}
                    </strong>{" "}
                    produced the most faithful explanation (lowest AUC
                    under deletion curve).{" "}
                    <strong style={{ color: "var(--amber)" }}>
                      {fastest}
                    </strong>{" "}
                    was the fastest method. Grad-CAM is recommended for
                    real-time use while LIME provides model-agnostic
                    superpixel-level explanations.
                  </>
                );
              })()}
            </div>
          </div>
        </>
      )}

      {!result && !loading && (
        <div className="card" style={{
          textAlign: "center",
          padding:   48,
          color:     "var(--text-muted)",
        }}>
          <GitCompare
            size={40}
            style={{ margin: "0 auto 16px", opacity: 0.3 }}
          />
          <div>
            Upload an image and click "Run Comparison" to see
            Grad-CAM and LIME side by side.
          </div>
        </div>
      )}
    </div>
  );
}