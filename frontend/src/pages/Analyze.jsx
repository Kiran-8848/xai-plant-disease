import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Loader2, Info, Zap, FlaskConical } from "lucide-react";
import axios from "axios";

const API = "http://localhost:8000/api";

const METHODS = [
  {
    id:    "gradcam",
    label: "Grad-CAM",
    icon:  Zap,
    color: "badge-gradcam",
    desc:  "Fast gradient-based heatmap"
  },
  {
    id:    "lime",
    label: "LIME",
    icon:  FlaskConical,
    color: "badge-lime",
    desc:  "Superpixel perturbation (~8s)"
  },
];

export default function Analyze() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [method,  setMethod]  = useState("gradcam");
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
    accept:   { "image/*": [".jpg", ".jpeg", ".png", ".bmp"] },
    maxFiles: 1,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const { data } = await axios.post(
        `${API}/explain/${method}`, fd
      );
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const formatClass = (name = "") =>
    name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div>
      <div className="page-title">Analyze Image</div>
      <div className="page-subtitle">
        Upload a plant leaf image to classify and explain the model's decision.
      </div>

      <div className="grid-2">
        {/* ── Left: upload + controls ── */}
        <div className="flex-col gap-4">
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "active" : ""}`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img
                src={preview}
                alt="preview"
                style={{
                  maxHeight:    200,
                  borderRadius: "var(--radius)",
                  margin:       "0 auto",
                  display:      "block",
                }}
              />
            ) : (
              <>
                <Upload
                  size={32}
                  style={{ color: "var(--text-muted)", margin: "0 auto 12px" }}
                />
                <div style={{ color: "var(--text-secondary)", fontSize: 14 }}>
                  Drop a leaf image here, or click to upload
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 6 }}>
                  JPG, PNG, BMP — PlantVillage dataset
                </div>
              </>
            )}
          </div>

          {/* Method selector */}
          <div className="card-sm">
            <div className="section-title mb-2">XAI Method</div>
            <div className="flex-col gap-2">
              {METHODS.map(m => {
                const Icon = m.icon;
                return (
                  <button
                    key={m.id}
                    onClick={() => setMethod(m.id)}
                    style={{
                      display:      "flex",
                      alignItems:   "center",
                      gap:          10,
                      padding:      "10px 12px",
                      borderRadius: "var(--radius-sm)",
                      background:   method === m.id
                        ? "var(--bg-elevated)"
                        : "transparent",
                      border: method === m.id
                        ? "1px solid var(--border-light)"
                        : "1px solid transparent",
                      cursor:    "pointer",
                      textAlign: "left",
                      width:     "100%",
                    }}
                  >
                    <Icon
                      size={15}
                      style={{
                        color: method === m.id
                          ? "var(--green)"
                          : "var(--text-muted)"
                      }}
                    />
                    <div>
                      <div style={{
                        fontSize:   13,
                        fontWeight: 500,
                        color:      method === m.id
                          ? "var(--text-primary)"
                          : "var(--text-secondary)",
                      }}>
                        {m.label}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        {m.desc}
                      </div>
                    </div>
                    {method === m.id && (
                      <div style={{ marginLeft: "auto" }}>
                        <span className={`badge ${m.color}`}>selected</span>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleAnalyze}
            disabled={!file || loading}
            style={{
              width:           "100%",
              justifyContent:  "center",
              padding:         "12px",
            }}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ width: 16, height: 16 }} />
                Analyzing...
              </>
            ) : (
              "Run Analysis"
            )}
          </button>

          {error && (
            <div style={{
              padding:      12,
              borderRadius: "var(--radius-sm)",
              background:   "var(--coral-dim)",
              color:        "var(--coral)",
              fontSize:     13,
            }}>
              {error}
            </div>
          )}
        </div>

        {/* ── Right: results ── */}
        <div>
          {result ? (
            <div className="flex-col gap-3">
              {/* Prediction */}
              <div className="card-sm">
                <div className="flex items-center gap-2 mb-3">
                  <span className="badge badge-green">Prediction</span>
                  <span
                    className="text-mono text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {result.computation_time_s}s
                  </span>
                </div>
                <div style={{
                  fontSize:    18,
                  fontWeight:  600,
                  fontFamily:  "var(--font-display)",
                  marginBottom: 8,
                }}>
                  {formatClass(result.pred_label)}
                </div>
                <div style={{
                  fontSize:     13,
                  color:        "var(--text-secondary)",
                  marginBottom: 12,
                }}>
                  Confidence:{" "}
                  <span style={{
                    color:      "var(--green)",
                    fontFamily: "var(--font-mono)",
                  }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Top 5 */}
                {result.top5?.map((t, i) => (
                  <div key={i} className="confidence-bar">
                    <div style={{
                      width:        160,
                      fontSize:     11,
                      color:        "var(--text-muted)",
                      whiteSpace:   "nowrap",
                      overflow:     "hidden",
                      textOverflow: "ellipsis",
                    }}>
                      {formatClass(t.label)}
                    </div>
                    <div className="confidence-bar-track">
                      <div
                        className="confidence-bar-fill"
                        style={{
                          width:      `${t.prob * 100}%`,
                          background: i === 0
                            ? "var(--green)"
                            : "var(--border-light)",
                        }}
                      />
                    </div>
                    <div style={{
                      width:      44,
                      fontSize:   11,
                      fontFamily: "var(--font-mono)",
                      color:      "var(--text-secondary)",
                      textAlign:  "right",
                    }}>
                      {(t.prob * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* Explanation images */}
              <div className="card-sm">
                <div className="flex items-center gap-2 mb-3">
                  <span className={`badge ${
                    METHODS.find(m => m.id === method)?.color
                  }`}>
                    {METHODS.find(m => m.id === method)?.label}
                  </span>
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    Explanation
                  </span>
                </div>

                <div className="grid-2 gap-2">
                  <div>
                    <div className="text-sm text-muted mb-2">Original</div>
                    <img
                      src={`data:image/png;base64,${result.original_image_b64}`}
                      alt="original"
                      className="explanation-image"
                    />
                  </div>
                  <div>
                    <div className="text-sm text-muted mb-2">Explanation</div>
                    <img
                      src={`data:image/png;base64,${result.explanation_b64}`}
                      alt="explanation"
                      className="explanation-image"
                    />
                  </div>
                </div>

                {result.heatmap_b64 && (
                  <div className="mt-3">
                    <div className="text-sm text-muted mb-2">Raw Heatmap</div>
                    <img
                      src={`data:image/png;base64,${result.heatmap_b64}`}
                      alt="heatmap"
                      className="explanation-image"
                    />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card" style={{
              height:          "100%",
              minHeight:       300,
              display:         "flex",
              flexDirection:   "column",
              alignItems:      "center",
              justifyContent:  "center",
              color:           "var(--text-muted)",
              textAlign:       "center",
            }}>
              <Info size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
              <div style={{ fontSize: 14 }}>
                Upload an image and select a method<br />
                to see XAI explanations here.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}