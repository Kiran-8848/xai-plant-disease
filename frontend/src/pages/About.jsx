import { BookOpen, Zap, FlaskConical, Brain, ChevronRight } from "lucide-react";

const Section = ({ title, children }) => (
  <div className="card mb-4">
    <div className="section-title mb-3">{title}</div>
    {children}
  </div>
);

const Step = ({ number, title, desc }) => (
  <div style={{ display: "flex", gap: 14, marginBottom: 14 }}>
    <div style={{
      width: 28, height: 28, borderRadius: "50%", background: "var(--green-dim)",
      color: "var(--green)", display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 12, fontWeight: 700, flexShrink: 0, marginTop: 2,
    }}>{number}</div>
    <div>
      <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", marginBottom: 3 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>{desc}</div>
    </div>
  </div>
);

const XAICard = ({ icon: Icon, name, color, badgeClass, how, math, pros, cons }) => (
  <div className="card" style={{ borderTop: `3px solid ${color}` }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <Icon size={18} style={{ color }} />
      <span style={{ fontFamily: "var(--font-display)", fontWeight: 600, fontSize: 16 }}>{name}</span>
      <span className={`badge ${badgeClass}`} style={{ marginLeft: "auto" }}>XAI Method</span>
    </div>
    <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12, lineHeight: 1.7 }}>{how}</div>
    <div style={{
      background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "10px 14px",
      fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", marginBottom: 12,
    }}>{math}</div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <div style={{ background: "var(--green-dim)", borderRadius: "var(--radius-sm)", padding: "8px 12px" }}>
        <div style={{ fontSize: 10, color: "var(--green)", fontWeight: 600, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Advantages</div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{pros}</div>
      </div>
      <div style={{ background: "var(--coral-dim)", borderRadius: "var(--radius-sm)", padding: "8px 12px" }}>
        <div style={{ fontSize: 10, color: "var(--coral)", fontWeight: 600, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Limitations</div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{cons}</div>
      </div>
    </div>
  </div>
);

const MetricCard = ({ title, formula, desc, color }) => (
  <div className="card-sm">
    <div style={{ fontSize: 13, fontWeight: 600, color, marginBottom: 6 }}>{title}</div>
    <div style={{
      background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)", padding: "8px 12px",
      fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", marginBottom: 8,
    }}>{formula}</div>
    <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{desc}</div>
  </div>
);

export default function About() {
  return (
    <div>
      <div className="page-title">Methodology</div>
      <div className="page-subtitle">
        Research design, XAI techniques, and evaluation framework
      </div>

      {/* ── Problem statement ───────────────────────── */}
      <Section title="Problem Statement">
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
          Deep learning models achieve state-of-the-art accuracy on plant disease classification,
          yet remain <strong style={{ color: "var(--text-primary)" }}>black boxes</strong> —
          their internal decision logic is opaque to agronomists and researchers.
          Without interpretability, high accuracy alone cannot justify deployment in
          real agricultural settings where incorrect decisions have significant economic impact.
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          {[
            { label: "Gap 1", desc: "No quantitative evaluation of explanations", color: "var(--coral)" },
            { label: "Gap 2", desc: "No robustness analysis under perturbations",  color: "var(--amber)" },
            { label: "Gap 3", desc: "No systematic cross-method comparison",       color: "var(--blue)" },
          ].map(g => (
            <div key={g.label} style={{
              flex: 1, padding: "10px 14px",
              borderRadius: "var(--radius-sm)",
              background: "var(--bg-elevated)",
              borderLeft: `3px solid ${g.color}`,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: g.color, marginBottom: 4 }}>{g.label}</div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{g.desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Methodology steps ───────────────────────── */}
      <Section title="Methodology Pipeline">
        <Step number={1} title="Data Preparation"
          desc="PlantVillage dataset (38 classes). Images resized to 224×224, normalized to ImageNet stats. 70/15/15 train/val/test split. Augmentation: random crop, horizontal flip, rotation ±30°, colour jitter." />
        <Step number={2} title="Model Development"
          desc="ResNet18 pre-trained on ImageNet. Final FC layer replaced: 512 → 38 classes. First 6 layers frozen during warmup, then fine-tuned. Adam optimizer, cosine LR schedule, cross-entropy loss." />
        <Step number={3} title="XAI Method Application"
          desc="Generate explanations using Grad-CAM (gradient-based), LIME (perturbation-based), and SHAP (game-theory). Each method produces a pixel-importance heatmap for the predicted class." />
        <Step number={4} title="Faithfulness Evaluation"
          desc="Deletion test: identify top-K% pixels via each XAI heatmap, remove/blur them, measure confidence drop. AUC under the deletion curve is the faithfulness score. Lower AUC = more faithful explanation." />
        <Step number={5} title="Robustness Testing"
          desc="Apply three perturbation types (Gaussian noise, blur, brightness change) to input images. Compare original vs. perturbed heatmaps using SSIM and Spearman rank correlation." />
        <Step number={6} title="Class-wise Analysis"
          desc="Analyse explanation quality separately for healthy vs. diseased leaves and across different disease types. Test whether XAI methods identify biologically meaningful disease regions." />
      </Section>

      {/* ── XAI methods ─────────────────────────────── */}
      <div className="section-title mb-3">XAI Methods Deep Dive</div>
      <div className="grid-3 mb-4">
        <XAICard
          icon={Zap}
          name="Grad-CAM"
          color="var(--amber)"
          badgeClass="badge-gradcam"
          how="Uses gradient of the class score with respect to the last convolutional feature maps. Channels weighted by their mean gradient → importance heatmap. Fast and differentiable."
          math="L^c_GradCAM = ReLU(Σ_k α^c_k · A^k)   α^c_k = (1/Z) Σ_ij ∂y^c/∂A^k_ij"
          pros="Extremely fast (~50ms). No model modification needed. High spatial resolution."
          cons="Tied to a specific layer. Cannot explain negative evidence. Coarse for small lesions."
        />
        <XAICard
          icon={FlaskConical}
          name="LIME"
          color="var(--blue)"
          badgeClass="badge-lime"
          how="Segments image into superpixels. Perturbs them randomly (1000 samples). Fits a locally linear weighted model. Superpixels with highest positive coefficients = important regions."
          math="ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)   π_x = exp(-D(x,z)²/σ²)"
          pros="Model-agnostic. Explains any black-box. Interpretable superpixel regions."
          cons="Slow (~8-10s). Unstable across runs. Sensitive to segmentation algorithm choice."
        />
        <XAICard
          icon={Brain}
          name="SHAP"
          color="var(--coral)"
          badgeClass="badge-shap"
          how="Computes Shapley values from cooperative game theory. Each pixel's contribution = its average marginal effect over all possible feature subsets. Uses DeepExplainer for efficiency."
          math="φ_i(f,x) = Σ_{S⊆F\\{i}} |S|!(|F|-|S|-1)!/|F|! · [f(S∪{i}) - f(S)]"
          pros="Theoretically grounded. Provides signed contributions. Satisfies fairness axioms."
          cons="Slow (~12s). Memory intensive. Background distribution choice affects results."
        />
      </div>

      {/* ── Evaluation metrics ───────────────────────── */}
      <Section title="Evaluation Metrics">
        <div className="grid-2 gap-3">
          <MetricCard
            title="Faithfulness Score"
            color="var(--green)"
            formula="Faith = -AUC(confidence_curve)  where  confidence_curve[k] = f(x with top-k% pixels removed)"
            desc="Measures how much the model relies on the regions identified as important. A good explanation should produce a steep confidence drop when its highlighted regions are removed."
          />
          <MetricCard
            title="Stability (SSIM)"
            color="var(--blue)"
            formula="SSIM(H, H') = (2μ_H μ_H' + c1)(2σ_HH' + c2) / (μ_H² + μ_H'² + c1)(σ_H² + σ_H'² + c2)"
            desc="Structural Similarity Index between the original and perturbed-input heatmaps. Values close to 1 indicate a robust explanation that doesn't change drastically with small input noise."
          />
          <MetricCard
            title="Spearman Rank Correlation"
            color="var(--amber)"
            formula="ρ = 1 - 6Σd_i² / n(n²-1)   where d_i = rank(H_i) - rank(H'_i)"
            desc="Rank correlation of pixel importance between original and perturbed explanations. Measures whether the relative ordering of important pixels is preserved under perturbations."
          />
          <MetricCard
            title="Model Performance"
            color="var(--purple)"
            formula="F1 = 2·(Precision·Recall)/(Precision+Recall)   Macro-F1 = (1/C)Σ F1_c"
            desc="Standard classification metrics: accuracy, per-class precision/recall/F1, confusion matrix. Macro-F1 is used as the primary metric due to class imbalance in PlantVillage."
          />
        </div>
      </Section>

      {/* ── Research contributions ──────────────────── */}
      <Section title="Research Contributions">
        {[
          { num: "C1", title: "Quantitative Faithfulness Evaluation", color: "var(--green)",
            desc: "First systematic deletion-test evaluation across Grad-CAM, LIME, and SHAP for plant disease images. Generates AUC-based faithfulness scores enabling direct comparison." },
          { num: "C2", title: "Cross-Method Comparative Analysis", color: "var(--blue)",
            desc: "Side-by-side comparison on identical test images. Reveals trade-offs between faithfulness, stability, visual coherence, and computation time not reported in prior work." },
          { num: "C3", title: "Robustness Under Domain Perturbations", color: "var(--amber)",
            desc: "Tests explanation stability under realistic agricultural perturbations (lighting variation, blur from camera shake, sensor noise). Relevant for deployment in field conditions." },
          { num: "C4", title: "Class-wise Explanation Analysis", color: "var(--coral)",
            desc: "Investigates whether XAI methods consistently focus on disease lesion regions vs. leaf background. Healthy vs. diseased comparison validates biological plausibility of explanations." },
        ].map(c => (
          <div key={c.num} style={{ display: "flex", gap: 14, marginBottom: 12 }}>
            <div style={{
              padding: "2px 8px", borderRadius: "var(--radius-sm)",
              background: "var(--bg-elevated)", color: c.color,
              fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 600,
              alignSelf: "flex-start", flexShrink: 0, marginTop: 2,
            }}>{c.num}</div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", marginBottom: 3 }}>{c.title}</div>
              <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>{c.desc}</div>
            </div>
          </div>
        ))}
      </Section>

      {/* ── Citation / dataset ──────────────────────── */}
      <div className="card-sm" style={{ borderColor: "var(--border-light)" }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Dataset</div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
          Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. <em>arXiv:1511.08060</em>. 38 classes, 54,306 images.
        </div>
      </div>
    </div>
  );
}
