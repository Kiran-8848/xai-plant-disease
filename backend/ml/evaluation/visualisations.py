"""
Visualisation utilities for generating publication-quality figures.

Covers:
  - Multi-method side-by-side comparison grid
  - Faithfulness curve plots
  - Robustness heatmaps
  - Class-wise mean heatmap grids
  - Similarity matrix heatmaps
  - Confusion matrix with class labels
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── Dark research theme ──────────────────────────────────────────────────────

DARK_STYLE = {
    "figure.facecolor":  "#0a0c0f",
    "axes.facecolor":    "#111318",
    "axes.edgecolor":    "#2a3040",
    "axes.labelcolor":   "#8b95a6",
    "axes.titlecolor":   "#e8ecf2",
    "text.color":        "#e8ecf2",
    "xtick.color":       "#4d5668",
    "ytick.color":       "#4d5668",
    "grid.color":        "#1e2530",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#111318",
    "legend.edgecolor":  "#1e2530",
    "legend.fontsize":   10,
    "font.family":       "monospace",
    "font.size":         11,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0a0c0f",
}

METHOD_COLORS = {
    "GradCAM": "#f0a832",
    "LIME":    "#4d9ef5",
    "SHAP":    "#f06060",
}

# Custom green heatmap colormap
GREEN_MAP = LinearSegmentedColormap.from_list(
    "plant_heat",
    ["#0a0c0f", "#0f3320", "#22d48a", "#f0a832", "#f06060"],
)


def _apply_dark_style():
    plt.rcParams.update(DARK_STYLE)


# ─────────────────────────────────────────────
#  Figure 1: Multi-method comparison grid
# ─────────────────────────────────────────────

def plot_method_comparison(
    images: List[np.ndarray],           # list of (H, W, 3) uint8 original images
    gcam_overlays: List[np.ndarray],    # list of (H, W, 3) Grad-CAM overlays
    lime_overlays: List[np.ndarray],    # list of (H, W, 3) LIME overlays
    shap_overlays: List[np.ndarray],    # list of (H, W, 3) SHAP overlays
    class_names: List[str],
    save_path: str,
) -> plt.Figure:
    """
    Creates a (4 rows × N cols) grid:
      Row 0: original images
      Row 1: Grad-CAM
      Row 2: LIME
      Row 3: SHAP
    """
    _apply_dark_style()
    n   = len(images)
    fig = plt.figure(figsize=(n * 3, 13))
    gs  = gridspec.GridSpec(4, n, figure=fig, hspace=0.08, wspace=0.04)

    row_labels = ["Original", "Grad-CAM", "LIME", "SHAP"]
    row_colors = ["#e8ecf2", METHOD_COLORS["GradCAM"], METHOD_COLORS["LIME"], METHOD_COLORS["SHAP"]]
    all_rows   = [images, gcam_overlays, lime_overlays, shap_overlays]

    for row_idx, (row_data, row_label, row_color) in enumerate(
        zip(all_rows, row_labels, row_colors)
    ):
        for col_idx, img in enumerate(row_data):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2530")
                spine.set_linewidth(0.5)

            if col_idx == 0:
                ax.set_ylabel(row_label, color=row_color, fontsize=12, fontweight="bold", labelpad=6)
            if row_idx == 0:
                name = class_names[col_idx].replace("_", " ")
                ax.set_title(
                    name[:22] + ("…" if len(name) > 22 else ""),
                    fontsize=9, color="#8b95a6", pad=4,
                )

    fig.suptitle(
        "XAI Method Comparison — Grad-CAM · LIME · SHAP",
        y=1.01, fontsize=14, color="#e8ecf2", fontweight="bold",
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 2: Faithfulness deletion curves
# ─────────────────────────────────────────────

def plot_faithfulness_curves(
    faith_results: Dict[str, List[dict]],   # {method: [faith_dict_per_image]}
    save_path: str,
    title: str = "Faithfulness — Deletion Test",
) -> plt.Figure:
    """
    Plot mean confidence curve ± std for each XAI method.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, results_list in faith_results.items():
        pcts   = np.array(results_list[0]["percentages"])
        confs  = np.array([r["confidences"] for r in results_list])
        mean_c = confs.mean(axis=0)
        std_c  = confs.std(axis=0)
        color  = METHOD_COLORS.get(method, "#ffffff")

        ax.plot(pcts, mean_c, label=f"{method} (AUC={np.trapz(mean_c, pcts):.3f})",
                color=color, linewidth=2.5)
        ax.fill_between(pcts, mean_c - std_c, mean_c + std_c, alpha=0.12, color=color)

    ax.set_xlabel("Fraction of pixels removed", fontsize=11)
    ax.set_ylabel("Model confidence", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True)

    # Annotation
    ax.annotate(
        "Steeper drop = more faithful\nexplanation",
        xy=(0.25, 0.6), fontsize=9,
        color="#4d5668", style="italic",
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 3: Robustness SSIM curves
# ─────────────────────────────────────────────

def plot_robustness_curves(
    rob_results: Dict[str, dict],    # {method: robustness_eval_result}
    perturbation_type: str,
    save_path: str,
) -> plt.Figure:
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax, ylabel in [
        ("ssim",     axes[0], "SSIM ↑"),
        ("spearman", axes[1], "Spearman ρ ↑"),
    ]:
        for method, r in rob_results.items():
            color = METHOD_COLORS.get(method, "#ffffff")
            ax.plot(r["levels"], r[metric],
                    label=method, color=color, linewidth=2.5, marker="o", markersize=5)
        ax.set_xlabel(f"{perturbation_type.capitalize()} level", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Robustness — {ylabel}", fontsize=12)
        ax.legend()
        ax.grid(True)

    fig.suptitle(
        f"Explanation Robustness Under {perturbation_type.capitalize()} Perturbations",
        fontsize=13, y=1.01,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 4: Class-wise mean heatmaps
# ─────────────────────────────────────────────

def plot_classwise_heatmaps(
    class_results: Dict[str, dict],  # from ClasswiseAnalyzer.compute_class_heatmaps
    save_path: str,
    max_classes: int = 12,
) -> plt.Figure:
    """
    Grid of mean Grad-CAM heatmaps per class, ordered by coverage score.
    """
    _apply_dark_style()

    classes = sorted(
        class_results.items(),
        key=lambda x: x[1]["coverage"],
        reverse=True,
    )[:max_classes]

    n_cols = 4
    n_rows = (len(classes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2))
    axes_flat = axes.flatten() if n_rows > 1 else np.array(axes).flatten()

    for i, (name, r) in enumerate(classes):
        ax  = axes_flat[i]
        hm  = r["mean_heatmap"]
        im  = ax.imshow(hm, cmap=GREEN_MAP, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        short = name.replace("_", " ")
        short = short[:22] + "…" if len(short) > 22 else short
        is_healthy = "healthy" in name.lower()
        label_color = "#22d48a" if is_healthy else "#f06060"

        ax.set_title(short, fontsize=9, color=label_color, pad=3)
        ax.text(
            0.03, 0.97,
            f"cov={r['coverage']:.2f}  n={r['n_images']}",
            transform=ax.transAxes, fontsize=7.5,
            color="#4d5668", va="top",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Mean Grad-CAM Heatmaps per Class  (green = healthy · red = diseased)",
        fontsize=12, y=1.01,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 5: Class similarity matrix
# ─────────────────────────────────────────────

def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> plt.Figure:
    """
    Heatmap of pairwise cosine similarity between class mean heatmaps.
    High similarity → the model uses similar visual features for both classes.
    """
    _apply_dark_style()
    import seaborn as sns

    short_names = [n.replace("_", " ")[:20] for n in class_names]
    fig, ax     = plt.subplots(figsize=(max(10, len(class_names) * 0.55),
                                       max(8, len(class_names) * 0.45)))

    sns.heatmap(
        sim_matrix,
        xticklabels=short_names,
        yticklabels=short_names,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=len(class_names) <= 12,
        fmt=".2f",
        linewidths=0.3,
        linecolor="#0a0c0f",
        ax=ax,
    )
    ax.set_title("Class-wise Heatmap Similarity Matrix", fontsize=12, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 6: Healthy vs. Diseased bar comparison
# ─────────────────────────────────────────────

def plot_healthy_vs_diseased(
    comparison: dict,   # from ClasswiseAnalyzer.healthy_vs_diseased()
    save_path: str,
) -> plt.Figure:
    """
    Side-by-side bars for healthy vs. diseased on coverage and consistency.
    """
    _apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    categories = ["healthy", "diseased"]
    colors     = ["#22d48a", "#f06060"]

    for ax, metric, ylabel in [
        (axes[0], "mean_coverage",    "Mean Heatmap Coverage ↓ (better focus)"),
        (axes[1], "mean_consistency", "Mean Explanation Consistency ↑"),
    ]:
        values = [comparison[c][metric] for c in categories]
        errors = [comparison[c].get("std_coverage", 0) for c in categories]
        bars   = ax.bar(categories, values, color=colors, alpha=0.85, width=0.45,
                        capsize=6, yerr=errors if metric == "mean_coverage" else None)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(values) * 1.3)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, color="#e8ecf2")

    fig.suptitle("Healthy vs. Diseased — XAI Explanation Properties", fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────
#  Figure 7: Full paper summary figure
# ─────────────────────────────────────────────

def plot_paper_summary(
    comparison: Dict[str, dict],   # from XAIComparator.compare()
    faith_results: Dict[str, List[dict]],
    save_path: str,
) -> plt.Figure:
    """
    2×2 summary figure suitable for the paper:
    [faithfulness curves] [radar / spider]
    [time bar]            [ssim comparison]
    """
    _apply_dark_style()
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── (0,0) Faithfulness curves ──────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    for method, results_list in faith_results.items():
        pcts   = np.array(results_list[0]["percentages"])
        confs  = np.array([r["confidences"] for r in results_list]).mean(axis=0)
        auc    = np.trapz(confs, pcts)
        ax0.plot(pcts, confs, label=f"{method} AUC={auc:.3f}",
                 color=METHOD_COLORS.get(method, "#fff"), linewidth=2.5)
    ax0.set_title("Faithfulness Curves", fontsize=12)
    ax0.set_xlabel("Fraction removed")
    ax0.set_ylabel("Confidence")
    ax0.set_ylim(0, 1)
    ax0.legend(fontsize=9)
    ax0.grid(True)

    # ── (0,1) Normalised bar comparison ────────────────────
    ax1   = fig.add_subplot(gs[0, 1])
    names = list(comparison.keys())
    faith_scores = [abs(comparison[m]["faithfulness_score"]) for m in names]
    ssim_scores  = [comparison[m]["mean_ssim"] for m in names]
    x            = np.arange(len(names))
    w            = 0.35
    ax1.bar(x - w/2, faith_scores, w, label="Faithfulness", color="#22d48a", alpha=0.85)
    ax1.bar(x + w/2, ssim_scores,  w, label="Robustness (SSIM)", color="#4d9ef5", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_title("Faithfulness vs. Robustness", fontsize=12)
    ax1.legend()
    ax1.grid(True, axis="y")

    # ── (1,0) Computation time ──────────────────────────────
    ax2    = fig.add_subplot(gs[1, 0])
    times  = [comparison[m]["computation_time_s"] for m in names]
    colors = [METHOD_COLORS.get(m, "#fff") for m in names]
    bars   = ax2.bar(names, times, color=colors, alpha=0.85, width=0.5)
    ax2.set_title("Computation Time (seconds)", fontsize=12)
    ax2.set_ylabel("seconds (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, axis="y")
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{t:.2f}s", ha="center", fontsize=10)

    # ── (1,1) Summary table ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    col_labels = ["Method", "Faith. AUC↓", "SSIM↑", "Spearman↑", "Time"]
    table_data = [
        [m,
         f"{comparison[m]['faithfulness_auc']:.4f}",
         f"{comparison[m]['mean_ssim']:.4f}",
         f"{comparison[m]['mean_spearman']:.4f}",
         f"{comparison[m]['computation_time_s']:.2f}s"]
        for m in names
    ]
    tbl = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.2, 1, 0.7],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#111318" if r > 0 else "#1e2530")
        cell.set_edgecolor("#2a3040")
        cell.set_text_props(color="#e8ecf2")
    ax3.set_title("Summary Table", fontsize=12, pad=20)

    fig.suptitle(
        "XAI for Plant Disease Classification — Research Summary",
        fontsize=15, fontweight="bold", y=1.01,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"[Figure] Saved → {save_path}")
    return fig
