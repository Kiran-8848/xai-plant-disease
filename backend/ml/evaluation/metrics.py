"""
Quantitative evaluation of XAI explanations.

Experiment 3: Faithfulness (Deletion Test)
─────────────────────────────────────────────
Delete the most-important pixels (identified by each XAI method)
and measure how fast the model's confidence drops.
A BETTER explanation → steeper confidence drop.

Metric: AUC of confidence-drop curve (lower AUC = more faithful explanation).
Also computed: accuracy drop at each deletion percentage.

Experiment 4: Robustness (Stability)
─────────────────────────────────────
Apply perturbations (noise, blur, brightness) to the input image.
Measure how much the XAI explanation changes.

Metric: Mean Structural Similarity (SSIM) or MSE between original and
perturbed explanations. Higher similarity = more robust.

Experiment 5: Comparison Summary
─────────────────────────────────
Aggregate faithfulness + robustness + computation time per XAI method.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

from backend.ml.utils.data_utils import get_inference_transforms, denormalize


# ─────────────────────────────────────────────
#  Perturbation helpers
# ─────────────────────────────────────────────

def add_gaussian_noise(image_tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise to a normalised tensor."""
    return image_tensor + torch.randn_like(image_tensor) * std


def add_blur(image_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply Gaussian blur via F.conv2d."""
    if kernel_size <= 1:
        return image_tensor
    # Use reflection padding to keep edges clean
    k    = kernel_size | 1   # ensure odd
    pad  = k // 2
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(k, dtype=torch.float32) - pad
    g      = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g      = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(image_tensor.shape[1], 1, 1, 1).to(image_tensor.device)
    padded = F.pad(image_tensor, [pad] * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=image_tensor.shape[1])


def adjust_brightness(image_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.clamp(image_tensor * factor, min=-3.0, max=3.0)


# ─────────────────────────────────────────────
#  Faithfulness evaluation
# ─────────────────────────────────────────────

class FaithfulnessEvaluator:
    """
    Performs the deletion test (pixel/region removal).

    Algorithm:
      1. Rank pixels by importance (descending) using XAI heatmap.
      2. For each deletion percentage p ∈ [0.05, 0.10, 0.20, 0.30, 0.50]:
          a. Mask the top-p% pixels (set to mean or blur).
          b. Get model confidence on masked image.
      3. Plot confidence vs. deletion curve.
      4. Faithfulness score = AUC of curve (lower = more faithful).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        percentages: List[float] = None,
        masking_method: str = "blur",        # "blur" | "zero" | "mean"
    ):
        self.model   = model
        self.device  = device
        self.pcts    = percentages or [0.05, 0.10, 0.20, 0.30, 0.50]
        self.masking = masking_method

    def _mask_image(
        self,
        image_tensor: torch.Tensor,     # (1, C, H, W)
        heatmap: np.ndarray,            # (H, W) importance in [0, 1]
        fraction: float,
    ) -> torch.Tensor:
        """Zero/blur top-fraction of pixels ranked by heatmap importance."""
        flat_sorted = np.argsort(heatmap.flatten())[::-1]
        threshold   = flat_sorted[int(fraction * len(flat_sorted))]
        thresh_val  = heatmap.flatten()[threshold]

        mask = (heatmap >= thresh_val).astype(np.float32)   # 1 = important pixel
        mask_tensor = torch.from_numpy(mask).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        if self.masking == "zero":
            return image_tensor * (1 - mask_tensor)
        elif self.masking == "blur":
            blurred = add_blur(image_tensor, kernel_size=31)
            return image_tensor * (1 - mask_tensor) + blurred * mask_tensor
        else:  # mean
            mean    = image_tensor.mean(dim=(2, 3), keepdim=True)
            return image_tensor * (1 - mask_tensor) + mean * mask_tensor

    @torch.no_grad()
    def evaluate(
        self,
        image_tensor: torch.Tensor,   # (1, C, H, W)
        heatmap: np.ndarray,          # (H, W)
        target_class: int,
    ) -> Dict:
        """Returns faithfulness metrics for a single image."""
        self.model.eval()
        tensor = image_tensor.to(self.device)

        # Baseline confidence (no deletion)
        baseline_conf = float(
            F.softmax(self.model(tensor), dim=1)[0, target_class].item()
        )

        confs = [baseline_conf]
        drops = [0.0]

        for pct in self.pcts:
            masked       = self._mask_image(tensor, heatmap, pct)
            conf         = float(F.softmax(self.model(masked), dim=1)[0, target_class].item())
            confs.append(conf)
            drops.append(baseline_conf - conf)

        x_points = [0.0] + list(self.pcts)
        auc      = float(np.trapezoid(confs, x_points))

        return {
            "percentages":      x_points,
            "confidences":      confs,
            "drops":            drops,
            "auc":              auc,
            "baseline_conf":    baseline_conf,
            "faithfulness_score": -auc,     # higher = more faithful
        }


# ─────────────────────────────────────────────
#  Robustness evaluation
# ─────────────────────────────────────────────

class RobustnessEvaluator:
    """
    Measures stability of explanations under input perturbations.

    Metric options:
      - SSIM between original and perturbed heatmaps (higher = more stable).
      - Spearman rank correlation of pixel importance rankings.
    """

    def __init__(
        self,
        noise_levels:      List[float] = None,
        blur_levels:       List[int]   = None,
        brightness_levels: List[float] = None,
        num_repetitions:   int         = 5,
    ):
        self.noise_levels      = noise_levels      or [0.01, 0.05, 0.10]
        self.blur_levels       = blur_levels       or [1, 3, 5]
        self.brightness_levels = brightness_levels or [0.8, 1.2]
        self.n_reps            = num_repetitions

    def evaluate(
        self,
        image_tensor: torch.Tensor,             # (1, C, H, W)
        explain_fn:   Callable,                 # fn(tensor) → heatmap (H, W)
        perturbation_type: str = "noise",       # "noise" | "blur" | "brightness"
    ) -> Dict:
        """
        Returns stability metrics across perturbation levels.
        explain_fn takes a tensor and returns a 2D heatmap.
        """
        original_heatmap = explain_fn(image_tensor)
        results          = {"levels": [], "ssim": [], "spearman": []}

        if perturbation_type == "noise":
            levels = self.noise_levels
            perturb_fn = lambda t, v: add_gaussian_noise(t, v)
        elif perturbation_type == "blur":
            levels = self.blur_levels
            perturb_fn = lambda t, v: add_blur(t, v)
        elif perturbation_type == "brightness":
            levels = self.brightness_levels
            perturb_fn = lambda t, v: adjust_brightness(t, v)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        for level in levels:
            ssim_scores, spearman_scores = [], []
            for _ in range(self.n_reps):
                perturbed = perturb_fn(image_tensor, level)
                pert_heatmap = explain_fn(perturbed)

                s = ssim(
                    original_heatmap.astype(np.float32),
                    pert_heatmap.astype(np.float32),
                    data_range=1.0,
                )
                rho, _ = spearmanr(original_heatmap.flatten(), pert_heatmap.flatten())
                ssim_scores.append(s)
                spearman_scores.append(rho)

            results["levels"].append(level)
            results["ssim"].append(float(np.mean(ssim_scores)))
            results["spearman"].append(float(np.mean(spearman_scores)))

        results["mean_ssim"]     = float(np.mean(results["ssim"]))
        results["mean_spearman"] = float(np.mean(results["spearman"]))
        return results


# ─────────────────────────────────────────────
#  Comparison aggregator
# ─────────────────────────────────────────────

class XAIComparator:
    """
    Runs Experiments 3, 4, 5 across multiple XAI methods and aggregates results.
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def compare(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        methods: Dict[str, Callable],   # {"GradCAM": fn, "LIME": fn, "SHAP": fn}
    ) -> Dict:
        """
        methods: dict mapping method name → explain_fn(tensor) → heatmap (H, W)
        Returns comparison table dict.
        """
        faith_eval  = FaithfulnessEvaluator(self.model, self.device)
        robust_eval = RobustnessEvaluator()

        results = {}
        for name, explain_fn in methods.items():
            print(f"\n[Comparing] {name}...")
            t0      = time.time()
            heatmap = explain_fn(image_tensor)
            elapsed = time.time() - t0

            faith   = faith_eval.evaluate(image_tensor, heatmap, target_class)
            robust  = robust_eval.evaluate(image_tensor, explain_fn, "noise")

            results[name] = {
                "faithfulness_auc":    faith["auc"],
                "faithfulness_score":  faith["faithfulness_score"],
                "confidence_drops":    faith["drops"],
                "mean_ssim":           robust["mean_ssim"],
                "mean_spearman":       robust["mean_spearman"],
                "computation_time_s":  elapsed,
                "heatmap":             heatmap,
                "faithfulness_curve":  {
                    "x": faith["percentages"],
                    "y": faith["confidences"],
                },
            }

        return results

    @staticmethod
    def summary_table(comparison: Dict) -> str:
        """Print a readable summary table."""
        header = f"{'Method':<12} {'Faith.Score':>12} {'SSIM':>8} {'Spearman':>10} {'Time(s)':>9}"
        rows   = [header, "─" * len(header)]
        for name, r in comparison.items():
            rows.append(
                f"{name:<12} {r['faithfulness_score']:>12.4f} "
                f"{r['mean_ssim']:>8.4f} {r['mean_spearman']:>10.4f} "
                f"{r['computation_time_s']:>9.2f}"
            )
        return "\n".join(rows)
