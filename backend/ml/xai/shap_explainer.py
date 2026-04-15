"""
SHAP (SHapley Additive exPlanations) for image classification.

How SHAP works for images:
  1. Use a background distribution (mean of N reference images) as baseline.
  2. Partition image into patches (or superpixels).
  3. Compute Shapley values: how much does each patch's *presence vs. baseline*
     contribute to the model's prediction for a given class?
  4. Positive SHAP values → patch pushes prediction TOWARD the class.
  5. Negative SHAP values → patch pushes prediction AWAY from the class.

Key differences from Grad-CAM / LIME:
  - Game-theory grounded: SHAP satisfies efficiency, symmetry, and dummy axioms.
  - Slower than Grad-CAM, but gives signed values (positive & negative contributions).
  - DeepExplainer (used here) leverages gradients internally but is still
    model-specific (unlike LIME).
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import shap

from backend.ml.utils.data_utils import get_inference_transforms, denormalize


class SHAPExplainer:
    """
    Wraps shap.DeepExplainer for PyTorch image classifiers.

    Usage:
        explainer = SHAPExplainer(model, background_loader, device)
        result = explainer.explain(image_tensor)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_samples: torch.Tensor,   # (N, C, H, W) — reference images
        device: torch.device,
    ):
        self.model   = model
        self.device  = device
        model.eval()

        background = background_samples.to(device)
        # shap.DeepExplainer expects model and background on same device
        self.explainer = shap.DeepExplainer(model, background)

    # ── Core explanation ────────────────────────────────────────────────

    def explain(
        self,
        image_tensor: torch.Tensor,          # (1, C, H, W)
        target_class: Optional[int] = None,
    ) -> dict:
        """
        Returns dict with:
            shap_values      - np.ndarray (C, H, W) for target class
            pos_heatmap      - np.ndarray (H, W) positive contributions
            neg_heatmap      - np.ndarray (H, W) |negative| contributions
            combined_heatmap - np.ndarray (H, W) all contributions, signed
            pred_class       - int
            confidence       - float
            computation_time - float
        """
        self.model.eval()
        tensor = image_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            logits     = self.model(tensor)
            probs      = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        target     = target_class if target_class is not None else pred_class

        print(f"  [SHAP] Computing Shapley values for class {target}...", end=" ", flush=True)
        t0 = time.time()

        # shap_values: list[np.ndarray] — one per output class
        # Each element: (1, C, H, W)
        shap_vals = self.explainer.shap_values(tensor)
        comp_time = time.time() - t0
        print(f"done in {comp_time:.1f}s")

        # Extract target class
        if isinstance(shap_vals, list):
            sv = shap_vals[target][0]           # (C, H, W)
        else:
            sv = shap_vals[0, :, :, :, target]  # newer SHAP API

        # Aggregate channels → 2D heatmaps
        sv_mean      = sv.mean(axis=0)           # (H, W) signed
        pos_heatmap  = np.maximum(sv_mean, 0)
        neg_heatmap  = np.abs(np.minimum(sv_mean, 0))

        def _norm(x):
            return x / x.max() if x.max() > 1e-8 else x

        return {
            "shap_values":      sv,
            "pos_heatmap":      _norm(pos_heatmap).astype(np.float32),
            "neg_heatmap":      _norm(neg_heatmap).astype(np.float32),
            "combined_heatmap": sv_mean.astype(np.float32),
            "pred_class":       pred_class,
            "confidence":       confidence,
            "computation_time": comp_time,
        }

    # ── Visualisation ───────────────────────────────────────────────────

    @staticmethod
    def visualize(
        original_image: np.ndarray,    # (H, W, 3) uint8
        pos_heatmap: np.ndarray,       # (H, W) [0,1]
        neg_heatmap: np.ndarray,       # (H, W) [0,1]
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Green overlay = positive SHAP (supports prediction).
        Red overlay   = negative SHAP (suppresses prediction).
        """
        overlay = original_image.astype(float)

        # Green channel for positive
        green_mask = (cv2.applyColorMap(
            (pos_heatmap * 255).astype(np.uint8), cv2.COLORMAP_SUMMER
        ))
        green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2RGB).astype(float)

        # Red channel for negative
        red_mask = (cv2.applyColorMap(
            (neg_heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT
        ))
        red_mask = cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB).astype(float)

        # Blend: high positive → more green, high negative → more red
        pos_weight = pos_heatmap[:, :, None]
        neg_weight = neg_heatmap[:, :, None]

        result = (
            overlay * (1 - alpha * (pos_weight + neg_weight)) +
            green_mask * alpha * pos_weight +
            red_mask   * alpha * neg_weight
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def save(
        self,
        image_tensor: torch.Tensor,
        save_path: str,
        target_class: Optional[int] = None,
    ) -> dict:
        result       = self.explain(image_tensor, target_class)
        original     = denormalize(image_tensor[0])
        vis          = self.visualize(original, result["pos_heatmap"], result["neg_heatmap"])

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(save_path)
        result["save_path"] = save_path
        result["original"]  = original

        return result


# ─────────────────────────────────────────────
#  Background sample builder
# ─────────────────────────────────────────────

def build_background_samples(
    loader: torch.utils.data.DataLoader,
    n: int = 100,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Collect N random training images to use as SHAP background."""
    samples = []
    for images, _ in loader:
        samples.append(images)
        if sum(x.shape[0] for x in samples) >= n:
            break
    background = torch.cat(samples, dim=0)[:n]
    return background.to(device)


# ─────────────────────────────────────────────
#  Batch helper
# ─────────────────────────────────────────────

def generate_shap_batch(
    model: torch.nn.Module,
    image_paths: List[str],
    background: torch.Tensor,
    output_dir: str,
    device: torch.device,
    image_size: int = 224,
) -> List[dict]:
    transform = get_inference_transforms(image_size)
    explainer = SHAPExplainer(model, background, device)
    results   = []

    for img_path in image_paths:
        pil    = Image.open(img_path).convert("RGB")
        tensor = transform(pil).unsqueeze(0)
        fname  = Path(img_path).stem
        out    = str(Path(output_dir) / f"{fname}_shap.png")
        result = explainer.save(tensor, out)
        result["image_path"] = img_path
        results.append(result)

    return results
