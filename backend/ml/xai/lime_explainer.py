"""
LIME (Local Interpretable Model-Agnostic Explanations) for image classification.

How LIME works for images:
  1. Segment the image into superpixels (perceptually similar regions).
  2. Create N perturbed samples: randomly toggle superpixels on/off.
  3. Get model predictions on all perturbed samples.
  4. Fit a weighted linear model (importance ∝ similarity to original).
  5. Top-K superpixels with highest positive weights = important regions.

Key insight: LIME is MODEL-AGNOSTIC — it treats the model as a black box.
This means it can explain ANY classifier, at the cost of being slower than
gradient-based methods like Grad-CAM.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from backend.ml.utils.data_utils import get_inference_transforms, denormalize


class LIMEExplainer:
    """
    Wraps lime_image.LimeImageExplainer for PyTorch models.

    Usage:
        explainer = LIMEExplainer(model, device)
        result = explainer.explain(pil_image, num_samples=1000)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        image_size: int = 224,
        num_samples: int = 1000,
        num_features: int = 10,
    ):
        self.model        = model
        self.device       = device
        self.image_size   = image_size
        self.num_samples  = num_samples
        self.num_features = num_features
        self.transform    = get_inference_transforms(image_size)
        self._lime        = lime_image.LimeImageExplainer(verbose=False)

    # ── Prediction function (required by LIME) ─────────────────────────

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        LIME calls this with a batch of uint8 HWC images (N, H, W, 3).
        Must return probabilities (N, num_classes).
        """
        self.model.eval()
        batch = []
        for img in images:
            pil   = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            batch.append(self.transform(pil))
        batch_tensor = torch.stack(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        return probs  # (N, num_classes)

    # ── Core explanation ────────────────────────────────────────────────

    def explain(
        self,
        pil_image: Image.Image,
        target_class: Optional[int] = None,
        num_samples: Optional[int] = None,
        num_features: Optional[int] = None,
    ) -> dict:
        """
        Returns dict with:
            explanation      - lime Explanation object
            mask             - np.ndarray (H, W) binary mask of important regions
            heatmap          - np.ndarray (H, W) float [0,1] soft importance
            pred_class       - int
            confidence       - float
            computation_time - float (seconds)
        """
        n_samples  = num_samples  or self.num_samples
        n_features = num_features or self.num_features

        img_np = np.array(pil_image.resize((self.image_size, self.image_size)))

        # Get model prediction first
        tensor  = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs      = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        target     = target_class if target_class is not None else pred_class

        print(f"  [LIME] Generating {n_samples} samples...", end=" ", flush=True)
        t0 = time.time()

        explanation = self._lime.explain_instance(
            image          = img_np,
            classifier_fn  = self._predict_fn,
            top_labels     = min(5, len(probs)),
            hide_color     = 0,
            num_samples    = n_samples,
            segmentation_fn= SegmentationAlgorithm(
                "quickshift", kernel_size=4, max_dist=200, ratio=0.2
            ),
        )

        comp_time = time.time() - t0
        print(f"done in {comp_time:.1f}s")

        # Extract binary mask (top positive segments)
        temp_img, mask = explanation.get_image_and_mask(
            label        = target,
            positive_only= True,
            num_features = n_features,
            hide_rest    = False,
        )

        # Soft importance map (per-superpixel weights)
        segments = explanation.segments
        weights  = dict(explanation.local_exp.get(target, []))
        heatmap  = np.zeros_like(segments, dtype=float)
        for seg_id, weight in weights.items():
            heatmap[segments == seg_id] = max(weight, 0)

        if heatmap.max() > 1e-8:
            heatmap /= heatmap.max()

        return {
            "explanation":      explanation,
            "mask":             mask.astype(np.float32),
            "heatmap":          heatmap.astype(np.float32),
            "pred_class":       pred_class,
            "confidence":       confidence,
            "computation_time": comp_time,
            "original_image":   img_np,
        }

    # ── Visualisation ───────────────────────────────────────────────────

    @staticmethod
    def overlay_segments(
        original_image: np.ndarray,   # (H, W, 3) uint8
        mask: np.ndarray,             # (H, W) binary
        segments: np.ndarray,         # (H, W) segment IDs
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Highlight important superpixels with a green overlay + boundary."""
        overlay = original_image.copy().astype(float)
        overlay[mask == 1] = (
            overlay[mask == 1] * (1 - alpha) +
            np.array([0, 200, 100]) * alpha
        )
        overlay = overlay.astype(np.uint8)
        overlay = (mark_boundaries(overlay, segments) * 255).astype(np.uint8)
        return overlay

    def save(
        self,
        pil_image: Image.Image,
        save_path: str,
        target_class: Optional[int] = None,
    ) -> dict:
        """Run LIME, save visualisation, return results dict."""
        result = self.explain(pil_image, target_class)

        seg = result["explanation"].segments
        vis = self.overlay_segments(result["original_image"], result["mask"], seg)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(save_path)
        result["save_path"] = save_path

        return result


# ─────────────────────────────────────────────
#  Batch helper
# ─────────────────────────────────────────────

def generate_lime_batch(
    model: torch.nn.Module,
    image_paths: List[str],
    output_dir: str,
    device: torch.device,
    image_size: int = 224,
    num_samples: int = 1000,
) -> List[dict]:
    explainer = LIMEExplainer(model, device, image_size, num_samples)
    results   = []

    for img_path in image_paths:
        pil   = Image.open(img_path).convert("RGB")
        fname = Path(img_path).stem
        out   = str(Path(output_dir) / f"{fname}_lime.png")
        result = explainer.save(pil, out)
        result["image_path"] = img_path
        results.append(result)

    return results
