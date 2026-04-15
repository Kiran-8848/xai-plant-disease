from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from backend.ml.utils.data_utils import get_inference_transforms, denormalize


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        self.remove_hooks()

        def forward_hook(module, input, output):
            # contiguous() + clone() ensures NOT a view
            self._activations = output.contiguous().clone().detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                # contiguous() + clone() ensures NOT a view
                self._gradients = grad_out[0].contiguous().clone().detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:

        self.model.eval()
        device            = next(self.model.parameters()).device
        image_tensor      = image_tensor.to(device)
        self._activations = None
        self._gradients   = None

        with torch.enable_grad():
            # Clone input to avoid any inplace issues upstream
            inp    = image_tensor.clone()
            output = self.model(inp)
            probs  = F.softmax(output.detach(), dim=1)

            pred_class = int(output.argmax(1).item())
            confidence = float(probs[0, pred_class].item())
            target     = target_class if target_class is not None else pred_class

            self.model.zero_grad()
            one_hot            = torch.zeros_like(output)
            one_hot[0, target] = 1.0
            output.backward(gradient=one_hot, retain_graph=False)

        # Safety check
        if self._gradients is None or self._activations is None:
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            return (
                np.ones((h, w), dtype=np.float32) * 0.5,
                pred_class,
                confidence,
            )

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        h, w    = image_tensor.shape[2], image_tensor.shape[3]
        heatmap = F.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )
        heatmap = heatmap[0, 0].cpu().numpy()

        return heatmap, pred_class, confidence

    @staticmethod
    def overlay_heatmap(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        if heatmap_rgb.shape[:2] != original_image.shape[:2]:
            heatmap_rgb = cv2.resize(
                heatmap_rgb,
                (original_image.shape[1], original_image.shape[0])
            )

        overlay = (
            alpha * heatmap_rgb + (1 - alpha) * original_image
        ).astype(np.uint8)
        return overlay

    def save(
        self,
        image_tensor: torch.Tensor,
        save_path: str,
        target_class: Optional[int] = None,
        class_name: str = "",
    ) -> dict:
        heatmap, pred_class, conf = self.generate(
            image_tensor, target_class
        )
        original = denormalize(image_tensor[0])
        overlay  = self.overlay_heatmap(original, heatmap)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(save_path)

        return {
            "heatmap":    heatmap,
            "overlay":    overlay,
            "pred_class": pred_class,
            "confidence": conf,
            "save_path":  save_path,
        }


def generate_gradcam_batch(
    model: nn.Module,
    image_paths: List[str],
    target_layer: nn.Module,
    output_dir: str,
    device: torch.device,
    image_size: int = 224,
) -> List[dict]:
    transform = get_inference_transforms(image_size)
    gcam      = GradCAM(model, target_layer)
    results   = []

    for img_path in image_paths:
        img    = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        fname  = Path(img_path).stem
        out    = str(Path(output_dir) / f"{fname}_gradcam.png")
        result = gcam.save(tensor.to(device), out)
        result["image_path"] = img_path
        results.append(result)
        print(
            f"[Grad-CAM] {fname} → "
            f"pred={result['pred_class']} "
            f"conf={result['confidence']:.3f}"
        )

    gcam.remove_hooks()
    return results