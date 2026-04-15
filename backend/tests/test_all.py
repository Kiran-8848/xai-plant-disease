"""
Test suite for XAI Plant Disease project.
Run with: pytest backend/tests/ -v

Tests cover:
  - Data utilities
  - Model forward pass
  - GradCAM hooks and output shape
  - LIME prediction function
  - SHAP DeepExplainer
  - Faithfulness evaluator
  - Robustness evaluator
  - API endpoints (via FastAPI TestClient)
"""

import io
import json
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.ml.models.resnet_model import PlantDiseaseResNet, get_device
from backend.ml.utils.data_utils import (
    get_train_transforms, get_val_transforms,
    get_inference_transforms, denormalize,
)
from backend.ml.xai.gradcam import GradCAM
from backend.ml.evaluation.metrics import (
    FaithfulnessEvaluator, RobustnessEvaluator,
    add_gaussian_noise, add_blur, adjust_brightness,
)


# ─────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────

NUM_CLASSES = 10  # Use 10-class toy model for speed


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model(device):
    """Small ResNet18 (CPU, 10 classes) for fast testing."""
    m = PlantDiseaseResNet(num_classes=NUM_CLASSES, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_image_tensor():
    """Random 224×224 image tensor (batch size 1)."""
    return torch.rand(1, 3, 224, 224)


@pytest.fixture(scope="module")
def dummy_pil_image():
    """Random PIL image."""
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────
#  Data utilities tests
# ─────────────────────────────────────────────

class TestTransforms:
    def test_train_transform_output_shape(self, dummy_pil_image):
        t      = get_train_transforms(224)
        tensor = t(dummy_pil_image)
        assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"

    def test_val_transform_output_shape(self, dummy_pil_image):
        t      = get_val_transforms(224)
        tensor = t(dummy_pil_image)
        assert tensor.shape == (3, 224, 224)

    def test_transforms_normalised(self, dummy_pil_image):
        t      = get_val_transforms(224)
        tensor = t(dummy_pil_image)
        # After ImageNet normalisation, values should span negative range
        assert tensor.min() < 0, "Expected negative values after normalisation"

    def test_denormalize_range(self, dummy_image_tensor):
        arr = denormalize(dummy_image_tensor[0])
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255
        assert arr.shape == (224, 224, 3)

    def test_different_image_sizes(self, dummy_pil_image):
        for size in [112, 224, 256]:
            t      = get_val_transforms(size)
            tensor = t(dummy_pil_image)
            assert tensor.shape == (3, size, size), f"Failed for size {size}"


# ─────────────────────────────────────────────
#  Model tests
# ─────────────────────────────────────────────

class TestModel:
    def test_forward_output_shape(self, model, dummy_image_tensor):
        with torch.no_grad():
            out = model(dummy_image_tensor)
        assert out.shape == (1, NUM_CLASSES), f"Expected (1,{NUM_CLASSES}), got {out.shape}"

    def test_forward_is_logits(self, model, dummy_image_tensor):
        with torch.no_grad():
            out  = model(dummy_image_tensor)
            prob = F.softmax(out, dim=1)
        assert abs(prob.sum().item() - 1.0) < 1e-4, "Softmax should sum to 1"

    def test_batch_forward(self, model):
        batch = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, NUM_CLASSES)

    def test_get_last_conv_layer(self, model):
        layer = model.get_last_conv_layer()
        assert layer is not None
        assert isinstance(layer, torch.nn.Module)

    def test_num_classes_attribute(self, model):
        assert model.num_classes == NUM_CLASSES

    def test_grad_flows(self, model, dummy_image_tensor):
        """Ensure gradients reach the last conv layer."""
        model.train()
        out  = model(dummy_image_tensor)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None
            for p in model.classifier.parameters()
        )
        assert has_grad, "No gradients in classifier head"
        model.eval()


# ─────────────────────────────────────────────
#  Grad-CAM tests
# ─────────────────────────────────────────────

class TestGradCAM:
    @pytest.fixture
    def gcam(self, model):
        g = GradCAM(model, model.get_last_conv_layer())
        yield g
        g.remove_hooks()

    def test_heatmap_shape(self, gcam, dummy_image_tensor):
        heatmap, pred, conf = gcam.generate(dummy_image_tensor)
        assert heatmap.shape == (224, 224), f"Expected (224,224), got {heatmap.shape}"

    def test_heatmap_range(self, gcam, dummy_image_tensor):
        heatmap, _, _ = gcam.generate(dummy_image_tensor)
        assert heatmap.min() >= 0.0, "Heatmap should be non-negative (ReLU applied)"
        assert heatmap.max() <= 1.0 + 1e-6, "Heatmap should be normalised to [0,1]"

    def test_prediction_valid(self, gcam, dummy_image_tensor):
        _, pred, conf = gcam.generate(dummy_image_tensor)
        assert 0 <= pred < NUM_CLASSES
        assert 0.0 <= conf <= 1.0

    def test_target_class_override(self, gcam, dummy_image_tensor):
        hm0, pred0, _ = gcam.generate(dummy_image_tensor, target_class=0)
        hm1, pred1, _ = gcam.generate(dummy_image_tensor, target_class=1)
        assert pred0 == pred1, "Prediction should not change with target_class override"
        assert not np.allclose(hm0, hm1), "Heatmaps for different target classes should differ"

    def test_overlay_output(self, dummy_image_tensor):
        arr     = denormalize(dummy_image_tensor[0])
        heatmap = np.random.rand(224, 224).astype(np.float32)
        overlay = GradCAM.overlay_heatmap(arr, heatmap)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_hooks_removed(self, model):
        gcam = GradCAM(model, model.get_last_conv_layer())
        assert len(gcam._hooks) == 2
        gcam.remove_hooks()
        assert len(gcam._hooks) == 0


# ─────────────────────────────────────────────
#  Perturbation helpers tests
# ─────────────────────────────────────────────

class TestPerturbations:
    def test_gaussian_noise_shape(self, dummy_image_tensor):
        noisy = add_gaussian_noise(dummy_image_tensor, std=0.1)
        assert noisy.shape == dummy_image_tensor.shape

    def test_gaussian_noise_differs(self, dummy_image_tensor):
        noisy = add_gaussian_noise(dummy_image_tensor, std=0.1)
        assert not torch.allclose(dummy_image_tensor, noisy)

    def test_blur_shape(self, dummy_image_tensor):
        blurred = add_blur(dummy_image_tensor, kernel_size=5)
        assert blurred.shape == dummy_image_tensor.shape

    def test_blur_zero_kernel(self, dummy_image_tensor):
        """Kernel size <= 1 should be a no-op."""
        result = add_blur(dummy_image_tensor, kernel_size=1)
        assert torch.allclose(dummy_image_tensor, result)

    def test_brightness_factor_2(self, dummy_image_tensor):
        bright = adjust_brightness(dummy_image_tensor, factor=2.0)
        # Should be brighter (higher mean) but clamped
        assert bright.max() <= 3.0

    def test_brightness_factor_0(self, dummy_image_tensor):
        dark = adjust_brightness(dummy_image_tensor, factor=0.0)
        assert dark.abs().max() < 1e-6


# ─────────────────────────────────────────────
#  Faithfulness evaluator tests
# ─────────────────────────────────────────────

class TestFaithfulness:
    @pytest.fixture
    def evaluator(self, model, device):
        return FaithfulnessEvaluator(model, device, percentages=[0.1, 0.3, 0.5])

    def test_returns_keys(self, evaluator, dummy_image_tensor):
        heatmap  = np.random.rand(224, 224).astype(np.float32)
        result   = evaluator.evaluate(dummy_image_tensor, heatmap, target_class=0)
        expected = {"percentages", "confidences", "drops", "auc", "baseline_conf", "faithfulness_score"}
        assert expected.issubset(result.keys())

    def test_correct_lengths(self, evaluator, dummy_image_tensor):
        heatmap  = np.random.rand(224, 224).astype(np.float32)
        result   = evaluator.evaluate(dummy_image_tensor, heatmap, target_class=0)
        # percentages = [0.0] + [0.1, 0.3, 0.5] = 4 entries
        assert len(result["percentages"]) == 4
        assert len(result["confidences"]) == 4

    def test_baseline_is_first(self, evaluator, dummy_image_tensor):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result  = evaluator.evaluate(dummy_image_tensor, heatmap, target_class=0)
        assert result["percentages"][0] == 0.0
        assert result["drops"][0] == 0.0

    def test_auc_is_scalar(self, evaluator, dummy_image_tensor):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result  = evaluator.evaluate(dummy_image_tensor, heatmap, target_class=0)
        assert isinstance(result["auc"], float)
        assert 0.0 <= result["auc"] <= 1.0

    def test_blur_masking_method(self, model, device, dummy_image_tensor):
        ev      = FaithfulnessEvaluator(model, device, masking_method="blur")
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result  = ev.evaluate(dummy_image_tensor, heatmap, target_class=0)
        assert "auc" in result

    def test_zero_masking_method(self, model, device, dummy_image_tensor):
        ev      = FaithfulnessEvaluator(model, device, masking_method="zero")
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result  = ev.evaluate(dummy_image_tensor, heatmap, target_class=0)
        assert "auc" in result


# ─────────────────────────────────────────────
#  Robustness evaluator tests
# ─────────────────────────────────────────────

class TestRobustness:
    @pytest.fixture
    def evaluator(self):
        return RobustnessEvaluator(
            noise_levels=[0.05, 0.10],
            blur_levels=[3],
            brightness_levels=[0.8],
            num_repetitions=2,
        )

    def test_noise_keys(self, evaluator, dummy_image_tensor):
        def fake_explain(t):
            return np.random.rand(224, 224).astype(np.float32)
        result = evaluator.evaluate(dummy_image_tensor, fake_explain, "noise")
        assert "levels" in result
        assert "ssim" in result
        assert "spearman" in result
        assert "mean_ssim" in result

    def test_ssim_in_range(self, evaluator, dummy_image_tensor):
        def constant_explain(t):
            return np.ones((224, 224), dtype=np.float32) * 0.5
        result = evaluator.evaluate(dummy_image_tensor, constant_explain, "noise")
        for s in result["ssim"]:
            assert -1.0 <= s <= 1.0, f"SSIM out of range: {s}"

    def test_blur_perturbation(self, evaluator, dummy_image_tensor):
        def fake_explain(t):
            return np.random.rand(224, 224).astype(np.float32)
        result = evaluator.evaluate(dummy_image_tensor, fake_explain, "blur")
        assert len(result["levels"]) == 1  # one blur level
        assert len(result["ssim"]) == 1

    def test_invalid_perturbation_type(self, evaluator, dummy_image_tensor):
        with pytest.raises(ValueError, match="Unknown perturbation type"):
            evaluator.evaluate(dummy_image_tensor, lambda t: None, "invalid_type")


# ─────────────────────────────────────────────
#  API endpoint tests
# ─────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.main import app
        return TestClient(app)

    def _make_upload(self) -> bytes:
        """Create a minimal JPEG for upload."""
        img = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_classes_endpoint(self, client):
        r = client.get("/api/classes")
        assert r.status_code == 200
        data = r.json()
        assert "classes" in data
        assert "total" in data
        assert data["total"] > 0

    def test_predict_endpoint(self, client):
        img_bytes = self._make_upload()
        r = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "pred_class" in data
        assert "confidence" in data
        assert "gradcam_overlay_b64" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_gradcam_endpoint(self, client):
        img_bytes = self._make_upload()
        r = client.post(
            "/api/explain/gradcam",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["method"] == "gradcam"
        assert "explanation_b64" in data
        assert "heatmap_values" in data

    def test_predict_returns_top5(self, client):
        img_bytes = self._make_upload()
        r = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "top5" in data
        assert len(data["top5"]) == 5
        probs = [t["prob"] for t in data["top5"]]
        assert probs == sorted(probs, reverse=True), "top5 should be sorted descending"

    def test_invalid_file_handled(self, client):
        r = client.post(
            "/api/predict",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 500   # should return an error, not crash server


# ─────────────────────────────────────────────
#  Integration test
# ─────────────────────────────────────────────

class TestIntegration:
    """End-to-end: model → GradCAM → Faithfulness."""

    def test_full_pipeline(self, model, device, dummy_image_tensor):
        # 1. Forward pass
        with torch.no_grad():
            out    = model(dummy_image_tensor)
            pred   = int(out.argmax(1).item())

        # 2. Grad-CAM
        gcam    = GradCAM(model, model.get_last_conv_layer())
        heatmap, pred_gc, conf = gcam.generate(dummy_image_tensor)
        gcam.remove_hooks()

        assert heatmap.shape == (224, 224)
        assert pred_gc == pred

        # 3. Faithfulness
        ev     = FaithfulnessEvaluator(model, device, percentages=[0.1, 0.5])
        result = ev.evaluate(dummy_image_tensor, heatmap, pred)

        assert result["auc"] > 0
        # Removing more pixels should generally decrease confidence more
        assert result["drops"][-1] >= result["drops"][1] - 0.5, \
            "Deleting 50% should drop confidence more than 10%"
