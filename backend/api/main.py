import io
import json
import time
import base64
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.ml.models.resnet_model import load_checkpoint, get_device
from backend.ml.utils.data_utils import get_inference_transforms, denormalize
from backend.ml.xai.gradcam import GradCAM
from backend.ml.xai.lime_explainer import LIMEExplainer
from backend.ml.evaluation.metrics import FaithfulnessEvaluator


app = FastAPI(
    title       = "XAI Plant Disease API",
    description = "Grad-CAM and LIME for plant disease classification",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)


class AppState:
    model         = None
    device        = None
    class_mapping = None
    idx_to_class  = None
    transform     = None

state = AppState()


@app.on_event("startup")
async def startup():
    state.device    = get_device()
    state.transform = get_inference_transforms(224)

    ckpt_path = "backend/ml/checkpoints/best_model.pth"
    mapping   = "backend/ml/checkpoints/class_mapping.json"

    if Path(ckpt_path).exists():
        state.model, meta = load_checkpoint(ckpt_path, state.device)
        state.model.eval()
        print(f"[API] Model loaded. Classes: {meta['num_classes']}")
    else:
        from backend.ml.models.resnet_model import PlantDiseaseResNet
        state.model = PlantDiseaseResNet(
            num_classes=16, pretrained=True
        ).to(state.device)
        state.model.eval()
        print("[API] WARNING: No checkpoint found. Demo mode.")

    if Path(mapping).exists():
        with open(mapping) as f:
            data                = json.load(f)
            state.class_mapping = data["class_to_idx"]
            state.idx_to_class  = {
                int(k): v for k, v in data["idx_to_class"].items()
            }
    else:
        from backend.ml.utils.data_utils import PLANT_CLASSES
        state.class_mapping = {c: i for i, c in enumerate(PLANT_CLASSES)}
        state.idx_to_class  = {i: c for i, c in enumerate(PLANT_CLASSES)}

    print("[API] Ready.")


# ── Helpers ───────────────────────────────────────────────

def make_gradcam() -> GradCAM:
    return GradCAM(state.model, state.model.get_last_conv_layer())


def load_pil_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_base64(arr: np.ndarray) -> str:
    return pil_to_base64(Image.fromarray(arr.astype(np.uint8)))


def heatmap_to_colored_base64(heatmap: np.ndarray) -> str:
    import cv2
    h8      = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return ndarray_to_base64(colored)


@torch.no_grad()
def run_prediction(tensor: torch.Tensor):
    logits   = state.model(tensor.to(state.device))
    probs    = F.softmax(logits, dim=1)[0].cpu().numpy()
    top5_idx = np.argsort(probs)[::-1][:5]
    return {
        "pred_class": int(top5_idx[0]),
        "pred_label": state.idx_to_class.get(int(top5_idx[0]), "Unknown"),
        "confidence": float(probs[top5_idx[0]]),
        "top5": [
            {
                "class_idx": int(i),
                "label":     state.idx_to_class.get(int(i), "Unknown"),
                "prob":      float(probs[i]),
            }
            for i in top5_idx
        ],
    }


# ── Endpoints ─────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/api/classes")
async def get_classes():
    return {
        "classes": [
            {"idx": idx, "name": name}
            for idx, name in state.idx_to_class.items()
        ],
        "total": len(state.idx_to_class),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        pil    = load_pil_from_upload(await file.read())
        tensor = state.transform(pil).unsqueeze(0)
        pred   = run_prediction(tensor)

        gcam          = make_gradcam()
        heatmap, _, _ = gcam.generate(tensor)
        gcam.remove_hooks()

        original = denormalize(tensor[0])
        overlay  = GradCAM.overlay_heatmap(original, heatmap)

        return {
            **pred,
            "original_image_b64":  ndarray_to_base64(original),
            "gradcam_overlay_b64": ndarray_to_base64(overlay),
            "heatmap_b64":         heatmap_to_colored_base64(heatmap),
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/explain/gradcam")
async def explain_gradcam(
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(None),
):
    try:
        pil    = load_pil_from_upload(await file.read())
        tensor = state.transform(pil).unsqueeze(0)
        pred   = run_prediction(tensor)
        target = target_class if target_class is not None else pred["pred_class"]

        gcam          = make_gradcam()
        t0            = time.time()
        heatmap, _, _ = gcam.generate(tensor, target_class=target)
        elapsed       = time.time() - t0
        gcam.remove_hooks()

        original = denormalize(tensor[0])
        overlay  = GradCAM.overlay_heatmap(original, heatmap)

        return {
            **pred,
            "method":             "gradcam",
            "target_class":       target,
            "computation_time_s": round(elapsed, 3),
            "original_image_b64": ndarray_to_base64(original),
            "explanation_b64":    ndarray_to_base64(overlay),
            "heatmap_b64":        heatmap_to_colored_base64(heatmap),
            "heatmap_values":     heatmap.tolist(),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/explain/lime")
async def explain_lime(
    file: UploadFile = File(...),
    num_samples:  int = Query(500),
    num_features: int = Query(10),
):
    try:
        pil    = load_pil_from_upload(await file.read())
        tensor = state.transform(pil).unsqueeze(0)
        pred   = run_prediction(tensor)

        explainer = LIMEExplainer(
            state.model, state.device, 224,
            num_samples=num_samples,
            num_features=num_features,
        )
        result  = explainer.explain(pil, target_class=pred["pred_class"])
        overlay = LIMEExplainer.overlay_segments(
            result["original_image"],
            result["mask"],
            result["explanation"].segments,
        )

        return {
            **pred,
            "method":             "lime",
            "computation_time_s": round(result["computation_time"], 3),
            "original_image_b64": ndarray_to_base64(result["original_image"]),
            "explanation_b64":    ndarray_to_base64(overlay),
            "heatmap_b64":        heatmap_to_colored_base64(result["heatmap"]),
            "heatmap_values":     result["heatmap"].tolist(),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/explain/compare")
async def compare_methods(
    file: UploadFile = File(...),
    run_faithfulness: bool = Query(True),
):
    try:
        pil      = load_pil_from_upload(await file.read())
        tensor   = state.transform(pil).unsqueeze(0).to(state.device)
        pred     = run_prediction(tensor.cpu())
        target   = pred["pred_class"]
        original = denormalize(tensor.cpu()[0])

        results = {}

        # ── Grad-CAM ──────────────────────────────
        gcam          = make_gradcam()
        t0            = time.time()
        hm_gcam, _, _ = gcam.generate(tensor)
        gcam.remove_hooks()
        results["GradCAM"] = {
            "heatmap":            hm_gcam,
            "computation_time_s": round(time.time() - t0, 3),
            "explanation_b64":    ndarray_to_base64(
                GradCAM.overlay_heatmap(original, hm_gcam)
            ),
        }

        # ── LIME ──────────────────────────────────
        lim_exp = LIMEExplainer(
            state.model, state.device, 224, num_samples=200
        )
        t0      = time.time()
        lim_res = lim_exp.explain(pil, target)
        results["LIME"] = {
            "heatmap":            lim_res["heatmap"],
            "computation_time_s": round(time.time() - t0, 3),
            "explanation_b64":    ndarray_to_base64(
                LIMEExplainer.overlay_segments(
                    lim_res["original_image"],
                    lim_res["mask"],
                    lim_res["explanation"].segments,
                )
            ),
        }

        # ── Faithfulness ──────────────────────────
        comparison_output = {}
        if run_faithfulness:
            faith_eval = FaithfulnessEvaluator(state.model, state.device)
            for method_name, r in results.items():
                faith = faith_eval.evaluate(
                    tensor, r["heatmap"], target
                )
                comparison_output[method_name] = {
                    "faithfulness_score": round(faith["faithfulness_score"], 4),
                    "faithfulness_auc":   round(faith["auc"], 4),
                    "confidence_curve": {
                        "x": faith["percentages"],
                        "y": [round(c, 4) for c in faith["confidences"]],
                    },
                    "computation_time_s": r["computation_time_s"],
                    "explanation_b64":    r["explanation_b64"],
                }

        return {
            "prediction":   pred,
            "original_b64": ndarray_to_base64(original),
            "comparison":   comparison_output,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )