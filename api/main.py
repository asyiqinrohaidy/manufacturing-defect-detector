"""
FastAPI service for the Manufacturing Defect Detector.

Endpoints:
    GET  /health           - health check
    GET  /classes          - list of defect classes the model can predict
    POST /predict           - upload an image, get prediction (no heatmap)
    POST /predict/gradcam   - upload an image, get prediction + Grad-CAM heatmap overlay (base64 PNG)

Run:
    uvicorn api.main:app --reload --port 8000
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import build_model          # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from dataset import EVAL_TRANSFORMS    # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Manufacturing Defect Detector API",
    description="CNN-based surface defect classifier with Grad-CAM explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"model": None, "classes": None, "gradcam": None}


@app.on_event("startup")
def load_model():
    if not MODEL_PATH.exists():
        print(f"[startup] WARNING: no checkpoint found at {MODEL_PATH}. "
              f"Run `python src/train.py` first. API will return 503 until then.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes), device=DEVICE, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _state["model"] = model
    _state["classes"] = classes
    _state["gradcam"] = GradCAM(model, model.target_layer)
    print(f"[startup] Loaded model with classes: {classes}")


def _ensure_ready():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first (src/train.py).")


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/classes")
def classes():
    _ensure_ready()
    return {"classes": _state["classes"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_ready()
    img = _read_image(await file.read())
    input_tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    model = _state["model"]
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    classes_list = _state["classes"]
    pred_idx = int(probs.argmax().item())

    return JSONResponse({
        "predicted_class": classes_list[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {classes_list[i]: float(probs[i]) for i in range(len(classes_list))},
    })


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    _ensure_ready()
    img = _read_image(await file.read())
    input_tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    cam_engine = _state["gradcam"]
    heatmap, pred_idx, confidence = cam_engine.generate(input_tensor)

    classes_list = _state["classes"]
    original_resized = np.array(img.resize((224, 224)))
    overlay = overlay_heatmap(original_resized, heatmap)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "predicted_class": classes_list[pred_idx],
        "confidence": confidence,
        "gradcam_overlay_png_base64": overlay_b64,
    })
