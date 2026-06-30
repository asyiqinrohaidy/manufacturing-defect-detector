"""Quick smoke test: load best_model.pt, run Grad-CAM on one test image, save overlay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from model import build_model
from gradcam import GradCAM, overlay_heatmap
from dataset import EVAL_TRANSFORMS

ROOT = Path(__file__).resolve().parent.parent
device = torch.device("cpu")

checkpoint = torch.load(ROOT / "models" / "best_model.pt", map_location=device)
classes = checkpoint["classes"]
model = build_model(num_classes=len(classes), device=device, pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Grab one sample test image
test_dir = ROOT / "data" / "processed" / "test"
sample_class_dir = next((test_dir / c for c in classes if (test_dir / c).exists()))
sample_img_path = next(sample_class_dir.glob("*.jpg"))
print(f"Testing on: {sample_img_path}")

img = Image.open(sample_img_path).convert("RGB")
input_tensor = EVAL_TRANSFORMS(img).unsqueeze(0)

cam_engine = GradCAM(model, model.target_layer)
heatmap, pred_idx, confidence = cam_engine.generate(input_tensor)

print(f"Predicted class: {classes[pred_idx]} (confidence={confidence:.3f})")
print(f"Heatmap shape: {heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")

original_resized = np.array(img.resize((224, 224)))
overlay = overlay_heatmap(original_resized, heatmap)

out_path = ROOT / "outputs" / "gradcam_smoketest.png"
out_path.parent.mkdir(exist_ok=True)
Image.fromarray(overlay).save(out_path)
print(f"Saved overlay to {out_path}")
