"""
Batch Grad-CAM visualization for the manufacturing defect classifier.

Generates heatmap overlays for several test images per class so you can
visually confirm the model is attending to actual defect regions, and
flags misclassifications (pred != true class) in the saved filename.

Usage:
    py -3.13 src\\grad_cam_batch.py --per-class 5
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from model import build_model
from gradcam import GradCAM, overlay_heatmap
from dataset import EVAL_TRANSFORMS

ROOT = Path(__file__).resolve().parent.parent


def run(args):
    device = torch.device("cpu")

    checkpoint = torch.load(ROOT / "models" / "best_model.pt", map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes), device=device, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cam_engine = GradCAM(model, model.target_layer)

    test_dir = ROOT / "data" / "processed" / "test"
    output_dir = ROOT / "outputs" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    total, correct = 0, 0

    for class_name in classes:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"Skipping missing class dir: {class_dir}")
            continue

        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(class_dir.glob("*.jpg"))[: args.per_class]

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            input_tensor = EVAL_TRANSFORMS(img).unsqueeze(0)

            heatmap, pred_idx, confidence = cam_engine.generate(input_tensor)
            pred_class = classes[pred_idx]
            is_correct = pred_class == class_name

            total += 1
            correct += int(is_correct)

            original_resized = np.array(img.resize((224, 224)))
            overlay = overlay_heatmap(original_resized, heatmap)

            tag = "correct" if is_correct else "WRONG"
            out_name = f"{img_path.stem}_pred-{pred_class}_{tag}.png"
            Image.fromarray(overlay).save(out_class_dir / out_name)

            print(f"[{class_name}] {img_path.name} -> pred={pred_class} "
                  f"({confidence:.3f}) [{tag}]")

    print(f"\nDone. {correct}/{total} correct on visualized sample.")
    print(f"Overlays saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Grad-CAM over test set")
    parser.add_argument("--per-class", type=int, default=5,
                         help="Number of images to visualize per class")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
