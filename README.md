# Manufacturing Defect Detector with Grad-CAM

A CNN-based surface defect classifier for manufacturing quality control, built on
the NEU Surface Defect Database (hot-rolled steel strips). The model is paired
with Grad-CAM explainability so predictions can be visually verified against
actual defect regions, rather than trusted as a black box.

## Overview

- **Task**: 6-class image classification of steel surface defects
- **Classes**: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- **Architecture**: ResNet18 (ImageNet pretrained, fine-tuned)
- **Explainability**: Grad-CAM heatmaps over the final convolutional block
- **Framework**: PyTorch, torchvision

## Dataset

NEU Surface Defect Database - 1,800 grayscale images (300 per class) of
hot-rolled steel strip surfaces, a standard benchmark for industrial defect
classification research.

| Split | Images |
|---|---|
| Train | 1,236 |
| Validation | 264 |
| Test | 270 |

## Results

Evaluated on the full 270-image held-out test set:

| Metric | Score |
|---|---|
| Test Accuracy | **100%** |
| Test Macro F1 | **1.00** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Crazing | 1.00 | 1.00 | 1.00 | 45 |
| Inclusion | 1.00 | 1.00 | 1.00 | 45 |
| Patches | 1.00 | 1.00 | 1.00 | 45 |
| Pitted Surface | 1.00 | 1.00 | 1.00 | 45 |
| Rolled-in Scale | 1.00 | 1.00 | 1.00 | 45 |
| Scratches | 1.00 | 1.00 | 1.00 | 45 |

Full confusion matrix and per-image predictions: [`outputs/eval/`](outputs/eval/)

### A note on the perfect score

100% test accuracy is a number that invites skepticism, and rightly so it's
the first thing I checked before trusting it. NEU-CLS is a well-documented
benchmark with visually distinct, lab-clean defect classes, and near-perfect
accuracy on this dataset is reported elsewhere in the literature, so the score
itself is plausible. To verify the model wasn't relying on a shortcut (e.g.
memorising crops or attending to image borders/lighting artifacts instead of
the actual defect), I ran Grad-CAM across a sample of test images from every
class. In all 30 images checked, the model's attention consistently localised
on the genuine defect texture like crack lines for crazing, pit clusters for
pitted surface, scratch marks for scratches, and so on rather than on
irrelevant regions. That's the evidence behind trusting this result rather
than just reporting it.

## Grad-CAM Examples

Heatmaps below show where the model focused when making each prediction
(red = high attention). The full set is in [`outputs/gradcam/`](outputs/gradcam/).

| Class | Observation |
|---|---|
| Crazing | Heatmap centers on the fine horizontal crack network |
| Pitted Surface | Distinct hot spots align with individual pit clusters |
| Scratches | Tight band of attention directly over the scratch line, including a low-confidence (0.725) case where a faint scratch was still correctly localised |
| Inclusion | Attention centers on the blocky dark inclusion artifact |
| Rolled-in Scale | Hot regions track the irregular rough-textured patches |
| Patches | More diffuse attention, consistent with patches being broader, lower-contrast blotchy regions |

## Project Structure

```
manufacturing-defect-detector/
├── data/
│   ├── raw/                  # Original NEU-CLS images by class
│   └── processed/            # train/val/test split (ImageFolder structure)
├── models/
│   ├── best_model.pt         # Trained checkpoint (model state + class list)
│   └── metrics.json          # Training history and final metrics
├── outputs/
│   ├── eval/                 # Confusion matrix, classification report, predictions.csv
│   └── gradcam/              # Grad-CAM heatmap overlays, organised by class
├── api/
│   └── main.py               # FastAPI service (predict + Grad-CAM endpoints)
├── frontend/
│   └── index.html            # Drag-and-drop demo UI (calls the API directly)
├── src/
│   ├── model.py              # ResNet18 model definition
│   ├── dataset.py            # DataLoader / transform definitions
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Test-set evaluation + confusion matrix
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── grad_cam_batch.py     # Batch Grad-CAM visualisation over test set
│   ├── prepare_data.py       # Train/val/test split generation
│   └── download_data.py      # Dataset download utility
└── README.md
```

## API

A FastAPI service (`api/main.py`) exposes the trained model for inference:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check, confirms model is loaded |
| `/classes` | GET | Returns the list of defect classes |
| `/predict` | POST | Upload an image, get predicted class + confidence + per-class probabilities |
| `/predict/gradcam` | POST | Upload an image, get prediction + Grad-CAM heatmap overlay (base64 PNG) |

**Run the server:**
```bash
py -3.13 -m uvicorn api.main:app --reload --port 8000
```

**Test it:**
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -F "file=@data/processed/test/crazing/crazing_101.jpg"
curl -X POST http://127.0.0.1:8000/predict/gradcam -F "file=@data/processed/test/scratches/scratches_126.jpg"
```

Both endpoints were tested live against real test images, including a low-confidence
edge case (a faint scratch correctly classified at 0.725 confidence), confirming
correct predictions and properly rendered Grad-CAM overlays end-to-end.

## Frontend

A lightweight single-page demo (`frontend/index.html`) provides drag-and-drop
image upload with live inference results — no build tools or dependencies
required, just a static HTML file that calls the FastAPI backend directly.

**Run it:**
1. Start the API server (see [API](#api) section above)
2. Open `frontend/index.html` directly in a browser

It displays the uploaded image alongside its Grad-CAM heatmap, the predicted
class, and a confidence breakdown across all 6 defect categories... useful for
quick visual demos without needing to use `curl` or Postman.

## Usage

**Train:**
```bash
py -3.13 src/train.py --epochs 15 --batch-size 32 --lr 1e-4
```

**Evaluate on test set:**
```bash
py -3.13 src/evaluate.py
```

**Generate Grad-CAM visualisations:**
```bash
py -3.13 src/grad_cam_batch.py --per-class 5
```

## Tech Stack

Python, PyTorch, torchvision (ResNet18 transfer learning), OpenCV (Grad-CAM
overlay rendering), FastAPI (model serving), scikit-learn (evaluation metrics),
pandas, matplotlib/seaborn (reporting).
