"""
Model definition: ResNet18 backbone (transfer learning) fine-tuned for
6-class manufacturing surface defect classification.
"""
import torch
import torch.nn as nn
from torchvision import models


class DefectClassifier(nn.Module):
    def __init__(self, num_classes: int = 6, freeze_backbone: bool = False, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception as e:
            # Pretrained weight download can fail in network-restricted environments.
            # Fall back to random init so the pipeline still runs end-to-end.
            print(f"[model.py] Could not download pretrained weights ({e}); "
                  f"falling back to random initialization.")
            backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone
        # Keep a handle on the last conv block for Grad-CAM
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 6, freeze_backbone: bool = False, device: str = "cpu", pretrained: bool = True):
    model = DefectClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone, pretrained=pretrained)
    return model.to(device)
