"""
Grad-CAM implementation for visualizing which regions of a defect image
the CNN focused on when making its prediction.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W) preprocessed image tensor
        Returns: heatmap (H, W) in range [0, 1], predicted class index, confidence
        """
        self.model.eval()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to input image size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))

        return cam, class_idx, confidence


def overlay_heatmap(original_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    original_rgb_uint8: (H, W, 3) uint8 RGB image
    heatmap: (H, W) float in [0, 1]
    Returns: (H, W, 3) uint8 RGB overlay
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colored + (1 - alpha) * original_rgb_uint8).astype(np.uint8)
    return overlay
