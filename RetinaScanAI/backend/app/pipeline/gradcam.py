"""
Stage 4: Explainability. Implements Grad-CAM (Selvaraju et al., 2017,
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization") from scratch with plain PyTorch forward/backward hooks —
no extra dependency needed.

This is the feature that answers the PS26038 requirement for a "human-in-
the-loop workflow": instead of a bare class prediction, every result ships
with a heatmap over the exact retinal region that drove the decision, so a
clinician can visually verify it in seconds rather than trusting a black box.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a HxW float heatmap in [0, 1], same spatial size as the
        input tensor (upsampled from the target conv layer's feature map)."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(base_bgr_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a Grad-CAM heatmap onto the (already resized) base image."""
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, base_bgr_uint8, 1 - alpha, 0)
    return overlay


def lesion_evidence_summary(cam: np.ndarray, top_fraction: float = 0.1) -> dict:
    """Cheap, interpretable summary stats of the CAM used in the referral
    report (not a substitute for the visual heatmap, just a quick number)."""
    flat = cam.flatten()
    k = max(1, int(len(flat) * top_fraction))
    top_vals = np.sort(flat)[-k:]
    return {
        "hot_region_fraction": float((cam > 0.5).sum()) / cam.size,
        "mean_activation_in_hot_region": float(top_vals.mean()),
    }
