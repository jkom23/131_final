"""Texture inference: classify a single BGR crop -> (texture_class, confidence)."""
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

import config
from texture.model import build_model

_model: Optional[torch.nn.Module] = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.TEXTURE_INPUT_SIZE, config.TEXTURE_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = build_model()
        model.load_state_dict(
            torch.load(config.TEXTURE_CHECKPOINT, map_location="cpu")
        )
        model.eval()
        _model = model
    return _model


def classify_texture(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Classify surface texture of a region crop.
    Returns (texture_class, confidence) where confidence ∈ [0, 1].
    Falls back to ("other", 0.0) if no checkpoint is available.
    """
    if not os.path.exists(config.TEXTURE_CHECKPOINT):
        return "other", 0.0

    model = _load_model()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)   # 1×C×H×W

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        confidence = probs[idx].item()

    return config.TEXTURE_CLASSES[idx], confidence
