import torch
import torch.nn as nn
import torchvision.models as models

import config


class TinyCNN(nn.Module):
    """Small 3-block CNN baseline. Fast on CPU, good for small labeled datasets."""
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,  32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(freeze_backbone: bool = False) -> nn.Module:
    """Return the configured model (tiny or mobilenet)."""
    n_classes = len(config.TEXTURE_CLASSES)

    if config.TEXTURE_MODEL == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)
        return model

    return TinyCNN(n_classes=n_classes)
