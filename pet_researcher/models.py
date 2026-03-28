from __future__ import annotations

from torch import nn
from torchvision import models


def load_backbone(name: str):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        return models.resnet18(weights=weights)
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        return models.resnet50(weights=weights)
    raise ValueError(f"Unsupported backbone: {name}")


def build_model(backbone: str, num_classes: int) -> nn.Module:
    model = load_backbone(backbone)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_fine_tune_scope(model: nn.Module, scope: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if scope == "fc":
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    if scope == "layer4":
        for name, param in model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
        return

    if scope == "layer3_4":
        for name, param in model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
        return

    if scope == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unsupported fine-tune scope: {scope}")
