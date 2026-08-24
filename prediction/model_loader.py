"""
=========================================
Model Loader (Preload into RAM for AWS)
AI-Based Chest X-ray Disease Detection
=========================================
"""

import torch
import timm

from prediction.config import (
    DEVICE,
    NUM_CLASSES,
    EFFICIENTNET_PATH,
    DENSENET_PATH,
    RESNET_PATH
)


def load_efficientnet():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_densenet():
    model = timm.create_model("densenet121", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(DENSENET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_resnet():
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_all_models():
    print("🧠 Loading models into RAM...")
    models = {
        "efficientnet": load_efficientnet(),
        "densenet": load_densenet(),
        "resnet": load_resnet()
    }
    print("✅ All 3 AI Doctor models loaded successfully!")
    return models