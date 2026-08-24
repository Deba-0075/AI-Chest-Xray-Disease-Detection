"""
=========================================
Model Loader (On-Demand Low RAM)
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


# ==========================================
# LOAD SINGLE MODEL ON-DEMAND
# ==========================================

def load_single_model(model_name: str):
    """
    Instantiates and loads weights for one model at a time.
    Keeps RAM overhead under 150MB by never holding all three models at once.
    """
    if model_name == "efficientnet":
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(
            torch.load(EFFICIENTNET_PATH, map_location=DEVICE)
        )

    elif model_name == "densenet":
        model = timm.create_model(
            "densenet121",
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(
            torch.load(DENSENET_PATH, map_location=DEVICE)
        )

    elif model_name == "resnet":
        model = timm.create_model(
            "resnet50",
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(
            torch.load(RESNET_PATH, map_location=DEVICE)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(DEVICE)
    model.eval()
    return model