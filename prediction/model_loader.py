"""
=========================================
Model Loader
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
# LOAD EFFICIENTNET-B0
# ==========================================

def load_efficientnet():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            EFFICIENTNET_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# LOAD DENSENET121
# ==========================================

def load_densenet():

    model = timm.create_model(
        "densenet121",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            DENSENET_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# LOAD RESNET50
# ==========================================

def load_resnet():

    model = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            RESNET_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# LOAD ALL MODELS
# ==========================================

def load_all_models():

    print("\nLoading AI Doctors...\n")

    efficientnet = load_efficientnet()
    print("✅ EfficientNet-B0 Loaded")

    densenet = load_densenet()
    print("✅ DenseNet121 Loaded")

    resnet = load_resnet()
    print("✅ ResNet50 Loaded")

    print("\n🏥 All AI Doctors are Ready!\n")

    return {
        "efficientnet": efficientnet,
        "densenet": densenet,
        "resnet": resnet
    }