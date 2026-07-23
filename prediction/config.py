"""
=========================================
Configuration File
AI-Based Chest X-ray Disease Detection
=========================================
"""

from pathlib import Path
import torch

# ==========================================
# PROJECT DIRECTORY
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMAGE_SIZE = 224

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]
NUM_CLASSES = len(CLASS_NAMES)
# ==========================================
# MODEL PATHS
# ==========================================

MODEL_DIR = PROJECT_ROOT / "models"

EFFICIENTNET_PATH = MODEL_DIR / "efficientnet_b0_best.pth"

DENSENET_PATH = MODEL_DIR / "densenet121_best.pth"

RESNET_PATH = MODEL_DIR / "resnet50_best.pth"

# ==========================================
# ENSEMBLE WEIGHTS
# ==========================================

ENSEMBLE_WEIGHTS = {
    "efficientnet": 0.50,
    "densenet": 0.30,
    "resnet": 0.20
}