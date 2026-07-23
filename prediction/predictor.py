"""
=========================================
Prediction Module
AI-Based Chest X-ray Disease Detection
=========================================
"""

import torch
from PIL import Image

from prediction.config import (
    DEVICE,
    CLASS_NAMES
)

from prediction.transforms import predict_transform


# ==========================================
# PREDICT SINGLE MODEL
# ==========================================

def predict(model, image_path):

    # Load Image
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Apply Transform
    image = predict_transform(image)

    # Add Batch Dimension
    image = image.unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    return {

        "prediction": CLASS_NAMES[predicted.item()],

        "confidence": confidence.item() * 100,

        "probabilities": probabilities.squeeze().cpu().numpy()
    }