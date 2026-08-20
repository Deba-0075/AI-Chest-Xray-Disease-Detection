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
    tensor_image = predict_transform(image)

    # Add Batch Dimension
    tensor_input = tensor_image.unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(tensor_input)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "prediction": CLASS_NAMES[predicted.item()],
        "class_index": predicted.item(),
        "confidence": confidence.item() * 100,
        "probabilities": probabilities.squeeze().cpu().numpy(),
        "tensor_input": tensor_input
    }