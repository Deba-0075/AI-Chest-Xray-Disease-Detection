"""
=========================================
Ensemble Prediction Pipeline
AI-Based Chest X-ray Disease Detection
=========================================
"""

import torch
import numpy as np

from prediction.predictor import predict
from prediction.config import (
    CLASS_NAMES,
    ENSEMBLE_WEIGHTS
)


def ensemble_predict(models, image_input):
    """Executes high-speed forward passes using resident in-memory models."""

    # 1. Forward passes with torch.no_grad()
    with torch.no_grad():
        efficientnet_result = predict(models["efficientnet"], image_input)
        densenet_result = predict(models["densenet"], image_input)
        resnet_result = predict(models["resnet"], image_input)

    # 2. Weighted Soft Voting
    probs_eff = np.array(efficientnet_result["probabilities"], dtype=np.float32)
    probs_dense = np.array(densenet_result["probabilities"], dtype=np.float32)
    probs_res = np.array(resnet_result["probabilities"], dtype=np.float32)

    final_probabilities = (
        ENSEMBLE_WEIGHTS["efficientnet"] * probs_eff
        + ENSEMBLE_WEIGHTS["densenet"] * probs_dense
        + ENSEMBLE_WEIGHTS["resnet"] * probs_res
    )

    final_index = int(np.argmax(final_probabilities))
    final_prediction = CLASS_NAMES[final_index]
    final_confidence = float(final_probabilities[final_index] * 100)

    # 3. Model Consensus
    predictions = [
        efficientnet_result["prediction"],
        densenet_result["prediction"],
        resnet_result["prediction"]
    ]
    agreement = predictions.count(final_prediction)

    return {
        "efficientnet": efficientnet_result,
        "densenet": densenet_result,
        "resnet": resnet_result,
        "final_prediction": final_prediction,
        "final_confidence": round(final_confidence, 2),
        "final_probabilities": [float(p) for p in final_probabilities],
        "agreement": f"{agreement}/3"
    }