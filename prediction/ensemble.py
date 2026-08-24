"""
=========================================
Memory-Optimized Sequential Ensemble Prediction
AI-Based Chest X-ray Disease Detection
=========================================
"""

import gc
import numpy as np

from prediction.predictor import predict
from prediction.config import (
    CLASS_NAMES,
    ENSEMBLE_WEIGHTS
)


# ==========================================
# ENSEMBLE PREDICTION
# ==========================================

def ensemble_predict(models, image_path):

    # --------------------------------------
    # 1. Run Each Model Sequentially + Free RAM
    # --------------------------------------

    efficientnet_result = predict(
        models["efficientnet"],
        image_path
    )
    gc.collect()

    densenet_result = predict(
        models["densenet"],
        image_path
    )
    gc.collect()

    resnet_result = predict(
        models["resnet"],
        image_path
    )
    gc.collect()

    # --------------------------------------
    # 2. Weighted Soft Voting
    # --------------------------------------

    probs_eff = np.array(efficientnet_result["probabilities"])
    probs_dense = np.array(densenet_result["probabilities"])
    probs_res = np.array(resnet_result["probabilities"])

    final_probabilities = (
        ENSEMBLE_WEIGHTS["efficientnet"] * probs_eff
        + ENSEMBLE_WEIGHTS["densenet"] * probs_dense
        + ENSEMBLE_WEIGHTS["resnet"] * probs_res
    )

    final_index = int(np.argmax(final_probabilities))
    final_prediction = CLASS_NAMES[final_index]
    final_confidence = float(final_probabilities[final_index] * 100)

    # --------------------------------------
    # 3. Model Agreement Consensus
    # --------------------------------------

    predictions = [
        efficientnet_result["prediction"],
        densenet_result["prediction"],
        resnet_result["prediction"]
    ]

    agreement = predictions.count(final_prediction)

    # --------------------------------------
    # 4. JSON-Serializable Output
    # --------------------------------------

    return {
        "efficientnet": efficientnet_result,
        "densenet": densenet_result,
        "resnet": resnet_result,
        "final_prediction": final_prediction,
        "final_confidence": round(final_confidence, 2),
        "final_probabilities": final_probabilities.tolist(),
        "agreement": f"{agreement}/3"
    }