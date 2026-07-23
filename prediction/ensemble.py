"""
=========================================
Ensemble Prediction
AI-Based Chest X-ray Disease Detection
=========================================
"""

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
    # Individual Predictions
    # --------------------------------------

    efficientnet_result = predict(
        models["efficientnet"],
        image_path
    )

    densenet_result = predict(
        models["densenet"],
        image_path
    )

    resnet_result = predict(
        models["resnet"],
        image_path
    )

    # --------------------------------------
    # Weighted Soft Voting
    # --------------------------------------

    final_probabilities = (

        ENSEMBLE_WEIGHTS["efficientnet"]
        * efficientnet_result["probabilities"]

        +

        ENSEMBLE_WEIGHTS["densenet"]
        * densenet_result["probabilities"]

        +

        ENSEMBLE_WEIGHTS["resnet"]
        * resnet_result["probabilities"]

    )

    final_index = np.argmax(final_probabilities)

    final_prediction = CLASS_NAMES[final_index]

    final_confidence = final_probabilities[final_index] * 100

    # --------------------------------------
    # Agreement
    # --------------------------------------

    predictions = [

        efficientnet_result["prediction"],

        densenet_result["prediction"],

        resnet_result["prediction"]

    ]

    agreement = predictions.count(final_prediction)

    # --------------------------------------
    # Return Everything
    # --------------------------------------

    return {

        "efficientnet": efficientnet_result,

        "densenet": densenet_result,

        "resnet": resnet_result,

        "final_prediction": final_prediction,

        "final_confidence": float(final_confidence),

        "final_probabilities": final_probabilities,

        "agreement": f"{agreement}/3"

    }