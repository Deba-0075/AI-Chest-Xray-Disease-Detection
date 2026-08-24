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

def ensemble_predict(models, image_input):
    """
    Runs forward-pass inference sequentially across 3 models,
    releasing memory activations after each step to prevent OOM.
    """

    # --------------------------------------
    # 1. Run Each Model Sequentially + Free RAM
    # --------------------------------------

    efficientnet_result = predict(
        models["efficientnet"],
        image_input
    )
    gc.collect()

    densenet_result = predict(
        models["densenet"],
        image_input
    )
    gc.collect()

    resnet_result = predict(
        models["resnet"],
        image_input
    )
    gc.collect()

    # --------------------------------------
    # 2. Weighted Soft Voting
    # --------------------------------------

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

    # Convert to pure Python list before freeing arrays
    final_probs_list = [float(p) for p in final_probabilities]

    # Clean up NumPy buffers
    del probs_eff, probs_dense, probs_res, final_probabilities
    gc.collect()

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
        "final_probabilities": final_probs_list,
        "agreement": f"{agreement}/3"
    }