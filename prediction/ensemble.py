"""
=========================================
Memory-Optimized Sequential Ensemble Prediction
AI-Based Chest X-ray Disease Detection
=========================================
"""

import gc
import torch
import numpy as np
from prediction.predictor import load_single_model, predict_from_tensor
from prediction.preprocessing import preprocess_image
from prediction.config import CLASS_NAMES, ENSEMBLE_WEIGHTS

def predict_model_isolated(model_name: str, tensor):
    """Loads a single model into RAM, runs inference, and frees memory immediately."""
    model = load_single_model(model_name)
    model.eval()
    
    with torch.no_grad():
        res = predict_from_tensor(model, tensor)
    
    # Immediately wipe the model from memory
    del model
    gc.collect()
    
    return res

def ensemble_predict(models_dict, image_path: str):
    # Preprocess image once into tensor
    tensor = preprocess_image(image_path)

    # 1. Run each model sequentially, freeing RAM after each step
    eff_res = predict_model_isolated("efficientnet", tensor)
    dense_res = predict_model_isolated("densenet", tensor)
    res_res = predict_model_isolated("resnet", tensor)

    del tensor
    gc.collect()

    # 2. Weighted Soft Voting
    probs_eff = np.array(eff_res["probabilities"])
    probs_dense = np.array(dense_res["probabilities"])
    probs_res = np.array(res_res["probabilities"])

    final_probabilities = (
        ENSEMBLE_WEIGHTS["efficientnet"] * probs_eff +
        ENSEMBLE_WEIGHTS["densenet"] * probs_dense +
        ENSEMBLE_WEIGHTS["resnet"] * probs_res
    )

    final_index = int(np.argmax(final_probabilities))
    final_prediction = CLASS_NAMES[final_index]
    final_confidence = float(final_probabilities[final_index] * 100)

    votes = [eff_res["prediction"], dense_res["prediction"], res_res["prediction"]]
    agreement = votes.count(final_prediction)

    return {
        "efficientnet": eff_res,
        "densenet": dense_res,
        "resnet": res_res,
        "final_prediction": final_prediction,
        "final_confidence": round(final_confidence, 2),
        "final_probabilities": final_probabilities.tolist(),
        "agreement": f"{agreement}/3"
    }