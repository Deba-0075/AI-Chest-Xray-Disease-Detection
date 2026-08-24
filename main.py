import io
import gc
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from prediction.config import DEVICE, CLASS_NAMES
from prediction.model_loader import load_all_models
from prediction.ensemble import ensemble_predict

app = FastAPI(
    title="Chest X-Ray Disease Detection API",
    description="High-speed multi-model ensemble diagnosis API (EfficientNet-B0, DenseNet121, ResNet50).",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS: Dict[str, Any] = {}

def get_models():
    """Lazy-load models on the first request to allow instant server port binding."""
    global MODELS
    if not MODELS:
        print("🧠 Loading models into memory...")
        MODELS = load_all_models()
        print("✅ Models loaded successfully!")
    return MODELS

@app.get("/")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "classes": CLASS_NAMES
    }

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload a JPEG or PNG file."
        )

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")

    try:
        # Retrieve models
        models = get_models()

        # Run forward-pass ensemble
        result = ensemble_predict(models, pil_image)

        response_data = {
            "final_diagnosis": {
                "prediction": result["final_prediction"],
                "confidence": round(float(result["final_confidence"]), 2),
                "agreement": result["agreement"]
            },
            "doctors": {
                "doctor1": {
                    "name": "AI Doctor 1",
                    "model": "EfficientNet-B0",
                    "prediction": result["efficientnet"]["prediction"],
                    "confidence": round(float(result["efficientnet"]["confidence"]), 2),
                    "probabilities": {
                        cls: round(float(prob) * 100, 2)
                        for cls, prob in zip(CLASS_NAMES, result["efficientnet"]["probabilities"])
                    }
                },
                "doctor2": {
                    "name": "AI Doctor 2",
                    "model": "DenseNet121",
                    "prediction": result["densenet"]["prediction"],
                    "confidence": round(float(result["densenet"]["confidence"]), 2),
                    "probabilities": {
                        cls: round(float(prob) * 100, 2)
                        for cls, prob in zip(CLASS_NAMES, result["densenet"]["probabilities"])
                    }
                },
                "doctor3": {
                    "name": "AI Doctor 3",
                    "model": "ResNet50",
                    "prediction": result["resnet"]["prediction"],
                    "confidence": round(float(result["resnet"]["confidence"]), 2),
                    "probabilities": {
                        cls: round(float(prob) * 100, 2)
                        for cls, prob in zip(CLASS_NAMES, result["resnet"]["probabilities"])
                    }
                }
            }
        }
        
        # Explicitly release image memory
        del pil_image
        del image_bytes
        gc.collect()

        return response_data

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")