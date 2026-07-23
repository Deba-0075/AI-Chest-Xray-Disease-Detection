"""
=========================================
Image Transformations
AI-Based Chest X-ray Disease Detection
=========================================
"""

from torchvision import transforms

from prediction.config import IMAGE_SIZE


# ==========================================
# IMAGE TRANSFORM
# ==========================================

predict_transform = transforms.Compose([
    
    transforms.Resize((256, 256)),
    
    transforms.CenterCrop(IMAGE_SIZE),
    
    transforms.ToTensor(),
    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])