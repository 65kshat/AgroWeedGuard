# utils/vit_utils.py

import torch
import torch.nn.functional as F

from torchvision import transforms
from utils.image_utils import get_class_name
from utils.model_loader import (load_vit, DEVICE)

# Transforming Images

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Prediction Class
def predict_vit(image):
    model = load_vit()

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    class_id = predicted.item()

    return {"class_id": class_id, "class_name": get_class_name(class_id), "confidence": round(confidence.item(), 4)}


# Top K Predictions
def predict_vit_topk(image, k=3):
    model = load_vit()

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)
        confidences, indices = torch.topk(probabilities, k)

    results = []

    for confidence, index in zip(confidences[0], indices[0]):
        class_id = index.item()

        results.append({"class_id": index.item(), "class_name": get_class_name(class_id), "confidence": round(confidence.item(), 4)})

    return results