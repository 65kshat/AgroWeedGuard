# utils/fasterrcnn_utils.py

import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from utils.model_loader import (load_fasterrcnn, DEVICE)
from utils.image_utils import get_class_name


# Image Transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()])

# Prediction Function
def predict_fasterrcnn(image, confidence_threshold=0.5):
    model = load_fasterrcnn()
    image_tensor = transform(image).to(DEVICE)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    keep = prediction["scores"] >= confidence_threshold

    prediction["boxes"] = prediction["boxes"][keep]
    prediction["labels"] = prediction["labels"][keep]
    prediction["scores"] = prediction["scores"][keep]

    return prediction


# Drawing Bounding Boxes
def draw_fasterrcnn_boxes(image, confidence_threshold=0.5):
    prediction = predict_fasterrcnn(image, confidence_threshold)

    image_np = np.array(image)

    original_h, original_w = image_np.shape[:2]

    scale_x = original_w / 640
    scale_y = original_h / 640

    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        x1, y1, x2, y2 = box.cpu().numpy()

        # Rescale back to original image
        x1 *= scale_x
        x2 *= scale_x

        y1 *= scale_y
        y2 *= scale_y

        class_id = int(label.item()) - 1
        confidence = float(score.item())

        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(image_np, f"{get_class_name(class_id)} {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return image_np


# Detection Summary
def get_fasterrcnn_predictions(image, confidence_threshold=0.5):
    prediction = predict_fasterrcnn(image, confidence_threshold)

    detections = []

    for label, score in zip(prediction["labels"], prediction["scores"]):
        class_id = int(label.item()) - 1

        detections.append({"class_id": class_id, "class_name": get_class_name(class_id), "confidence": round(float(score.item()), 4)})

    return detections


# Combined Prediction
def run_fasterrcnn(image, confidence_threshold=0.5):
    prediction = predict_fasterrcnn(image, confidence_threshold)

    image_np = np.array(image).copy()
    original_h, original_w = image_np.shape[:2]

    scale_x = original_w / 640
    scale_y = original_h / 640

    detections = []

    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        x1, y1, x2, y2 = (box.cpu().numpy())

        # Rescale bounding box to original image   
        x1 *= scale_x
        x2 *= scale_x

        y1 *= scale_y
        y2 *= scale_y

        class_id = int(label.item()) - 1
        confidence = float(score.item())
        class_name = get_class_name(class_id)

        # Detection Information        
        detections.append({"class_id": class_id, "class_name": class_name, "confidence": round(confidence, 4)})

        # Bounding Box        
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(image_np, f"{class_name} {confidence:.2f}", (int(x1), max(int(y1) - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return {"detections": detections, "image": image_np}