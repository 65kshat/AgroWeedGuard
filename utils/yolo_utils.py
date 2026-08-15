# utils/yolo_utils.py

import cv2
import numpy as np

from PIL import Image

from utils.model_loader import (
    load_yolo,
    CLASS_NAMES
)

# --------------------------------------------------
# Predict
# --------------------------------------------------

def predict_yolo(image):

    model = load_yolo()

    results = model(
        image,
        verbose=False
    )

    return results


# --------------------------------------------------
# Draw Bounding Boxes
# --------------------------------------------------

def draw_yolo_boxes(image):

    results = predict_yolo(image)

    image_np = np.array(image)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            label = (
                f"{CLASS_NAMES[class_id]} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                image_np,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image_np,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return image_np


# --------------------------------------------------
# Detection Summary
# --------------------------------------------------

def get_yolo_predictions(image):

    results = predict_yolo(image)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": round(
                        confidence,
                        4
                    )
                }
            )

    return detections