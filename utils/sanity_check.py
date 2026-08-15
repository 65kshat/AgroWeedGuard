# utils/sanity_check.py
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)
from pathlib import Path

from PIL import Image

from model_loader import (
    load_yolo,
    load_cnn,
    load_vit,
    load_fasterrcnn
)

from yolo_utils import (
    get_yolo_predictions
)

from cnn_utils import (
    predict_cnn
)

from vit_utils import (
    predict_vit
)

from fasterrcnn_utils import (
    get_fasterrcnn_predictions
)

from metrics_utils import (
    measure_inference_time
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TEST_IMAGE = Path(
    input(
        "\nEnter test image path:\n"
    )
)

# --------------------------------------------------
# IMAGE CHECK
# --------------------------------------------------

print("\n" + "="*60)
print("IMAGE CHECK")
print("="*60)

if not TEST_IMAGE.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n{TEST_IMAGE}"
    )

image = Image.open(
    TEST_IMAGE
).convert("RGB")

print("✓ Image Loaded")
print("Image Size:", image.size)

# --------------------------------------------------
# MODEL LOADING CHECK
# --------------------------------------------------

print("\n" + "="*60)
print("MODEL LOADING CHECK")
print("="*60)

try:
    load_yolo()
    print("✓ YOLO Loaded")
except Exception as e:
    print("✗ YOLO Failed")
    print(e)

try:
    load_cnn()
    print("✓ CNN Loaded")
except Exception as e:
    print("✗ CNN Failed")
    print(e)

try:
    load_vit()
    print("✓ ViT Loaded")
except Exception as e:
    print("✗ ViT Failed")
    print(e)

try:
    load_fasterrcnn()
    print("✓ FasterRCNN Loaded")
except Exception as e:
    print("✗ FasterRCNN Failed")
    print(e)

# --------------------------------------------------
# YOLO TEST
# --------------------------------------------------

print("\n" + "="*60)
print("YOLO TEST")
print("="*60)

try:

    yolo_result, yolo_time = (
        measure_inference_time(
            get_yolo_predictions,
            image
        )
    )

    print(
        f"✓ YOLO Success "
        f"({yolo_time:.2f} ms)"
    )

    print(
        "Detections:",
        len(yolo_result)
    )

except Exception as e:

    print("✗ YOLO Failed")
    print(e)

# --------------------------------------------------
# CNN TEST
# --------------------------------------------------

print("\n" + "="*60)
print("CNN TEST")
print("="*60)

try:

    cnn_result, cnn_time = (
        measure_inference_time(
            predict_cnn,
            image
        )
    )

    print(
        f"✓ CNN Success "
        f"({cnn_time:.2f} ms)"
    )

    print(cnn_result)

except Exception as e:

    print("✗ CNN Failed")
    print(e)

# --------------------------------------------------
# ViT TEST
# --------------------------------------------------

print("\n" + "="*60)
print("ViT TEST")
print("="*60)

try:

    vit_result, vit_time = (
        measure_inference_time(
            predict_vit,
            image
        )
    )

    print(
        f"✓ ViT Success "
        f"({vit_time:.2f} ms)"
    )

    print(vit_result)

except Exception as e:

    print("✗ ViT Failed")
    print(e)

# --------------------------------------------------
# FasterRCNN TEST
# --------------------------------------------------

print("\n" + "="*60)
print("FasterRCNN TEST")
print("="*60)

try:

    frcnn_result, frcnn_time = (
        measure_inference_time(
            get_fasterrcnn_predictions,
            image
        )
    )

    print(
        f"✓ FasterRCNN Success "
        f"({frcnn_time:.2f} ms)"
    )

    print(
        "Detections:",
        len(frcnn_result)
    )

except Exception as e:

    print("✗ FasterRCNN Failed")
    print(e)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "="*60)
print("SANITY CHECK COMPLETE")
print("="*60)