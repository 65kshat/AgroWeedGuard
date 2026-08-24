# utils/image_utils.py

import numpy as np
from PIL import Image

# Class Labels
CLASS_NAMES = {
    0: "weed: mollugo verticillata",
    1: "weed: amaranthus palmeri",
    2: "weed: eclipta",
    3: "weed: portulaca oleracea",
    4: "weed: amaranthus tuberculatus",
    5: "weed: euphorbia maculata",
    6: "weed: ipomoea indica",
    7: "weed: eleusine indica",
    8: "weed: sida rhombifolia",
    9: "weed: senna obtusifolia",
    10: "weed: physalis angulata",
    11: "weed: ambrosia artemisiifolia"}


def get_class_name(class_id):
    return CLASS_NAMES.get(int(class_id), f"Unknown Class ({class_id})")


# Open Uploaded Image
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    return image

# PIL --> NumPy
def pil_to_numpy(image):
    return np.array(image)

# NumPy --> PIL
def numpy_to_pil(image):
    return Image.fromarray(image.astype(np.uint8))

# Get Image Information
def get_image_info(image):
    width, height = image.size

    return {"width": width, "height": height, "mode": image.mode}


# Resize Image
def resize_image(image, width, height):
    return image.resize((width, height))


# Thumbnail
def create_thumbnail(image, size=(300, 300)):
    thumbnail = image.copy()
    thumbnail.thumbnail(size)

    return thumbnail

# Image Dimensions
def get_dimensions(image):
    width, height = image.size

    return width, height

# Safe Copy
def copy_image(image):
    return image.copy()