# utils/image_utils.py

import numpy as np
from PIL import Image

# Class Labels
CLASS_NAMES = {
    0: "Weed: Mollugo Verticillata",
    1: "Weed: Amaranthus Palmeri",
    2: "Weed: Eclipta",
    3: "Weed: Portulaca Oleracea",
    4: "Weed: Amaranthus Tuberculatus",
    5: "Weed: Euphorbia Maculata",
    6: "Weed: Ipomoea Indica",
    7: "Weed: Eleusine Indica",
    8: "Weed: Sida Rhombifolia",
    9: "Weed: Senna Obtusifolia",
    10: "Weed: Physalis Angulata",
    11: "Weed: Ambrosia Artemisiifolia"}


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