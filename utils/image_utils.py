# utils/image_utils.py

import numpy as np

from PIL import Image

# --------------------------------------------------
# Open Uploaded Image
# --------------------------------------------------

def load_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    return image


# --------------------------------------------------
# PIL → NumPy
# --------------------------------------------------

def pil_to_numpy(image):

    return np.array(image)


# --------------------------------------------------
# NumPy → PIL
# --------------------------------------------------

def numpy_to_pil(image):

    return Image.fromarray(
        image.astype(np.uint8)
    )


# --------------------------------------------------
# Get Image Information
# --------------------------------------------------

def get_image_info(image):

    width, height = image.size

    return {
        "width": width,
        "height": height,
        "mode": image.mode
    }


# --------------------------------------------------
# Resize Image
# --------------------------------------------------

def resize_image(
    image,
    width,
    height
):

    return image.resize(
        (width, height)
    )


# --------------------------------------------------
# Thumbnail
# --------------------------------------------------

def create_thumbnail(
    image,
    size=(300, 300)
):

    thumbnail = image.copy()

    thumbnail.thumbnail(size)

    return thumbnail


# --------------------------------------------------
# Image Dimensions
# --------------------------------------------------

def get_dimensions(image):

    width, height = image.size

    return width, height


# --------------------------------------------------
# Safe Copy
# --------------------------------------------------

def copy_image(image):

    return image.copy()