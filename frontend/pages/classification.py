from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import time

from PIL import Image

from utils.cnn_utils import predict_cnn
from utils.vit_utils import predict_vit


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Classification | AgroWeedGuard",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🌿 Weed Classification")

st.markdown(
    "Classify weed species using CNN (ResNet18) "
    "or Vision Transformer (ViT)."
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model_choice = st.radio(
    "Choose Classification Model",
    [
        "CNN (ResNet18)",
        "Vision Transformer (ViT)"
    ],
    horizontal=True
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Weed Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if uploaded_files:

    st.subheader(
        f"📷 {len(uploaded_files)} Image(s) Selected"
    )

    if st.button(
        "🔍 Run Classification",
        use_container_width=True
    ):

        st.divider()

        st.header("🎯 Classification Results")

        # ------------------------------------------
        # PROCESS EACH IMAGE
        # ------------------------------------------

        for image_number, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.subheader(
                f"Image {image_number}: "
                f"{uploaded_file.name}"
            )

            col1, col2 = st.columns(2)

            # --------------------------------------
            # IMAGE
            # --------------------------------------

            with col1:

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            # --------------------------------------
            # MODEL INFERENCE
            # --------------------------------------

            with col2:

                start_time = time.perf_counter()

                try:

                    if model_choice == "CNN (ResNet18)":

                        result = predict_cnn(
                            image
                        )

                    else:

                        result = predict_vit(
                            image
                        )

                    inference_time = (
                        time.perf_counter()
                        - start_time
                    )

                    # ----------------------------------
                    # PRIMARY RESULT
                    # ----------------------------------

                    st.success(
                        f"🌿 {result['class_name']}"
                    )

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                    # ----------------------------------
                    # TECHNICAL DETAILS
                    # ----------------------------------

                    st.caption(
                        f"Class ID: {result['class_id']}"
                    )

                    st.caption(
                        f"Model: {model_choice}"
                    )

                    st.caption(
                        f"Inference Time: "
                        f"{inference_time * 1000:.2f} ms"
                    )

                except Exception as e:

                    st.error(
                        f"Classification Failed: {e}"
                    )

            st.divider()