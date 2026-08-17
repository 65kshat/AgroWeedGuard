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

uploaded_file = st.file_uploader(
    "Upload Weed Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Selected Model")

        st.info(model_choice)

    st.divider()

    if st.button(
        "🔍 Run Classification",
        use_container_width=True
    ):

        start_time = time.perf_counter()

        try:

            # --------------------------------------
            # CNN
            # --------------------------------------

            if model_choice == "CNN (ResNet18)":

                result = predict_cnn(
                    image
                )

            # --------------------------------------
            # ViT
            # --------------------------------------

            else:

                result = predict_vit(
                    image
                )

            inference_time = (
                time.perf_counter()
                - start_time
            )

            # --------------------------------------
            # RESULT
            # --------------------------------------

            st.divider()

            st.header("🎯 Classification Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.success(
                    f"Prediction: "
                    f"{result['class_name']}"
                )

            with result_col2:

                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.2f}%"
                )

            # --------------------------------------
            # STATISTICS
            # --------------------------------------

            st.subheader(
                "📊 Classification Statistics"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Class ID",
                    result["class_id"]
                )

            with c2:

                st.metric(
                    "Model",
                    model_choice
                )

            with c3:

                st.metric(
                    "Inference Time",
                    f"{inference_time * 1000:.2f} ms"
                )

        except Exception as e:

            st.error(
                f"Classification Failed: {e}"
            )