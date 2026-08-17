from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
import time

from PIL import Image

from utils.yolo_utils import (
    draw_yolo_boxes,
    get_yolo_predictions
)

from utils.fasterrcnn_utils import (
    draw_fasterrcnn_boxes,
    get_fasterrcnn_predictions
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Detection | AgroWeedGuard",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🔍 Weed Detection")

st.markdown(
    "Detect and localize weeds using YOLOv8 "
    "or Faster R-CNN."
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model_choice = st.radio(
    "Choose Detection Model",
    [
        "YOLOv8",
        "Faster R-CNN"
    ],
    horizontal=True
)


# --------------------------------------------------
# CONFIDENCE THRESHOLD
# --------------------------------------------------

confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Field Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Run Detection",
        use_container_width=True
    ):

        start_time = time.perf_counter()

        try:

            # ======================================
            # YOLO
            # ======================================

            if model_choice == "YOLOv8":

                # YOLO uses its own confidence
                # threshold internally.
                #
                # Current utility does not expose
                # threshold as an argument, so the
                # slider is not applied here yet.

                annotated_image = draw_yolo_boxes(
                    image
                )

                detections = get_yolo_predictions(
                    image
                )

            # ======================================
            # FASTER R-CNN
            # ======================================

            else:

                annotated_image = (
                    draw_fasterrcnn_boxes(
                        image,
                        confidence_threshold
                    )
                )

                detections = (
                    get_fasterrcnn_predictions(
                        image,
                        confidence_threshold
                    )
                )

            inference_time = (
                time.perf_counter()
                - start_time
            )

            # ======================================
            # RESULT IMAGE
            # ======================================

            st.header("🎯 Detection Result")

            st.image(
                annotated_image,
                use_container_width=True
            )

            # ======================================
            # STATISTICS
            # ======================================

            st.divider()

            st.subheader(
                "📊 Detection Statistics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Detections",
                    len(detections)
                )

            with col2:

                st.metric(
                    "Model",
                    model_choice
                )

            with col3:

                st.metric(
                    "Inference Time",
                    f"{inference_time * 1000:.2f} ms"
                )

            # ======================================
            # DETECTION DETAILS
            # ======================================

            st.subheader(
                "🌿 Detected Weeds"
            )

            if len(detections) == 0:

                st.warning(
                    "No weeds detected above "
                    "the confidence threshold."
                )

            else:

                for i, detection in enumerate(
                    detections,
                    start=1
                ):

                    class_name = detection[
                        "class_name"
                    ]

                    confidence = detection[
                        "confidence"
                    ]

                    st.write(
                        f"**Detection {i}:** "
                        f"{class_name}"
                    )

                    st.progress(
                        confidence,
                        text=(
                            f"Confidence: "
                            f"{confidence * 100:.2f}%"
                        )
                    )

        except Exception as e:

            st.error(
                f"Detection Failed: {e}"
            )