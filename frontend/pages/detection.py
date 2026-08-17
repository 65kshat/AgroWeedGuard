from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import tempfile
import time

from PIL import Image

from utils.yolo_utils import predict_yolo
from utils.fasterrcnn_utils import predict_fasterrcnn

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🔍 Weed Detection")

st.markdown(
    "Detect weeds using YOLOv8 or Faster R-CNN."
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
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Field Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    if st.button("Run Detection"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)

            start_time = time.time()

            try:

                if model_choice == "YOLOv8":

                    result = predict_yolo(
                        temp_file.name
                    )

                else:

                    result = predict_fasterrcnn(
                        temp_file.name
                    )

                inference_time = (
                    time.time() - start_time
                )

                with col2:

                    st.subheader("Detection Result")

                    st.image(
                        result["image"],
                        use_container_width=True
                    )

                st.divider()

                st.subheader("Detection Statistics")

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric(
                        "Detections",
                        result["count"]
                    )

                with col_b:
                    st.metric(
                        "Model",
                        model_choice
                    )

                with col_c:
                    st.metric(
                        "Time (sec)",
                        f"{inference_time:.2f}"
                    )

            except Exception as e:

                st.error(
                    f"Detection Failed:\n{e}"
                )