from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
import time

from PIL import Image

from utils.yolo_utils import run_yolo
from utils.fasterrcnn_utils import run_fasterrcnn

from utils.cnn_utils import predict_cnn
from utils.vit_utils import predict_vit

from utils.cnn_utils import predict_cnn
from utils.vit_utils import predict_vit


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("📊 Model Comparison")

st.markdown(
    """
Compare the predictions and performance of all
AgroWeedGuard models on a single image.
"""
)

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# COMPARISON
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
        "Run All Models",
        use_container_width=True
    ):

        try:

            # --------------------------------------
            # YOLO
            # --------------------------------------

            start = time.perf_counter()

            yolo_result = run_yolo(
                image
            )

            yolo_time = (
                time.perf_counter() - start
            )

            yolo_detections = yolo_result["detections"]
            yolo_image = yolo_result["image"]

            # --------------------------------------
            # Faster R-CNN
            # --------------------------------------

            start = time.perf_counter()

            frcnn_result = run_fasterrcnn(
                image
            )

            frcnn_time = (
                time.perf_counter() - start
            )

            frcnn_detections = frcnn_result["detections"]
            frcnn_image = frcnn_result["image"]


            # ======================================
            # CNN
            # ======================================

            start = time.perf_counter()

            cnn_result = predict_cnn(
                image
            )

            cnn_time = (
                time.perf_counter() - start
            )


            # ======================================
            # ViT
            # ======================================

            start = time.perf_counter()

            vit_result = predict_vit(
                image
            )

            vit_time = (
                time.perf_counter() - start
            )


            # ======================================
            # MODEL RESULTS
            # ======================================

            st.divider()

            st.header("📊 Model Results")


            # --------------------------------------
            # Detection summaries
            # --------------------------------------

            yolo_count = len(
                yolo_detections
            )

            frcnn_count = len(
                frcnn_detections
            )


            # Get highest-confidence detection

            yolo_best = (
                max(
                    yolo_detections,
                    key=lambda x: x["confidence"]
                )
                if yolo_detections
                else None
            )

            frcnn_best = (
                max(
                    frcnn_detections,
                    key=lambda x: x["confidence"]
                )
                if frcnn_detections
                else None
            )


            comparison_data = [

                {
                    "Model": "YOLOv8",
                    "Type": "Detection",
                    "Output": (
                        yolo_best["class_name"]
                        if yolo_best
                        else "No Detection"
                    ),
                    "Detections": yolo_count,
                    "Confidence": (
                        f"{yolo_best['confidence'] * 100:.2f}%"
                        if yolo_best
                        else "-"
                    ),
                    "Time (s)": round(
                        yolo_time,
                        3
                    )
                },

                {
                    "Model": "Faster R-CNN",
                    "Type": "Detection",
                    "Output": (
                        frcnn_best["class_name"]
                        if frcnn_best
                        else "No Detection"
                    ),
                    "Detections": frcnn_count,
                    "Confidence": (
                        f"{frcnn_best['confidence'] * 100:.2f}%"
                        if frcnn_best
                        else "-"
                    ),
                    "Time (s)": round(
                        frcnn_time,
                        3
                    )
                },

                {
                    "Model": "CNN (ResNet18)",
                    "Type": "Classification",
                    "Output": cnn_result[
                        "class_name"
                    ],
                    "Detections": "-",
                    "Confidence": (
                        f"{cnn_result['confidence'] * 100:.2f}%"
                    ),
                    "Time (s)": round(
                        cnn_time,
                        3
                    )
                },

                {
                    "Model": "Vision Transformer",
                    "Type": "Classification",
                    "Output": vit_result[
                        "class_name"
                    ],
                    "Detections": "-",
                    "Confidence": (
                        f"{vit_result['confidence'] * 100:.2f}%"
                    ),
                    "Time (s)": round(
                        vit_time,
                        3
                    )
                }
            ]


            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )


            # ======================================
            # DETECTION RESULTS
            # ======================================

            st.divider()

            st.header("🔍 Detection Results")

            col1, col2 = st.columns(2)


            with col1:

                st.subheader("YOLOv8")

                st.image(
                    yolo_image,
                    use_container_width=True
                )

                if yolo_detections:

                    for i, detection in enumerate(
                        yolo_detections,
                        start=1
                    ):

                        st.write(
                            f"**{i}. "
                            f"{detection['class_name']}**"
                        )

                        st.progress(
                            detection["confidence"],
                            text=(
                                f"Confidence: "
                                f"{detection['confidence'] * 100:.2f}%"
                            )
                        )

                else:

                    st.info(
                        "No weeds detected."
                    )


            with col2:

                st.subheader("Faster R-CNN")

                st.image(
                    frcnn_image,
                    use_container_width=True
                )

                if frcnn_detections:

                    for i, detection in enumerate(
                        frcnn_detections,
                        start=1
                    ):

                        st.write(
                            f"**{i}. "
                            f"{detection['class_name']}**"
                        )

                        st.progress(
                            detection["confidence"],
                            text=(
                                f"Confidence: "
                                f"{detection['confidence'] * 100:.2f}%"
                            )
                        )

                else:

                    st.info(
                        "No weeds detected."
                    )


            # ======================================
            # CLASSIFICATION RESULTS
            # ======================================

            st.divider()

            st.header("🌿 Classification Results")

            col1, col2 = st.columns(2)


            with col1:

                st.subheader("CNN (ResNet18)")

                st.success(
                    f"Prediction: "
                    f"{cnn_result['class_name']}"
                )

                st.metric(
                    "Confidence",
                    f"{cnn_result['confidence'] * 100:.2f}%"
                )


            with col2:

                st.subheader("Vision Transformer")

                st.success(
                    f"Prediction: "
                    f"{vit_result['class_name']}"
                )

                st.metric(
                    "Confidence",
                    f"{vit_result['confidence'] * 100:.2f}%"
                )


            # ======================================
            # PERFORMANCE SUMMARY
            # ======================================

            st.divider()

            st.header("⚡ Performance Summary")


            times = {
                "YOLOv8": yolo_time,
                "Faster R-CNN": frcnn_time,
                "CNN (ResNet18)": cnn_time,
                "Vision Transformer": vit_time
            }


            fastest = min(
                times,
                key=times.get
            )


            st.success(
                f"⚡ Fastest Model: "
                f"**{fastest}** "
                f"({times[fastest] * 1000:.2f} ms)"
            )


        except Exception as e:

            st.error(
                f"Comparison Failed:\n{e}"
            )