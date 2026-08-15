import streamlit as st
import tempfile
import time

from PIL import Image

from utils.yolo_utils import predict_yolo
from utils.fasterrcnn_utils import predict_fasterrcnn
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

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Run All Models"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)

            try:

                # ----------------------------------
                # YOLO
                # ----------------------------------

                start = time.time()

                yolo_result = predict_yolo(
                    temp_file.name
                )

                yolo_time = time.time() - start

                # ----------------------------------
                # FasterRCNN
                # ----------------------------------

                start = time.time()

                frcnn_result = predict_fasterrcnn(
                    temp_file.name
                )

                frcnn_time = time.time() - start

                # ----------------------------------
                # CNN
                # ----------------------------------

                start = time.time()

                cnn_result = predict_cnn(
                    temp_file.name
                )

                cnn_time = time.time() - start

                # ----------------------------------
                # ViT
                # ----------------------------------

                start = time.time()

                vit_result = predict_vit(
                    temp_file.name
                )

                vit_time = time.time() - start

                st.divider()

                st.header("Model Results")

                comparison_data = [
                    {
                        "Model": "YOLOv8",
                        "Output": f"{yolo_result['count']} Detection(s)",
                        "Confidence": "-",
                        "Time (s)": round(yolo_time, 3)
                    },

                    {
                        "Model": "Faster R-CNN",
                        "Output": f"{frcnn_result['count']} Detection(s)",
                        "Confidence": "-",
                        "Time (s)": round(frcnn_time, 3)
                    },

                    {
                        "Model": "CNN",
                        "Output": cnn_result["class_name"],
                        "Confidence": round(
                            cnn_result["confidence"] * 100,
                            2
                        ),
                        "Time (s)": round(cnn_time, 3)
                    },

                    {
                        "Model": "ViT",
                        "Output": vit_result["class_name"],
                        "Confidence": round(
                            vit_result["confidence"] * 100,
                            2
                        ),
                        "Time (s)": round(vit_time, 3)
                    }
                ]

                st.dataframe(
                    comparison_data,
                    use_container_width=True
                )

                st.divider()

                st.header("Detection Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("YOLOv8")

                    st.image(
                        yolo_result["image"],
                        use_container_width=True
                    )

                with col2:

                    st.subheader("Faster R-CNN")

                    st.image(
                        frcnn_result["image"],
                        use_container_width=True
                    )

                st.divider()

                st.header("Classification Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.info(
                        f"""
                        CNN Prediction

                        Class: {cnn_result['class_name']}

                        Confidence:
                        {cnn_result['confidence']*100:.2f}%
                        """
                    )

                with col2:

                    st.info(
                        f"""
                        ViT Prediction

                        Class: {vit_result['class_name']}

                        Confidence:
                        {vit_result['confidence']*100:.2f}%
                        """
                    )

                st.divider()

                st.header("Performance Summary")

                fastest = min(
                    {
                        "YOLO": yolo_time,
                        "FasterRCNN": frcnn_time,
                        "CNN": cnn_time,
                        "ViT": vit_time
                    },
                    key=lambda x: {
                        "YOLO": yolo_time,
                        "FasterRCNN": frcnn_time,
                        "CNN": cnn_time,
                        "ViT": vit_time
                    }[x]
                )

                st.success(
                    f"⚡ Fastest Model: {fastest}"
                )

            except Exception as e:

                st.error(
                    f"Comparison Failed:\n{e}"
                )