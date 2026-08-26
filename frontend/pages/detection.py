import sys
import time

import streamlit as st
from PIL import Image
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.yolo_utils import (draw_yolo_boxes, get_yolo_predictions)
from utils.fasterrcnn_utils import (draw_fasterrcnn_boxes, get_fasterrcnn_predictions)


st.set_page_config(page_title="Detection | AgroWeedGuard", page_icon="🔍", layout="wide")

st.title("🔍 Weed Detection")
st.markdown(
            "Detect and localize weeds using YOLOv8 "
            "or Faster R-CNN.")
st.divider()

model_choice = st.radio("Choose Detection Model", ["YOLOv8", "Faster R-CNN"],  horizontal=True)

confidence_threshold = st.slider("Detection Confidence", min_value=0.10, max_value=0.95, value=0.50, step=0.05,
    help=(
            "Higher values show only more confident "
            "detections. Lower values may detect more "
            "weeds but can increase false positives."))


# Session State
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []


# Image Upload
uploaded_files = st.file_uploader("Upload Field Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


# Detections
if uploaded_files:
    st.subheader(f"📷 {len(uploaded_files)} Image(s) Selected")

    if st.button("🔍 Run Detection", use_container_width=True):      
        st.divider()

        st.header("🎯 Detection Results")

        # Image Processing
        for image_number, uploaded_file in enumerate(uploaded_files, start=1):
            image = Image.open(uploaded_file).convert("RGB")

            st.subheader(
                            f"Image {image_number}: "
                            f"{uploaded_file.name}")

            try:
                # Starting Inference Timer
                start_time = time.perf_counter()

                # YOLO Model
                if model_choice == "YOLOv8":
                    annotated_image = draw_yolo_boxes(image, confidence_threshold)
                    detections = get_yolo_predictions(image, confidence_threshold)

                # Faster R-CNN Model
                else:
                    annotated_image = (draw_fasterrcnn_boxes(image, confidence_threshold))
                    detections = (get_fasterrcnn_predictions(image, confidence_threshold))

                
                inference_time = (time.perf_counter() - start_time)

                # Recording Results For AI Assistant              
                detection_record = {
                                    "image_name": uploaded_file.name,
                                    "model": model_choice,
                                    "confidence_threshold": confidence_threshold,
                                    "inference_time": inference_time,
                                    "detections": [{"class_name": detection["class_name"], "confidence": detection["confidence"]}
                                                for detection in detections]}

                st.session_state.detection_results = [existing for existing in st.session_state.detection_results
                                                        if not (existing["image_name"] == uploaded_file.name and existing["model"] == model_choice)]

                    # Adding new Results
                st.session_state.detection_results.append(detection_record)


                # Image Results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.markdown("Detection Result")
                    st.image(annotated_image, use_container_width=True)

                # Statistics
                st.subheader("📊 Detection Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:                  
                    st.metric("Detections", len(detections))

                with col2:
                    st.metric("Model", model_choice)

                with col3:
                    st.metric("Inference Time", f"{inference_time * 1000:.2f} ms")

                # Detection Details
                st.subheader("🌿 Detected Weeds")

                if len(detections) == 0:
                    st.warning(
                                "No weeds detected above "
                                "the confidence threshold.")

                else:
                    for i, detection in enumerate(detections, start=1):
                        class_name = detection["class_name"]
                        confidence = detection["confidence"]

                        st.write(
                                    f"Detected -  "
                                    f"{class_name}")

                        st.progress(confidence, text=(
                                                        f"Confidence: "
                                                        f"{confidence * 100:.2f}%"))

            except Exception as e:
                st.error(
                            f"Detection Failed for "
                            f"{uploaded_file.name}: {e}")
                st.divider()


# Sidebar
with st.sidebar:
    st.subheader("🔍 Detection")

    st.caption("Locate weeds within field images using "
                "object detection models.")

    st.divider()
    st.subheader("🤖 Available Models")

    st.markdown("""
                    YOLOv8
                    - Fast object detection
                    - Suitable for real-time analysis

                    Faster R-CNN
                    - Region-based detection
                    - Detailed object localization
                    """)
    st.divider()

    st.subheader("📊 Model Performance")
    st.caption("YOLOv8")
    st.metric("Accuracy mAP50", "95.17%")

    st.caption("Faster R-CNN")
    st.metric("Accuracy mAP50", "79.53%")

    st.divider()
    
    if st.session_state.detection_results:
        st.subheader("🔬 Session")

        st.caption(f"{len(st.session_state.detection_results)} "
                    "model result(s) available for AI Assistant.")