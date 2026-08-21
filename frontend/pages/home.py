# 1_Home.py
import base64
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AgroWeedGuard",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# ASSETS
# --------------------------------------------------

CURRENT_DIR = Path(__file__).parent
FRONTEND_DIR = CURRENT_DIR.parent

LOGO_PATH = FRONTEND_DIR / "assets" / "ascii-image.png"
BANNER_PATH = FRONTEND_DIR / "assets" / "background.jpg"

# --------------------------------------------------
# HEADER
# --------------------------------------------------

if LOGO_PATH.exists():

    with open(LOGO_PATH, "rb") as logo_file:

        logo_base64 = base64.b64encode(
            logo_file.read()
        ).decode()

    st.html(
        f"""
        <div class="agro-header">

            <img
                src="data:image/png;base64,{logo_base64}"
                class="agro-logo"
            >

            <div class="agro-header-text">

                <h1>AgroWeedGuard</h1>

                <div class="agro-subtitle">
                    AI-Powered Weed Detection and Classification System
                </div>

            </div>

        </div>
        """
    )

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

if BANNER_PATH.exists():
    st.image(
        str(BANNER_PATH),
        use_container_width=True
    )

st.markdown("""
### Welcome to AgroWeedGuard

AgroWeedGuard is an intelligent agricultural monitoring platform
designed to detect and classify weeds using Machine Learning and
Deep Learning models.

The system combines object detection and image classification
techniques to help farmers, researchers, and agricultural
professionals identify unwanted plant species quickly and accurately.
""")

st.divider()

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.header("📌 Project Overview")

st.markdown("""
AgroWeedGuard integrates multiple Artificial Intelligence models
to provide robust weed identification capabilities.

The platform currently supports:

- Weed Detection using YOLOv8
- Weed Detection using Faster R-CNN
- Weed Classification using CNN (ResNet18)
- Weed Classification using Vision Transformer (ViT)
- Model Performance Comparison
- Future AI Assistant Integration
""")

# --------------------------------------------------
# MODEL CARDS
# --------------------------------------------------

st.header("🤖 Models Included")

col1, col2 = st.columns(2)

with col1:

    st.info("""
### YOLOv8

Real-time object detection model capable of
identifying weeds with bounding boxes and
high-speed inference.
""")

    st.info("""
### CNN (ResNet18)

Convolutional Neural Network trained for
weed species classification across multiple
categories.
""")

with col2:

    st.info("""
### Faster R-CNN

Region Proposal based detector providing
high-quality object localization and
detection accuracy.
""")

    st.info("""
### Vision Transformer (ViT)

Transformer-based image classification model
capable of capturing global image features.
""")

# --------------------------------------------------
# PROJECT STATISTICS
# --------------------------------------------------

st.header("📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Detection Models",
        value="2"
    )

with col2:
    st.metric(
        label="Classification Models",
        value="2"
    )

with col3:
    st.metric(
        label="Weed Classes",
        value="12"
    )

with col4:
    st.metric(
        label="AI Models Total",
        value="4"
    )

# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------

st.header("⚙️ System Workflow")

st.markdown("""
1. Upload an agricultural field image.
2. Run Weed Detection (YOLOv8 or Faster R-CNN).
3. Run Weed Classification (CNN or ViT).
4. Compare model predictions.
5. Analyze results and confidence scores.
6. Generate insights for agricultural decision-making.
""")

# --------------------------------------------------
# FUTURE ENHANCEMENTS
# --------------------------------------------------

st.header("🚀 Future Enhancements")

st.markdown("""
- AI Chat Assistant
- Weed Treatment Recommendations
- PDF Report Generation
- Real-Time Camera Detection
- Mobile Application Support
- Cloud Deployment
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AgroWeedGuard Capstone Project | AI-Powered Weed Detection and Classification"
)