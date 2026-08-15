import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AgroWeedGuard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

CURRENT_DIR = Path(__file__).parent

ASSETS_DIR = CURRENT_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "logo.png"

CSS_PATH = CURRENT_DIR / "styles.css"

# --------------------------------------------------
# LOAD CSS
# --------------------------------------------------

if CSS_PATH.exists():

    with open(CSS_PATH) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            use_container_width=True
        )

    st.title("🌱 AgroWeedGuard")

    st.markdown("---")

    st.markdown("""
### Navigation

Use the pages panel above to access:

- 🏠 Home
- 🔍 Detection
- 🌿 Classification
- 📊 Model Comparison
- ℹ️ About Project
""")

    st.markdown("---")

    st.markdown("""
### Models

✓ YOLOv8

✓ Faster R-CNN

✓ CNN (ResNet18)

✓ Vision Transformer
""")

# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.title("🌱 AgroWeedGuard")

st.subheader(
    "AI-Powered Weed Detection and Classification System"
)

st.markdown("---")

st.markdown("""
Welcome to AgroWeedGuard.

Use the navigation menu on the left to explore
the application's features.

### Available Modules

🔍 Detection
- YOLOv8
- Faster R-CNN

🌿 Classification
- CNN (ResNet18)
- Vision Transformer

📊 Model Comparison
- Compare all AI models on the same image

ℹ️ About Project
- Project details
- Dataset information
- System architecture
""")

st.info(
    "Select a page from the sidebar to begin."
)