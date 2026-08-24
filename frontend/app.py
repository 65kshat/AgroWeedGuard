from pathlib import Path
import streamlit as st
import base64

CURRENT_DIR = Path(__file__).parent

ASSETS_DIR = CURRENT_DIR / "assets"
KINETIC_GRID_PATH = CURRENT_DIR / "kinetic_grid.html"
CSS_PATH = CURRENT_DIR / "style.css"

BACKGROUND_PATH = ASSETS_DIR / "background_image.png"
SIDEBAR_PATH = ASSETS_DIR / "sidebar.jpg"



# PAGE CONFIG
st.set_page_config(page_title="AgroWeedGuard", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")


# Loading CSS 
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()

    # Main Background
    if BACKGROUND_PATH.exists():
        with open(BACKGROUND_PATH, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        background_url = (f"url(data:image/jpeg;base64,{encoded_image})")
        css = css.replace("AGRO_BACKGROUND_IMAGE", background_url)

    
    # Sidebar Background
    if SIDEBAR_PATH.exists():
        with open(SIDEBAR_PATH, "rb") as image_file:
            encoded_sidebar = base64.b64encode(image_file.read()).decode()

        sidebar_url = (f"url(data:image/jpeg;base64,{encoded_sidebar})")

        css = css.replace("AGRO_SIDEBAR_IMAGE", sidebar_url)

    # Applying CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# KINETIC GRID BACKGROUND
if KINETIC_GRID_PATH.exists():
    st.html(KINETIC_GRID_PATH, unsafe_allow_javascript=True)

# NAVIGATION

pages = [
            st.Page("pages/home.py", title="Home", icon="🏠"),
            st.Page("pages/detection.py", title="Detection", icon="🔍"),
            st.Page("pages/classification.py", title="Classification", icon="🌿"),
            st.Page("pages/model_comparison.py", title="Model Comparison", icon="📊"),
            st.Page("pages/chatbot.py", title="AI Assistant", icon="🤖")]


# Run Aplication
navigation = st.navigation(pages)
navigation.run()