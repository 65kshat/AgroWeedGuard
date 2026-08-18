from pathlib import Path
import streamlit as st


# --------------------------------------------------
# PATHS
# --------------------------------------------------

CURRENT_DIR = Path(__file__).parent
CSS_PATH = CURRENT_DIR / "style.css"


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
# LOAD GLOBAL CSS
# --------------------------------------------------

if CSS_PATH.exists():

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

pages = [

    st.Page(
        "pages/home.py",
        title="Home",
        icon="🏠"
    ),

    st.Page(
        "pages/detection.py",
        title="Detection",
        icon="🔍"
    ),

    st.Page(
        "pages/classification.py",
        title="Classification",
        icon="🌿"
    ),

    st.Page(
        "pages/model_comparison.py",
        title="Model Comparison",
        icon="📊"
    )

]


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

navigation = st.navigation(pages)

navigation.run()