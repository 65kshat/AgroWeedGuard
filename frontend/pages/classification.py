import streamlit as st
import tempfile
import time

from PIL import Image

from utils.cnn_utils import predict_cnn
from utils.vit_utils import predict_vit

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🌿 Weed Classification")

st.markdown(
    "Classify weed species using CNN (ResNet18) or Vision Transformer (ViT)."
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

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

    if st.button("Run Classification"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)

            start_time = time.time()

            try:

                if model_choice == "CNN (ResNet18)":

                    result = predict_cnn(
                        temp_file.name
                    )

                else:

                    result = predict_vit(
                        temp_file.name
                    )

                inference_time = (
                    time.time() - start_time
                )

                with col2:

                    st.subheader("Prediction")

                    st.success(
                        f"{result['class_name']}"
                    )

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                st.divider()

                st.subheader("Classification Statistics")

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
                        f"{inference_time:.2f}s"
                    )

            except Exception as e:

                st.error(
                    f"Classification Failed:\n{e}"
                )