import streamlit as st
from prediction.model_loader import load_all_models
from prediction.ensemble import ensemble_predict
from prediction.config import CLASS_NAMES
from PIL import Image

# =====================================================
# LOAD CSS
# =====================================================

def load_css():

    with open("assets/style.css") as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True

        )
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Chest X-ray Disease Detection",
    page_icon="🩺",
    layout="wide"
)
load_css()
# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():
    return load_all_models()

models = load_models()

# =====================================================
# TITLE
# =====================================================

st.markdown("""
<div class="main-title">

🩺 AI-Based Chest X-ray Disease Detection

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">

Ensemble Learning using EfficientNet-B0, DenseNet121 & ResNet50

</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Read Uploaded Image (No temporary file)
    image = Image.open(uploaded_file).convert("RGB")

    # Display Image
    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    # Prediction
    result = ensemble_predict(models, image)

    with col2:

        agreement = result["agreement"]

        if agreement == "3/3":
            st.success("🟢 All 3 AI Doctors agree")
        elif agreement == "2/3":
            st.warning("🟡 2 of 3 AI Doctors agree")
        else:
            st.error("🔴 AI Doctors disagree")

        with st.container(border=True):

            st.subheader("🏥 Final AI Diagnosis")

            st.metric(
                "Predicted Disease",
                result["final_prediction"]
            )

            st.metric(
                "Confidence",
                f"{result['final_confidence']:.2f}%"
            )

    st.divider()

    # =====================================================
    # AI DOCTORS
    # =====================================================

    st.subheader("🩺 AI Doctors")

    col1, col2, col3 = st.columns(3)

    doctors = [

        ("🩺 AI Doctor 1", "EfficientNet-B0", result["efficientnet"]),

        ("🩺 AI Doctor 2", "DenseNet121", result["densenet"]),

        ("🩺 AI Doctor 3", "ResNet50", result["resnet"])

    ]

    for column, doctor in zip([col1, col2, col3], doctors):

        title, model_name, pred = doctor

        with column:

            with st.container(border=True):

                st.markdown(f"#### {title}")

                st.caption(model_name)

                st.success(f"**{pred['prediction']}**")

                st.metric(
                    label="Confidence",
                    value=f"{pred['confidence']:.2f}%"
                )

                probs = pred["probabilities"]

                for i, disease in enumerate(CLASS_NAMES):

                    st.progress(float(probs[i]), text=f"{disease} ({probs[i]*100:.1f}%)")