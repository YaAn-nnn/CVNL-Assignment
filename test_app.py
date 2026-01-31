import streamlit as st
from PIL import Image
# Only import your CNN logic
from cnn_vision_variants.infer import VisionPredictor 

st.set_page_config(page_title="CNN Test Only", layout="wide")
st.title("Independent CNN Aircraft Classifier Test")

# Only load your model
@st.cache_resource
def load_vision_only():
    # Points to your specific folder and model name
    return VisionPredictor("cnn_vision_variants/airline_model_best.pt")

vision_model = load_vision_only()

st.subheader("Upload Aircraft Image")
uploaded = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)
    if st.button("Predict Aircraft Type"):
        # This calls your 79.48% accuracy model
        result = vision_model.predict(img)
        st.success(f"✈️ Identified Variant: {result}")