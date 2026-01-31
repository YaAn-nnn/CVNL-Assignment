import streamlit as st
from PIL import Image

from cnn_vision_variants.infer import VisionPredictor
from rnn_sentiment.infer import predict_sentiment

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Changi Airport Operation Assistant",
    layout="wide"
)

st.title("Changi Airport Operation Assistant (Integrated Demo)")

# --------------------------------------------------
# Load models (ONLY models that actually exist)
# --------------------------------------------------
@st.cache_resource
def load_models():
    vision = VisionPredictor("cnn_vision/model.pt")
    return vision

vision_model = load_models()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs([
    "Image Classifier (CNN)",
    "Text Sentiment (RNN)"
])

# --------------------------------------------------
# TAB 1: CNN Image Classification
# --------------------------------------------------
with tab1:
    st.subheader("Upload Image → Visual Class")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Predict Image"):
            pred = vision_model.predict(img)
            st.success(f"Predicted class: {pred}")

# --------------------------------------------------
# TAB 2: RNN Sentiment Classification
# --------------------------------------------------
with tab2:
    st.subheader("Passenger Message → Sentiment")

    sentiment_text = st.text_area(
        "Enter passenger feedback or message",
        height=140
    )

    if st.button("Predict Sentiment"):
        label, confidence = predict_sentiment(sentiment_text)
        st.success(
            f"Predicted sentiment: {label} (confidence {confidence:.2f})"
        )
