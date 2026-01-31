import streamlit as st
from PIL import Image

from rnn_intent.infer import IntentPredictor
from cnn_vision_variants.infer import VisionPredictor
from rnn_sentiment.infer import predict_sentiment

st.set_page_config(page_title="Changi Airport Operation Assistant", layout="wide")
st.title("Changi Airport Operation Assistant (Integrated Demo)")

@st.cache_resource
def load_models():
    intent = IntentPredictor("models/intent_deploy.pt")
    vision = VisionPredictor("models/vision_best.pt")
    return intent, vision

intent_model, vision_model = load_models()

tab1, tab2, tab3 = st.tabs(["Text Intent", "Image Classifier", "Text Sentiment"])

with tab1:
    st.subheader("Passenger/Staff Message → Intent")
    text = st.text_area("Type a message", height=120)
    if st.button("Predict Intent"):
        st.success(f"Predicted intent: {intent_model.predict(text)}")

with tab2:
    st.subheader("Upload Image → Visual Class")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
        if st.button("Predict Image"):
            st.success(f"Predicted class: {vision_model.predict(img)}")

with tab3:
    st.subheader("Passenger Message → Sentiment")

    sentiment_text = st.text_area(
        "Enter passenger feedback or message",
        height=120,
        key="sentiment_input"
    )

    if st.button("Predict Sentiment"):
        label, confidence = predict_sentiment(sentiment_text)
        st.success(f"Predicted sentiment: {label} (confidence {confidence:.2f})")
