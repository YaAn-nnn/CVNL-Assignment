import streamlit as st
from PIL import Image
from cnn_vision_variants.infer import VisionPredictor
from rnn_sentiment.infer import predict_sentiment

# --------------------------------------------------
# page config
# --------------------------------------------------
st.set_page_config(
    page_title="Changi Airport Operation Assistant",
    layout="wide"
)

st.title("Changi Airport Operation Assistant (Integrated Demo)")

# --------------------------------------------------
# loads cnn model
# --------------------------------------------------
@st.cache_resource
def load_vision_model():
    return VisionPredictor("cnn_vision_variants/airline_model_best.pt")

vision_model = load_vision_model()

# --------------------------------------------------
# UI Sections
# --------------------------------------------------
tab1, tab2 = st.tabs(["Aircraft Identification (CNN)", "Passenger Sentiment (RNN)"])

# --- Jayden: CNN SECTION ---
with tab1:
    st.subheader("Upload Image → Aircraft Variant Identification")
    st.markdown("Identifies specific variants using your 448px ResNet-18 model.") #
    
    uploaded_file = st.file_uploader("Choose a tarmac photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict Aircraft Variant"):
            with st.spinner("Analyzing fine-grained features..."):
                # Call your specific prediction logic
                result = vision_model.predict(img)
                st.success(f"✈️ Identified Variant: **{result}**")

# --- Braden: SENTIMENT SECTION ---
with tab2:
    st.subheader("Passenger Message → Sentiment")
    sentiment_text = st.text_area(
        "Enter passenger feedback or message",
        height=160,
        placeholder="e.g. My flight was delayed and nobody helped me"
    )

    if st.button("Predict Sentiment"):
        if sentiment_text.strip() == "":
            st.warning("Please enter a message first.")
        else:
            label, confidence = predict_sentiment(sentiment_text)
            st.success(f"Predicted sentiment: {label} (confidence {confidence:.2f})")