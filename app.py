import streamlit as st
from PIL import Image
from cnn_vision_variants.infer import VisionPredictor
from rnn_sentiment.infer import predict_sentiment
from rnn_intent.infer import IntentPredictor
from cnn_family_classification.cnn_infer import predict_aircraft_family

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

@st.cache_resource
def load_intent_model():
    return IntentPredictor("rnn_intent/intent_deploy.pt")

vision_model = load_vision_model()
intent_model = load_intent_model()

# --------------------------------------------------
# UI Sections
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Aircraft Identification (CNN)",
    "Passenger Sentiment (RNN)",
    "Text intent (RNN)",
    "Aircraft Family Classification (CNN)"
])

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
                variant, confidence = vision_model.predict(img)
                
                st.success(f"✈️ Identified Variant: **{variant}**")
                st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%") #

                if confidence < 0.50:
                    st.warning("Low confidence: The model is seeing features common to multiple variants.")

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

with tab3:
    st.subheader("Passenger/Staff Message → Intent")
    text = st.text_area("Type a message", height=120)

    if st.button("Predict Intent"):
        if text.strip() == "":
            st.warning("Please enter a message first.")
        else:
            st.success(f"Predicted intent: {intent_model.predict(text)}")

# --- Yoshi: CNN FAMILY CLASSIFICATION ---
with tab4:
    st.subheader("Upload Image → Aircraft Family Classification")
    st.markdown("Classifies aircraft images into family categories using a custom CNN.")

    uploaded_file = st.file_uploader(
        "Upload an aircraft image",
        type=["jpg", "jpeg", "png"],
        key="family_upload"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Aircraft Family"):
            with st.spinner("Running CNN inference..."):
                label, confidence = predict_aircraft_family(img)

            st.success(f"Predicted Aircraft Family: **{label}**")
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            if confidence < 0.5:
                st.warning("Low confidence prediction. Image may contain overlapping features.")
