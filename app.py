import streamlit as st
from rnn_sentiment.infer import predict_sentiment

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Changi Airport Operation Assistant",
    layout="wide"
)

st.title("Changi Airport Operation Assistant (Sentiment Demo)")

st.markdown(
    """
    This demo showcases an RNN-based sentiment analysis model trained on airline
    passenger feedback. The model predicts whether a message expresses
    **positive**, **neutral**, or **negative** sentiment.
    """
)

# --------------------------------------------------
# Sentiment Analysis Section
# --------------------------------------------------
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
        st.success(
            f"Predicted sentiment: {label} (confidence {confidence:.2f})"
        )
