#python -m streamlit run app.py

import os
import torch
import pickle
import re
import torch.nn.functional as F

from .model import EmotionRNN

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(__file__)

# constants
MAX_LEN = 75
EMBED_DIM = 100
HIDDEN_DIM = 128

# ---------- load saved assets ----------

with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(BASE_DIR, "label.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# rebuild model architecture
model = EmotionRNN(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(label_encoder.classes_)
).to(device)

# load weights
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "model_weights.pt"),
        map_location=device
    )
)

model.eval()

# ---------- helper functions ----------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def encode_text(text):
    tokens = text.split()
    ids = [vocab.get(t, 1) for t in tokens][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

# ---------- THIS IS WHAT app.py IMPORTS ----------

def predict_sentiment(sentence):
    sentence = clean_text(sentence)
    encoded = encode_text(sentence)

    tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0]

    idx = probs.argmax().item()
    label = label_encoder.classes_[idx]
    confidence = float(probs[idx])

    return label, confidence
