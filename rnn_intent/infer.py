import re
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from .model import LSTMIntent

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []

class IntentPredictor:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")  # deploy checkpoint has no custom classes

        self.stoi = ckpt["stoi"]
        self.itos = ckpt["itos"]
        self.label2id = ckpt["label2id"]
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.model = LSTMIntent(
            vocab_size=len(self.itos),
            num_classes=len(self.label2id),
            emb_dim=128,
            hidden_dim=128,
            dropout=0.2
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def encode(self, text: str):
        ids = []
        for w in tokenize(text):
            ids.append(self.stoi.get(w, self.stoi.get(UNK_TOKEN, 1)))
        return ids

    def predict(self, text: str) -> str:
        x = self.encode(text)
        if len(x) == 0:
            return "UNKNOWN"

        x_t = torch.tensor(x, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([len(x)], dtype=torch.long)
        x_packed = pack_padded_sequence(x_t, lengths, batch_first=True, enforce_sorted=False)

        with torch.no_grad():
            logits = self.model(x_packed)
            pred_id = int(torch.argmax(logits, dim=1).item())

        return self.id2label[pred_id]