import torch.nn as nn

class LSTMIntent(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_packed):
        emb = self.embedding(x_packed.data)
        packed = nn.utils.rnn.PackedSequence(
            emb, x_packed.batch_sizes, x_packed.sorted_indices, x_packed.unsorted_indices
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1]))