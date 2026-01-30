import torch
import torch.nn as nn

class EmotionRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        embedding_matrix=None
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # If pretrained embeddings (e.g. GloVe) are provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float)
            )
            self.embedding.weight.requires_grad = True

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.lstm(x)
        pooled = torch.mean(outputs, dim=1)
        return self.fc(self.dropout(pooled))
