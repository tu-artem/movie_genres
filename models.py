import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        n_out: int,
        vocab_size: int,
        vectors: torch.Tensor,
        seq_len: int = 100,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.embeddings = nn.Embedding.from_pretrained(vectors)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, n_out)

    def _lstm_forward(self, data, lengths):

        embeddings = self.embeddings(data)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        lstm, (h, c) = self.lstm(packed)
        lstm_unpacked = pad_packed_sequence(
            lstm, batch_first=True, total_length=self.seq_len
        )[0]

        return lstm_unpacked, (h, c)

    def forward(self, data, lengths):

        lstm_unpacked, (h, c) = self._lstm_forward(data, lengths)

        # Like output, the layers can be separated
        # using h_n.view(num_layers, num_directions, batch, hidden_size)
        h = h.view(self.num_layers, self.num_directions, -1, self.hidden_dim)

        # TODO: Check dimensions for multi-layer!
        if self.bidirectional:
            last_states = torch.cat([h[-1, 0], h[-1, 1]], dim=1)
        else:
            last_states = h[-1, 0]

        linear = self.fc(self.dropout(last_states))
        linear2 = self.fc2(self.dropout(linear))
        return linear2
