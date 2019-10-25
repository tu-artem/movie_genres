from typing import List

import torch
from torch import nn
import torch.nn.functional as F
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


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_out: int,
        # vocab_size: int,
        vectors: torch.Tensor,
        # seq_len: int = 100,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_filters: int = 12,
        filter_sizes: List[int] = [1, 3, 5],
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.embeddings = nn.Embedding.from_pretrained(vectors)
        self.num_filters = num_filters

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, self.embedding_dim)) for K in self.filter_sizes]
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, n_out)

    def forward(self, data):
        embeddings = self.embeddings(data).unsqueeze(1)
        convs = [F.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in convs]
        pooled = self.dropout(torch.cat(pooled, 1))
        linear = self.fc(pooled)
        return linear
