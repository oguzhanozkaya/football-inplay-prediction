"""Raw PyTorch model definitions."""

from __future__ import annotations

import torch
from torch import nn


class NumericMLP(nn.Module):
    """Tabular numeric deep learning baseline."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        return self.network(numeric_features).squeeze(-1)


class NumericGRU(nn.Module):
    """GRU model over lag-structured numeric feature sequences."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, numeric_sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(numeric_sequence)
        return self.head(hidden[-1]).squeeze(-1)


class TextCNNEncoder(nn.Module):
    """TextCNN encoder with randomly initialized project-trained embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        channel_count: int = 48,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channel_count, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = channel_count * len(kernel_sizes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            pooled.append(torch.max(activation, dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class TextCNNRegressor(nn.Module):
    """Text-only inflation forecast model."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.encoder = TextCNNEncoder(vocabulary_size)
        self.head = nn.Linear(self.encoder.output_size, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(token_ids)).squeeze(-1)


class FusionRegressor(nn.Module):
    """Numeric plus text fusion forecast model."""

    def __init__(self, numeric_input_size: int, vocabulary_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.text_encoder = TextCNNEncoder(vocabulary_size)
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_input_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + self.text_encoder.output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, numeric_features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        numeric_representation = self.numeric_projection(numeric_features)
        text_representation = self.text_encoder(token_ids)
        return self.head(torch.cat([numeric_representation, text_representation], dim=1)).squeeze(-1)
