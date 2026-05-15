from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn

# ============================================================================
# Common Components
# ============================================================================


class EmbeddingPredHead(nn.Module):
    """Generic prediction head for embedding-based downstream tasks."""

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        return self.fc2(x)
    
    
class LinearPredHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim : int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)