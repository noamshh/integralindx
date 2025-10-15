import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, max_seq_len: int, dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, dim)  # [L, D]
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)  # [B, L, D]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: vocabulary size
            dim: embedding dimension
            num_layers: number of encoder layers
            num_heads: number of attention heads
            feedforward_dim: feedforward dimension
            max_seq_len: maximum sequence length
            dropout: dropout
        """
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = PositionalEncoding(max_seq_len, dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(token_ids) * math.sqrt(self.dim)
        x = self.pos_emb(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return x

    def mean_pool(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)
        attention_mask = (~mask).unsqueeze(-1).float()  # [B, L, 1]
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)  # [B, D]
        sum_mask = attention_mask.sum(dim=1)  # [B, 1]
        return sum_hidden / sum_mask.clamp(min=1e-9)
