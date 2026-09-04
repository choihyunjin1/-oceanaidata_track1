"""Compact same-time public-depth set interaction model for P2.

The encoder has no temporal axis and no positional encoding.  Its sole
attention block therefore models interactions among the currently available
public-depth tokens while remaining equivariant to their ordering.  The final
masked mean/max reduction is permutation invariant.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CompactSetTransformer(nn.Module):
    """One-block, two-head permutation-invariant set regressor."""

    def __init__(
        self,
        token_features: int,
        context_features: int,
        hidden: int = 32,
        heads: int = 2,
        blocks: int = 1,
    ) -> None:
        super().__init__()
        if blocks != 1:
            raise ValueError("the sealed compact architecture has exactly one block")
        if hidden % heads:
            raise ValueError("hidden width must be divisible by attention heads")
        self.blocks = blocks
        self.heads = heads
        self.element = nn.Sequential(nn.Linear(token_features, hidden), nn.ReLU())
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        """Return public-layer states equivariant to token permutation."""
        valid = token_mask.bool()
        if torch.any(valid.sum(dim=1) < 1):
            raise ValueError("each set needs at least one valid token")
        values = self.element(tokens)
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = self.norm1(values + attended)
        values = self.norm2(values + self.feed_forward(values))
        return values * valid.unsqueeze(-1)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.encode(tokens, token_mask)
        valid = token_mask.bool().unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = encoded.sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~valid, negative).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


def architecture_receipt(model: CompactSetTransformer) -> dict[str, int | bool]:
    """Return a stable receipt for the sealed architecture contract."""
    return {
        "attention_blocks": int(model.blocks),
        "attention_heads": int(model.heads),
        "positional_encoding": False,
        "temporal_attention": False,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


__all__ = ["CompactSetTransformer", "architecture_receipt"]
