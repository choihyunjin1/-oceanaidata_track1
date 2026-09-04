"""Fixed physical-depth message passing for same-time P2 public profiles."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def fixed_depth_adjacency(
    tokens: Tensor,
    token_mask: Tensor,
    bandwidth_normalized_depth: float,
) -> Tensor:
    """Build a masked, row-normalized graph from nominal-depth coordinates."""
    if bandwidth_normalized_depth <= 0:
        raise ValueError("depth bandwidth must be positive")
    depth = tokens[..., 3]
    distance = torch.abs(depth.unsqueeze(2) - depth.unsqueeze(1))
    adjacency = torch.exp(-distance / float(bandwidth_normalized_depth))
    valid = token_mask.bool()
    edge_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    identity = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool).unsqueeze(0)
    adjacency = adjacency.masked_fill(~edge_mask | identity, 0.0)
    denominator = adjacency.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return adjacency / denominator


class FixedDepthGraphEncoder(nn.Module):
    """One fixed-adjacency message block followed by invariant pooling."""

    def __init__(
        self,
        token_features: int,
        context_features: int,
        hidden: int = 32,
        blocks: int = 1,
        bandwidth_normalized_depth: float = 0.20,
    ) -> None:
        super().__init__()
        if blocks != 1:
            raise ValueError("the sealed graph encoder has exactly one block")
        self.blocks = int(blocks)
        self.bandwidth_normalized_depth = float(bandwidth_normalized_depth)
        self.element = nn.Sequential(nn.Linear(token_features, hidden), nn.ReLU())
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        valid = token_mask.bool()
        if torch.any(valid.sum(dim=1) < 2):
            raise ValueError("each graph needs at least two valid public-depth nodes")
        encoded = self.element(tokens)
        adjacency = fixed_depth_adjacency(
            tokens, token_mask, self.bandwidth_normalized_depth
        )
        message = torch.bmm(adjacency, encoded)
        updated = self.update(torch.cat((encoded, message), dim=2))
        return updated * valid.unsqueeze(-1)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.encode(tokens, token_mask)
        valid = token_mask.bool().unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = encoded.sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~valid, negative).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


__all__ = ["FixedDepthGraphEncoder", "fixed_depth_adjacency"]
