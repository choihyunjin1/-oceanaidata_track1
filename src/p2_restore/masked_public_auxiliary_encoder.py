"""Local prefix-only masked-public auxiliary encoder for P2."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskedPublicAuxiliaryEncoder(nn.Module):
    """Ordered public-layer encoder with target and public-reconstruction heads."""

    def __init__(
        self,
        token_features: int,
        public_layers: int,
        context_features: int,
        hidden: int = 64,
        latent: int = 32,
    ) -> None:
        super().__init__()
        self.public_layers = int(public_layers)
        width = token_features * public_layers + public_layers + context_features
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, 1),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, public_layers),
        )

    def encode(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        flat = torch.cat((tokens.flatten(start_dim=1), token_mask, context), dim=1)
        return self.encoder(flat)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        return self.target_head(self.encode(tokens, token_mask, context)).squeeze(1)

    def reconstruct(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        return self.reconstruction_head(self.encode(tokens, token_mask, context))


def mask_public_index(
    tokens: Tensor,
    token_mask: Tensor,
    index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mask exactly one fixed public-layer slot and return eligible rows."""
    if not 0 <= index < tokens.shape[1]:
        raise ValueError("public-layer mask index out of range")
    masked_tokens = tokens.clone()
    masked_mask = token_mask.clone()
    eligible = token_mask[:, index].bool() & (token_mask.sum(dim=1) >= 3)
    masked_tokens[:, index] = 0.0
    masked_mask[:, index] = 0.0
    return masked_tokens, masked_mask, eligible


__all__ = ["MaskedPublicAuxiliaryEncoder", "mask_public_index"]
