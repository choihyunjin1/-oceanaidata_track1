"""P2-adapted local implementations of the shortlisted sequence-model families.

These are deliberately named ``*_style``: they preserve the central structural
idea of the cited family under one fair P2 input/output contract, but they are
not copied upstream repositories or pretrained checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DepthQueryDecoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        depths = torch.tensor([7.04, 9.44, 14.74], dtype=torch.float32) / 50.0
        encodings = torch.stack(
            (
                depths,
                torch.sin(2 * math.pi * depths),
                torch.cos(2 * math.pi * depths),
                torch.sin(4 * math.pi * depths),
                torch.cos(4 * math.pi * depths),
            ),
            dim=-1,
        )
        self.register_buffer("depth_encodings", encodings)
        self.query = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state: Tensor) -> Tensor:
        query = self.query(self.depth_encodings).view(1, 1, 3, -1)
        expanded = state.unsqueeze(2).expand(-1, -1, 3, -1)
        return self.output(torch.cat((expanded, query.expand_as(expanded)), dim=-1)).squeeze(-1)


class ModernTCNBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, kernel: int = 7, dropout: float = 0.05) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel, padding=padding, dilation=dilation, groups=hidden
        )
        self.norm = nn.LayerNorm(hidden)
        self.expand = nn.Linear(hidden, hidden * 4)
        self.contract = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.depthwise(inputs.transpose(1, 2)).transpose(1, 2)
        state = self.norm(state)
        left, gate = self.expand(state).chunk(2, dim=-1)
        state = self.contract(F.gelu(left) * torch.sigmoid(gate))
        return inputs + self.dropout(state)


class DepthQueryBiTCN(nn.Module):
    """ModernTCN temporal branch plus a DeepONet-inspired depth query head."""

    def __init__(self, channels: int, hidden: int = 256, blocks: int = 10) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList(
            ModernTCNBlock(hidden, 2 ** (index % 8)) for index in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.input(inputs)
        for block in self.blocks:
            state = block(state)
        return self.decoder(self.norm(state))


class LSTIStyle(nn.Module):
    """Short convolution and bidirectional long-context branches with a learned gate."""

    def __init__(self, channels: int, hidden: int = 192, layers: int = 3) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.short = nn.Sequential(
            nn.Conv1d(hidden, hidden, 7, padding=3, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.long = nn.GRU(
            hidden,
            hidden // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.gate = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        embedded = self.input(inputs)
        short = self.short(embedded.transpose(1, 2)).transpose(1, 2)
        long, _ = self.long(embedded)
        weight = torch.sigmoid(self.gate(torch.cat((short, long), dim=-1)))
        return self.decoder(self.norm(weight * short + (1.0 - weight) * long + embedded))


class ImputeFormerStyle(nn.Module):
    """Low-rank channel projection with patchwise temporal self-attention."""

    def __init__(
        self,
        channels: int,
        hidden: int = 256,
        rank: int = 48,
        layers: int = 4,
        heads: int = 8,
        patch: int = 6,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.low_rank = nn.Sequential(nn.Linear(channels, rank), nn.GELU(), nn.Linear(rank, hidden))
        encoder = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.skip = nn.Linear(channels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        pooled = F.avg_pool1d(inputs.transpose(1, 2), self.patch, self.patch, ceil_mode=True)
        state = self.encoder(self.low_rank(pooled.transpose(1, 2)))
        state = F.interpolate(
            state.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.decoder(self.norm(state + self.skip(inputs)))


class MixerBranch(nn.Module):
    def __init__(self, hidden: int, blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ModernTCNBlock(hidden, 2**index, kernel=5, dropout=0.1) for index in range(blocks)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class TimeMixerPlusPlusStyle(nn.Module):
    """Multi-resolution temporal mixing with an explicit low-frequency branch."""

    def __init__(self, channels: int, hidden: int = 192) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.branches = nn.ModuleList(MixerBranch(hidden) for _ in (1, 3, 12))
        self.scales = (1, 3, 12)
        self.frequency_gate = nn.Parameter(torch.zeros(hidden, 24, 2))
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.input(inputs)
        length = base.shape[1]
        outputs: list[Tensor] = []
        for scale, branch in zip(self.scales, self.branches, strict=True):
            current = base
            if scale > 1:
                current = F.avg_pool1d(
                    base.transpose(1, 2), scale, scale, ceil_mode=True
                ).transpose(1, 2)
            current = branch(current)
            if current.shape[1] != length:
                current = F.interpolate(
                    current.transpose(1, 2), size=length, mode="linear", align_corners=False
                ).transpose(1, 2)
            outputs.append(current)
        # CUDA FFT does not accept bf16; keep only the spectral operation in fp32.
        spectrum = torch.fft.rfft(base.float(), dim=1)
        modes = min(spectrum.shape[1], self.frequency_gate.shape[1])
        gate = torch.view_as_complex(self.frequency_gate[:, :modes].contiguous()).transpose(0, 1)
        filtered = torch.zeros_like(spectrum)
        filtered[:, :modes] = spectrum[:, :modes] * torch.tanh(gate).unsqueeze(0)
        frequency = torch.fft.irfft(filtered, n=length, dim=1)
        state = self.fuse(torch.cat((*outputs, frequency), dim=-1)) + base
        return self.decoder(self.norm(state))


class SpectralConv1d(nn.Module):
    def __init__(self, hidden: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(hidden, hidden, modes, 2) * (hidden**-0.5))

    def forward(self, inputs: Tensor) -> Tensor:
        # CUDA FFT does not accept bf16; keep only the spectral operation in fp32.
        spectrum = torch.fft.rfft(inputs.float(), dim=1)
        modes = min(self.modes, spectrum.shape[1])
        weights = torch.view_as_complex(self.weight[:, :, :modes].contiguous())
        output = torch.zeros_like(spectrum)
        output[:, :modes] = torch.einsum("bmi,iom->bmo", spectrum[:, :modes], weights)
        return torch.fft.irfft(output, n=inputs.shape[1], dim=1)


class FNOStyle(nn.Module):
    def __init__(self, channels: int, hidden: int = 128, layers: int = 4, modes: int = 32) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.spectral = nn.ModuleList(SpectralConv1d(hidden, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.input(inputs)
        for spectral, local, norm in zip(self.spectral, self.local, self.norms, strict=True):
            state = norm(state + F.gelu(spectral(state) + local(state)))
        return self.decoder(state)


class PatchFoundationStyle(nn.Module):
    """From-scratch patch Transformer proxy for MOMENT/UniTS-like capacity."""

    def __init__(
        self,
        channels: int,
        hidden: int = 256,
        layers: int = 6,
        heads: int = 8,
        patch: int = 6,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Conv1d(channels, hidden, kernel_size=patch, stride=patch)
        encoder = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.skip = nn.Linear(channels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DepthQueryDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        patches = self.patch_embed(inputs.transpose(1, 2)).transpose(1, 2)
        position = torch.arange(patches.shape[1], device=inputs.device, dtype=inputs.dtype)
        frequencies = torch.arange(1, patches.shape[-1] // 2 + 1, device=inputs.device)
        positional = torch.cat(
            (
                torch.sin(position[:, None] / frequencies[None, :]),
                torch.cos(position[:, None] / frequencies[None, :]),
            ),
            dim=-1,
        )[:, : patches.shape[-1]]
        state = self.encoder(patches + positional.unsqueeze(0))
        state = F.interpolate(
            state.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.decoder(self.norm(state + self.skip(inputs)))


class DiffusionTransformerDenoiser(nn.Module):
    def __init__(self, channels: int, hidden: int = 160) -> None:
        super().__init__()
        self.input = nn.Linear(channels + 3 + 16, hidden)
        encoder = nn.TransformerEncoderLayer(
            hidden,
            8,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, 4, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 3)

    def forward(self, condition: Tensor, noisy: Tensor, step_embedding: Tensor) -> Tensor:
        repeated = step_embedding[:, None, :].expand(-1, condition.shape[1], -1)
        return self.output(self.encoder(self.input(torch.cat((condition, noisy, repeated), -1))))


class DiffusionSSMDenoiser(nn.Module):
    def __init__(self, channels: int, hidden: int = 192) -> None:
        super().__init__()
        self.input = nn.Linear(channels + 3 + 16, hidden)
        self.blocks = nn.ModuleList(
            ModernTCNBlock(hidden, 2 ** (index % 8), kernel=5, dropout=0.05) for index in range(10)
        )
        self.output = nn.Linear(hidden, 3)

    def forward(self, condition: Tensor, noisy: Tensor, step_embedding: Tensor) -> Tensor:
        repeated = step_embedding[:, None, :].expand(-1, condition.shape[1], -1)
        state = self.input(torch.cat((condition, noisy, repeated), -1))
        for block in self.blocks:
            state = block(state)
        return self.output(state)


class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        denoiser: Literal["transformer", "ssm"] = "transformer",
        steps: int = 20,
    ) -> None:
        super().__init__()
        self.steps = steps
        betas = torch.linspace(1e-4, 0.05, steps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
        self.denoiser = (
            DiffusionTransformerDenoiser(channels)
            if denoiser == "transformer"
            else DiffusionSSMDenoiser(channels)
        )

    @staticmethod
    def _step_embedding(step: Tensor, width: int = 16) -> Tensor:
        half = width // 2
        scale = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=step.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = step.float()[:, None] * scale[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def training_loss(
        self,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        layer_weights: Tensor | None = None,
    ) -> Tensor:
        step = torch.randint(self.steps, (condition.shape[0],), device=condition.device)
        noise = torch.randn_like(target)
        alpha = self.alpha_bar[step].view(-1, 1, 1)
        noisy = alpha.sqrt() * target + (1.0 - alpha).sqrt() * noise
        predicted_noise = self.denoiser(condition, noisy, self._step_embedding(step))
        squared = (predicted_noise - noise).square()
        if layer_weights is not None:
            squared = squared * layer_weights.view(1, 1, 3)
        return (squared * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def predict(self, condition: Tensor, *, samples: int = 4) -> Tensor:
        predictions: list[Tensor] = []
        for _ in range(samples):
            current = torch.randn((*condition.shape[:2], 3), device=condition.device)
            for index in reversed(range(self.steps)):
                step = torch.full(
                    (condition.shape[0],), index, device=condition.device, dtype=torch.long
                )
                epsilon = self.denoiser(condition, current, self._step_embedding(step))
                alpha = self.alpha_bar[index]
                clean = (current - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()
                if index:
                    previous = self.alpha_bar[index - 1]
                    current = previous.sqrt() * clean + (1.0 - previous).sqrt() * epsilon
                else:
                    current = clean
            predictions.append(current)
        return torch.stack(predictions).mean(0)


ModelName = Literal[
    "depth_query_bitcn",
    "lsti_style",
    "imputeformer_style",
    "timemixerpp_style",
    "fno_style",
    "moment_units_scratch",
    "csdi_style",
    "sssd_ssm_style",
]


@dataclass(frozen=True)
class ModelSpec:
    name: ModelName
    learning_rates: tuple[float, ...]
    weight_decay: float
    max_epochs: int
    patience: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("depth_query_bitcn", (1e-4, 3e-4, 1e-3), 1e-3, 300, 30),
    ModelSpec("lsti_style", (1e-4, 3e-4, 1e-3), 1e-3, 300, 30),
    ModelSpec("imputeformer_style", (1e-4, 3e-4), 1e-3, 250, 25),
    ModelSpec("timemixerpp_style", (1e-4, 3e-4, 1e-3), 1e-3, 300, 30),
    ModelSpec("fno_style", (1e-4, 3e-4, 1e-3), 1e-3, 250, 25),
    ModelSpec("moment_units_scratch", (1e-4, 3e-4), 1e-3, 250, 25),
    ModelSpec("csdi_style", (1e-4, 3e-4), 1e-4, 200, 20),
    ModelSpec("sssd_ssm_style", (1e-4, 3e-4), 1e-4, 200, 20),
)


def build_model(name: ModelName, channels: int) -> nn.Module:
    builders = {
        "depth_query_bitcn": lambda: DepthQueryBiTCN(channels),
        "lsti_style": lambda: LSTIStyle(channels),
        "imputeformer_style": lambda: ImputeFormerStyle(channels),
        "timemixerpp_style": lambda: TimeMixerPlusPlusStyle(channels),
        "fno_style": lambda: FNOStyle(channels),
        "moment_units_scratch": lambda: PatchFoundationStyle(channels),
        "csdi_style": lambda: ConditionalDiffusion(channels, denoiser="transformer"),
        "sssd_ssm_style": lambda: ConditionalDiffusion(channels, denoiser="ssm"),
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"unknown P2 deep model: {name}") from exc


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
