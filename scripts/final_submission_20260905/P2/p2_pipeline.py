"""Shared scratch-training and inference primitives for the frozen P2 v52 model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
TEMPORAL_FEATURES = (
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
    "m2_sin",
    "m2_cos",
)


def activate_source(package_dir: str | Path) -> None:
    source = Path(package_dir).resolve() / "07_source" / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"missing packaged P2 source tree: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


class MaskedThirdCentralMomentProfileVerticalDeepSet(nn.Module):
    """Exact 5-token v52 DeepSets architecture (5,889 parameters)."""

    def __init__(self, token_features: int = 8, context_features: int = 11, hidden: int = 32):
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(token_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Preserve the historical v50 RNG/initialization order exactly: build the
        # v13 two-pool head first, then allocate the expanded three-pool layer and
        # identity-initialize its inherited mean/max/context columns. Constructing
        # a three-pool head directly changes every seed's retained head weights.
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        original = self.head[0]
        expanded = nn.Linear(hidden * 3 + context_features, hidden)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.bias.copy_(original.bias)
            expanded.weight[:, : hidden * 2].copy_(original.weight[:, : hidden * 2])
            expanded.weight[:, hidden * 3 :].copy_(original.weight[:, hidden * 2 :])
        self.head[0] = expanded

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1).to(encoded.dtype)
        raw_count = mask.sum(dim=1)
        count = raw_count.clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        centered = (encoded - mean.unsqueeze(1)) * mask
        third = centered.pow(3).sum(dim=1) / count
        third = torch.where(raw_count > 0, third, torch.zeros_like(third))
        return self.head(torch.cat((mean, maximum, third, context), dim=1)).squeeze(1)


def build_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from p2_restore.normalized_curvature_residual import build_normalized_curvature_design

    design = build_normalized_curvature_design(frame)
    n_rows = len(frame)
    public_psal = np.column_stack(
        [pd.to_numeric(frame[f"psal_{layer}"], errors="coerce") for layer in PUBLIC_LAYERS]
    ).astype(float)
    psal_finite = np.isfinite(public_psal)
    psal_count = psal_finite.sum(axis=1)
    psal_mean = np.divide(
        np.nansum(public_psal, axis=1),
        psal_count,
        out=np.zeros(n_rows, dtype=float),
        where=psal_count > 0,
    )
    psal_min = np.zeros(n_rows, dtype=float)
    psal_max = np.zeros(n_rows, dtype=float)
    usable_psal = psal_count > 0
    psal_min[usable_psal] = np.nanmin(public_psal[usable_psal], axis=1)
    psal_max[usable_psal] = np.nanmax(public_psal[usable_psal], axis=1)
    psal_scale = np.maximum(psal_max - psal_min, 0.05)
    target_depth = pd.to_numeric(frame["target_depth"], errors="raise").to_numpy(float)
    baseline = design.baseline
    tokens = np.zeros((n_rows, len(PUBLIC_LAYERS), 8), dtype=np.float32)
    token_mask = np.zeros((n_rows, len(PUBLIC_LAYERS)), dtype=np.float32)
    for index, layer in enumerate(PUBLIC_LAYERS):
        temp = pd.to_numeric(frame[f"temp_{layer}"], errors="coerce").to_numpy(float)
        psal = pd.to_numeric(frame[f"psal_{layer}"], errors="coerce").to_numpy(float)
        depth = pd.to_numeric(frame[f"depth_{layer}"], errors="coerce").to_numpy(float)
        nominal = pd.to_numeric(frame[f"nominal_{layer}"], errors="coerce").to_numpy(float)
        present = np.column_stack(
            (np.isfinite(temp), np.isfinite(psal), np.isfinite(depth), np.isfinite(nominal))
        ).astype(np.float32)
        values = np.column_stack(
            (
                (temp - baseline) / design.profile_scale,
                (psal - psal_mean) / psal_scale,
                (depth - target_depth) / 50.0,
                (nominal - target_depth) / 50.0,
                present,
            )
        )
        values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
        tokens[:, index] = np.clip(values, -12.0, 12.0).astype(np.float32)
        token_mask[:, index] = (np.isfinite(temp) & np.isfinite(nominal)).astype(np.float32)
    if np.any(token_mask.sum(axis=1) < 2):
        raise ValueError("P2 row has fewer than two public temperature/depth tokens")
    layer = pd.to_numeric(frame["layer"], errors="raise").to_numpy(int)
    one_hot = np.column_stack([layer == value for value in TARGET_LAYERS]).astype(np.float32)
    context_values: list[np.ndarray] = [
        target_depth / 50.0,
        *[one_hot[:, index] for index in range(3)],
        np.log1p(design.profile_scale),
    ]
    context_values.extend(
        pd.to_numeric(frame[name], errors="raise").to_numpy(float)
        for name in TEMPORAL_FEATURES
    )
    context = np.column_stack(context_values).astype(np.float32)
    if context.shape[1] != 11 or not np.isfinite(context).all():
        raise ValueError("P2 context feature contract failed")
    return tokens, token_mask, context


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    local = pd.DatetimeIndex(local_time)
    frame = pd.DataFrame(
        {
            "layer": np.asarray(layer, dtype=int),
            "calendar_month": local.month,
            "kst_date": local.date,
        }
    )
    groups = sorted(
        frame[["layer", "calendar_month"]].drop_duplicates().itertuples(index=False, name=None)
    )
    raw = np.zeros(len(frame), dtype=float)
    receipt: dict[str, Any] = {}
    for target_layer, month in groups:
        group = frame["layer"].eq(target_layer) & frame["calendar_month"].eq(month)
        days = sorted(frame.loc[group, "kst_date"].unique())
        for day in days:
            selected = group & frame["kst_date"].eq(day)
            raw[selected.to_numpy()] = 1.0 / (len(groups) * len(days) * int(selected.sum()))
        receipt[f"layer{target_layer}:month{month:02d}"] = {
            "rows": int(group.sum()),
            "days": len(days),
            "raw_weight_sum": float(raw[group.to_numpy()].sum()),
        }
    if not (np.isfinite(raw).all() and np.all(raw > 0.0)):
        raise ValueError("P2 domain-balanced weights are invalid")
    weights = raw / raw.mean()
    return weights.astype(np.float32), {
        "groups": receipt,
        "group_count": len(groups),
        "normalized_mean": float(weights.mean()),
        "normalized_min": float(weights.min()),
        "normalized_max": float(weights.max()),
    }


def observed_temperature_gradient_penalty(
    per_row_loss: Tensor,
    tokens: Tensor,
    token_mask: Tensor,
    row_weights: Tensor,
) -> Tensor:
    gradient = torch.autograd.grad(
        per_row_loss.sum(), tokens, create_graph=True, retain_graph=True
    )[0]
    observed = token_mask.to(dtype=gradient.dtype)
    per_row = (gradient[..., 0].square() * observed).sum(dim=1)
    per_row = per_row / observed.sum(dim=1).clamp_min(1.0)
    return (per_row * row_weights).sum() / row_weights.sum().clamp_min(1e-12)


def train_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    epochs: int = 60,
    batch_size: int = 4096,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_coefficient: float = 0.01,
) -> tuple[MaskedThirdCentralMomentProfileVerticalDeepSet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedThirdCentralMomentProfileVerticalDeepSet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train = tuple(
        torch.from_numpy(value)
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    penalties: list[float] = []
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(target), generator=generator)
        data_numerator = 0.0
        penalty_numerator = 0.0
        denominator = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            batch_tokens = batch[0].detach().clone().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_tokens, batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(prediction, batch[3], beta=1.0, reduction="none")
            data_loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            penalty = observed_temperature_gradient_penalty(
                raw_loss, batch_tokens, batch[1], batch[4]
            )
            (data_loss + gradient_coefficient * penalty).backward()
            optimizer.step()
            weight_sum = float(batch[4].sum().detach().cpu())
            data_numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            penalty_numerator += float(penalty.detach().cpu())
            denominator += weight_sum
            batches += 1
        losses.append(data_numerator / denominator)
        penalties.append(penalty_numerator / max(batches, 1))
    if not np.isfinite(losses).all() or not np.isfinite(penalties).all():
        raise ValueError("P2 training became non-finite")
    return model.cpu().eval(), {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "data_loss_first": losses[0],
        "data_loss_last": losses[-1],
        "gradient_penalty_first": penalties[0],
        "gradient_penalty_last": penalties[-1],
    }


def predict_model(
    model: nn.Module,
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            stop = start + batch_size
            output.append(
                model(
                    torch.from_numpy(tokens[start:stop]).to(device),
                    torch.from_numpy(mask[start:stop]).to(device),
                    torch.from_numpy(context[start:stop]).to(device),
                )
                .cpu()
                .numpy()
            )
    prediction = np.concatenate(output).astype(float)
    if not np.isfinite(prediction).all():
        raise ValueError("P2 model prediction is non-finite")
    return prediction
