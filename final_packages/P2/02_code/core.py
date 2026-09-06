"""Standalone frozen C3 numerical primitives, no research-repository imports."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
TEMPORAL_FEATURES = ("doy_sin", "doy_cos", "hour_sin", "hour_cos", "m2_sin", "m2_cos")


def arrays(frame, actualdepth=False):
    if actualdepth:
        raise ValueError("Only frozen C3 context is packaged")
    result = build_arrays(frame)
    if not all(np.isfinite(value).all() for value in result):
        raise ValueError("nonfinite C3 inputs")
    return result


def make_model(arm, context_features):
    if arm != "v23_blockmask":
        raise ValueError("Only frozen C3 architecture is packaged")
    return VerticalDeepSet(8, context_features, hidden=32)


def _wide(observations: pd.DataFrame, value: str) -> pd.DataFrame:
    return observations.pivot(index="time", columns="layer", values=value).sort_index()


def _common_features(observations: pd.DataFrame) -> tuple[pd.Index, dict[str, np.ndarray]]:
    temp, psal = _wide(observations, "temp"), _wide(observations, "psal")
    depth, nominal = _wide(observations, "depth"), _wide(observations, "nominal_depth")
    times = temp.index
    common: dict[str, np.ndarray] = {}
    for layer in PUBLIC_LAYERS:
        for prefix, wide in (
            ("temp", temp),
            ("psal", psal),
            ("depth", depth),
            ("nominal", nominal),
        ):
            common[f"{prefix}_{layer}"] = (
                wide.get(layer, pd.Series(index=times, dtype=float)).reindex(times).to_numpy(float)
            )
    public = np.column_stack([common[f"temp_{layer}"] for layer in PUBLIC_LAYERS])
    count = np.isfinite(public).sum(axis=1)
    mean = np.divide(
        np.nansum(public, axis=1), count, out=np.full(len(public), np.nan), where=count > 0
    )
    variance = np.divide(
        np.nansum((public - mean[:, None]) ** 2, axis=1),
        count,
        out=np.full(len(public), np.nan),
        where=count > 0,
    )
    value_range = np.full(len(public), np.nan)
    populated = count > 0
    value_range[populated] = np.nanmax(public[populated], axis=1) - np.nanmin(
        public[populated], axis=1
    )
    common["public_temp_count"] = count
    common["public_temp_mean"] = mean
    common["public_temp_std"] = np.sqrt(variance)
    common["public_temp_range"] = value_range
    kst = pd.to_datetime(times, utc=True).tz_convert("Asia/Seoul")
    minute = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    doy = kst.dayofyear.to_numpy() + minute / 1440
    parsed_time = pd.to_datetime(times, utc=True).as_unit("ns")
    epoch_seconds = parsed_time.asi8 / 1e9
    common.update(
        {
            "year": kst.year.to_numpy(),
            "elapsed_days": (epoch_seconds - epoch_seconds.min()) / 86_400,
            "doy_sin": np.sin(2 * np.pi * doy / 365.2425),
            "doy_cos": np.cos(2 * np.pi * doy / 365.2425),
            "hour_sin": np.sin(2 * np.pi * minute / 1440),
            "hour_cos": np.cos(2 * np.pi * minute / 1440),
            "m2_sin": np.sin(2 * np.pi * epoch_seconds / (12.42 * 3600)),
            "m2_cos": np.cos(2 * np.pi * epoch_seconds / (12.42 * 3600)),
        }
    )
    return times, common


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise KeyError(f"required NCR_LGBM column missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def compute_profile_scale(frame: pd.DataFrame, *, floor_c: float = 0.5) -> np.ndarray:
    """Compute the preregistered endpoint/public-range temperature scale."""

    if not np.isfinite(floor_c) or floor_c <= 0:
        raise ValueError("profile scale floor must be finite and positive")
    temp_1 = _numeric(frame, "temp_1")
    temp_5 = _numeric(frame, "temp_5")
    public_range = _numeric(frame, "public_temp_range")
    endpoints_available = np.isfinite(temp_1) & np.isfinite(temp_5)
    raw_scale = np.where(endpoints_available, np.abs(temp_1 - temp_5), public_range)
    scale = np.maximum(raw_scale, floor_c)
    if not np.isfinite(scale).all():
        raise ValueError("profile scale is unavailable for one or more rows")
    return scale


def build_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = SimpleNamespace(
        baseline=_numeric(frame, "baseline"), profile_scale=compute_profile_scale(frame)
    )
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
        pd.to_numeric(frame[name], errors="raise").to_numpy(float) for name in TEMPORAL_FEATURES
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


class VerticalDeepSet(nn.Module):
    """Shared public-depth element map followed by commutative pooling."""

    def __init__(self, token_features: int, context_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(token_features, hidden),
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

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


def nominal_baseline(frame: pd.DataFrame) -> np.ndarray:
    """One contract for train/query: nominal interpolation with endpoint clamp."""
    temperatures = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    depths = frame[[f"nominal_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    targets = frame["target_depth"].to_numpy(float)
    result = np.full(len(frame), np.nan)
    for index, target in enumerate(targets):
        keep = np.isfinite(temperatures[index]) & np.isfinite(depths[index])
        if keep.sum() < 2 or not np.isfinite(target):
            continue
        order = np.argsort(depths[index, keep], kind="stable")
        result[index] = np.interp(
            target, depths[index, keep][order], temperatures[index, keep][order]
        )
    return result


def refresh_public(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    values = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    minimum = np.min(np.where(finite, values, np.inf), axis=1)
    maximum = np.max(np.where(finite, values, -np.inf), axis=1)
    frame["public_temp_count"] = count
    frame["public_temp_range"] = np.where(count > 0, maximum - minimum, np.nan)
    frame["baseline"] = nominal_baseline(frame)
    # A truth-free placeholder is required by legacy array validation, not used
    # as a feature or training target. Real labels are maintained separately.
    frame["target"] = frame["baseline"]
    return frame


def public_frame(observations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    public = observations.loc[observations["layer"].isin(PUBLIC_LAYERS)].copy()
    times, common = _common_features(public)
    lookup = pd.DataFrame(common, index=pd.DatetimeIndex(times))
    metadata = observations.loc[
        observations["layer"].isin(TARGET_LAYERS), ["time", "layer", "nominal_depth", "depth"]
    ].copy()
    metadata = metadata.rename(
        columns={"nominal_depth": "target_depth", "depth": "target_actual_depth"}
    )
    labels = observations.loc[observations["layer"].isin(TARGET_LAYERS), "temp"].to_numpy(float)
    frame = metadata.join(lookup, on="time", validate="many_to_one").reset_index(drop=True)
    frame["station"] = "S-ORS"
    frame = refresh_public(frame)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).astype(str)
    assert not {"temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"}.intersection(frame)
    return frame, labels


def block_selection(times: pd.Series, config: dict) -> np.ndarray:
    """Fixed calendar blocks, independent of labels and measured performance."""
    parsed = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    unique = parsed.unique().sort_values()
    rng = np.random.default_rng(config["seed"])
    selected = np.zeros(len(unique), dtype=bool)
    target = int(np.ceil(config["coverage"] * len(unique)))
    for _ in range(10000):
        if selected.sum() >= target:
            break
        start = unique[int(rng.integers(len(unique)))]
        stop = start + pd.Timedelta(days=int(rng.choice(config["length_days"])))
        selected |= np.asarray((unique >= start) & (unique < stop))
    return np.asarray(parsed.isin(unique[selected]))


def training_arrays(
    frame: pd.DataFrame, truth: np.ndarray, arm: str, config: dict
) -> tuple[tuple[np.ndarray, ...], dict]:
    weights, weight_receipt = domain_balanced_weights(
        frame["layer"].to_numpy(),
        pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul"),
    )
    origin_count = len(frame)
    duplicated = 0
    if arm.endswith("blockmask"):
        mask_rows = block_selection(frame["time"], config["blockmask"])
        masked = frame.copy()
        masked.loc[mask_rows, ["temp_5", "psal_5"]] = np.nan
        masked = refresh_public(masked)
        eligible = (
            mask_rows
            & (masked["public_temp_count"].to_numpy() >= 2)
            & np.isfinite(masked["baseline"].to_numpy())
        )
        duplicated = int(eligible.sum())
        augmentation_weights = weights[eligible] * 0.5
        weights[eligible] *= 0.5
        weights = np.concatenate((weights, augmentation_weights))
        frame = pd.concat((frame, masked.loc[eligible]), ignore_index=True)
        truth = np.concatenate((truth, truth[eligible]))
    feature_arrays = arrays(frame, arm.endswith("actualdepth"))
    target = ((truth - frame["baseline"].to_numpy(float)) / compute_profile_scale(frame)).astype(
        np.float32
    )
    assert np.isfinite(target).all() and np.isclose(weights.sum(), origin_count, rtol=1e-5)
    return (*feature_arrays, target, weights), {
        "original_rows": origin_count,
        "augmented_rows": duplicated,
        "training_weight_sum": float(weights.sum()),
        "domain_weights": weight_receipt,
    }


def fit_model(
    data: tuple[np.ndarray, ...], arm: str, seed: int, config: dict, progress
) -> tuple[torch.nn.Module, dict]:
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(config["device"])
    model = make_model(arm, data[2].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    # Whole numeric arrays fit comfortably in VRAM; one worker, no CPU loaders.
    tensors = tuple(torch.from_numpy(value).to(device) for value in data)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(data[0]), generator=generator).to(device)
        numerator, denominator = 0.0, 0.0
        for offset in range(0, len(order), config["batch_size"]):
            ids = order[offset : offset + config["batch_size"]]
            tokens = tensors[0][ids].detach().clone().requires_grad_(True)
            batch_mask, context, target, weights = [value[ids] for value in tensors[1:]]
            optimizer.zero_grad(set_to_none=True)
            estimate = model(tokens, batch_mask, context)
            raw_loss = F.smooth_l1_loss(estimate, target, beta=1.0, reduction="none")
            loss = (raw_loss * weights).sum() / weights.sum().clamp_min(1e-12)
            penalty = observed_temperature_gradient_penalty(raw_loss, tokens, batch_mask, weights)
            objective = loss + config["gradient_coefficient"] * penalty
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite training objective")
            objective.backward()
            optimizer.step()
            numerator += float((raw_loss.detach() * weights).sum().cpu())
            denominator += float(weights.sum().cpu())
        losses.append(numerator / denominator)
        if epoch % 10 == 0 or epoch + 1 == config["epochs"]:
            progress(epoch + 1, time.monotonic() - started)
    model = model.cpu().eval()
    del tensors, optimizer
    torch.cuda.empty_cache()
    return model, {
        "seed": seed,
        "epochs": config["epochs"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "runtime_seconds": time.monotonic() - started,
        "device": str(device),
        "cpu_threads": config["cpu_threads"],
    }
