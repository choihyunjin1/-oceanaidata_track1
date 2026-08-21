"""Leak-safe preparation utilities for the preregistered P3 RevIN PatchTST probe.

The module deliberately separates past-context preparation from target attachment.  The
preparation runner imports only the past-context path; outer targets remain unopened until a
future, explicitly authorised training run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import LEADS, STATIONS

CONTEXT_ROWS = 289
WAVE_ROWS = 145
ATMOS_ROWS = 289
RAW_COLUMNS = ("hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph")
WAVE_RAW_INDICES = (0, 1, 2, 3)
ATMOS_RAW_INDICES = (4, 5, 6, 7, 8, 9)
ACTIVE_BLEND_LEADS = (12, 18, 24)
PROTECTED_LEADS = (3, 6, 9)
EXPECTED_EXPERIMENT_ID = "p3_revin_patch_v1"
FULL_AUTHORIZATION_TOKEN = "ROOT_APPROVED_P3_REVIN_PATCH_V1"


@dataclass(frozen=True)
class PatchModelConfig:
    """Frozen architecture for the first nonlinear trajectory one-shot."""

    d_model: int = 64
    encoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 128
    dropout: float = 0.1
    wave_patch_steps: int = 6
    wave_stride_steps: int = 3
    atmos_patch_steps: int = 12
    atmos_stride_steps: int = 6
    robust_scale_floor: float = 0.05

    @property
    def wave_patch_count(self) -> int:
        return 1 + (WAVE_ROWS - self.wave_patch_steps) // self.wave_stride_steps

    @property
    def atmos_patch_count(self) -> int:
        return 1 + (ATMOS_ROWS - self.atmos_patch_steps) // self.atmos_stride_steps

    def validate(self) -> None:
        expected = PatchModelConfig()
        if self != expected:
            raise ValueError(f"architecture differs from frozen v1 contract: {self!r}")
        if self.wave_patch_count != 47 or self.atmos_patch_count != 47:
            raise ValueError("two-hour/one-hour stride contract must yield 47 patches per stream")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")


@dataclass(frozen=True)
class PreparedStreams:
    wave: torch.Tensor
    atmos: torch.Tensor
    current_hs: torch.Tensor
    hs_scale: torch.Tensor


@dataclass(frozen=True)
class EpisodeDisjointFold:
    name: str
    train_ids: np.ndarray
    validation_ids: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class InnerEpisodeSplit:
    train_ids: np.ndarray
    validation_ids: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class EpochSelection:
    selected_epoch: int
    epochs_ran: int
    best_inner_rmse: float
    inner_rmse_history: tuple[float, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("preregistration must be a JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} differs from preregistered value: {actual!r} != {expected!r}")


def validate_preregistration(
    config: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    verify_frozen_files: bool = True,
) -> dict[str, Any]:
    """Fail closed if a structural, leakage, or frozen-artifact contract changed."""

    _require_equal(config.get("schema_version"), "1.0", "schema_version")
    _require_equal(config.get("experiment_id"), EXPECTED_EXPERIMENT_ID, "experiment_id")
    _require_equal(config.get("status"), "prepared_pending_root_approval", "status")
    _require_equal(config.get("seed"), 20260821, "seed")

    data = config["data_contract"]
    _require_equal(data["context_hours"], 48, "data_contract.context_hours")
    _require_equal(data["context_rows_10m"], CONTEXT_ROWS, "data_contract.context_rows_10m")
    _require_equal(data["wave_rows_20m"], WAVE_ROWS, "data_contract.wave_rows_20m")
    _require_equal(data["atmos_rows_10m"], ATMOS_ROWS, "data_contract.atmos_rows_10m")
    _require_equal(tuple(data["official_leads_h"]), LEADS, "data_contract.official_leads_h")
    _require_equal(data["anchor_minimum_hs_m"], 1.5, "data_contract.anchor_minimum_hs_m")
    _require_equal(data["absolute_test_time_recovery"], False, "absolute_test_time_recovery")
    _require_equal(data["cross_case_context"], False, "cross_case_context")

    architecture = config["architecture"]
    frozen_architecture = PatchModelConfig(
        d_model=int(architecture["d_model"]),
        encoder_layers=int(architecture["encoder_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        wave_patch_steps=int(architecture["wave_patch_steps"]),
        wave_stride_steps=int(architecture["wave_stride_steps"]),
        atmos_patch_steps=int(architecture["atmos_patch_steps"]),
        atmos_stride_steps=int(architecture["atmos_stride_steps"]),
        robust_scale_floor=float(architecture["robust_scale_floor"]),
    )
    frozen_architecture.validate()
    _require_equal(architecture["wave_patch_hours"], 2, "architecture.wave_patch_hours")
    _require_equal(architecture["wave_stride_hours"], 1, "architecture.wave_stride_hours")
    _require_equal(architecture["atmos_patch_hours"], 2, "architecture.atmos_patch_hours")
    _require_equal(architecture["atmos_stride_hours"], 1, "architecture.atmos_stride_hours")
    _require_equal(architecture["dense_72_step_auxiliary"], False, "dense_72_step_auxiliary")
    _require_equal(architecture["external_pretrained_weights"], False, "external weights")

    sampling = config["sampling"]
    _require_equal(
        sampling["storm_episode_definition"],
        "station-local contiguous 20-minute anchors with current hs >= 1.5m",
        "sampling.storm_episode_definition",
    )
    _require_equal(sampling["anchor_weight"], "1/sqrt(outer-train episode size)", "anchor_weight")
    _require_equal(sampling["normalize_weight_mean_to_one"], True, "weight normalization")

    loss = config["loss"]
    _require_equal(loss["target"], "official_6lead_future_hs_minus_current_hs", "loss.target")
    _require_equal(loss["official_lead_weight"], 1.0, "loss.official_lead_weight")
    _require_equal(loss["dense_path_auxiliary_weight"], 0.0, "dense auxiliary weight")

    training = config["training"]
    _require_equal(training["optimizer"], "AdamW", "training.optimizer")
    _require_equal(training["learning_rate"], 0.0003, "training.learning_rate")
    _require_equal(training["weight_decay"], 0.0002, "training.weight_decay")
    _require_equal(training["batch_size"], 256, "training.batch_size")
    _require_equal(training["maximum_epochs"], 50, "training.maximum_epochs")
    _require_equal(training["patience"], 8, "training.patience")
    _require_equal(training["precision"], "bf16_amp_on_cuda", "training.precision")
    _require_equal(training["gradient_clip_norm"], 1.0, "training.gradient_clip_norm")
    _require_equal(
        tuple(training["fixed_seeds"]),
        (20260821, 20260822, 20260823),
        "training.fixed_seeds",
    )
    _require_equal(training["hyperparameter_search"], False, "training.hyperparameter_search")

    blend = config["blend"]
    _require_equal(
        tuple(blend["protected_exact_incumbent_leads_h"]), PROTECTED_LEADS, "protected leads"
    )
    _require_equal(tuple(blend["active_leads_h"]), ACTIVE_BLEND_LEADS, "active blend leads")
    _require_equal(blend["patch_model_weight"], 0.2, "blend.patch_model_weight")

    validation = config["validation"]
    _require_equal(validation["case_gap_hours"], 78, "validation.case_gap_hours")
    _require_equal(validation["embargo_hours"], 78, "validation.embargo_hours")
    _require_equal(validation["inner_validation_days"], 45, "validation.inner_validation_days")
    _require_equal(validation["split_unit"], "station_storm_episode_and_78h_case", "split_unit")
    _require_equal(validation["phase_shift_lattice_allowed"], False, "phase lattice prohibition")
    forbidden_validation_keys = {
        key for key in validation if "phase" in key.lower() or "lattice" in key.lower()
    }
    if forbidden_validation_keys != {"phase_shift_lattice_allowed"}:
        raise ValueError(
            f"phase-shift lattice fields are forbidden: {sorted(forbidden_validation_keys)}"
        )

    gate = config["gate"]
    _require_equal(gate["incumbent_local_rmse"], 0.7801609198910191, "gate incumbent")
    _require_equal(gate["maximum_candidate_rmse"], 0.7701609198910191, "gate candidate")
    _require_equal(gate["paired_case_bootstrap_ci90_upper_below_zero"], True, "gate bootstrap")
    _require_equal(gate["lead_18_non_degrading"], True, "gate lead 18")
    _require_equal(gate["lead_24_non_degrading"], True, "gate lead 24")
    _require_equal(gate["maximum_station_rmse_degradation"], 0.01, "gate station")

    execution = config["execution"]
    _require_equal(execution["dry_run_only_until_root_approval"], True, "dry-run authorization")
    _require_equal(execution["runner_mode_now"], "dry-run", "runner_mode_now")
    _require_equal(
        execution["future_full_authorization_token"],
        FULL_AUTHORIZATION_TOKEN,
        "future authorization token",
    )
    _require_equal(execution["outer_validation_labels_may_open_now"], False, "outer label gate")
    _require_equal(execution["checkpoint_may_be_written_now"], False, "checkpoint gate")
    _require_equal(execution["hyperparameter_search"], False, "hyperparameter search")

    prohibitions = config["prohibitions"]
    for field in (
        "future_test_values",
        "external_observations",
        "submission_write_or_upload",
        "frozen_artifact_mutation",
        "phase_shift_lattice",
    ):
        _require_equal(prohibitions[field], True, f"prohibitions.{field}")

    verified: dict[str, str] = {}
    if verify_frozen_files:
        if root is None:
            raise ValueError("root is required when verify_frozen_files=True")
        repository = Path(root).resolve()
        for name, record in config["frozen_inputs"].items():
            relative = Path(record["path"])
            if relative.is_absolute():
                raise ValueError(f"frozen path must be repository-relative: {relative}")
            path = repository / relative
            if not path.is_file():
                raise FileNotFoundError(f"frozen artifact is missing: {relative.as_posix()}")
            actual = sha256_file(path)
            _require_equal(actual, record["sha256"], f"frozen_inputs.{name}.sha256")
            verified[name] = actual

    return {
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "architecture": {
            "wave_patches": frozen_architecture.wave_patch_count,
            "atmos_patches": frozen_architecture.atmos_patch_count,
            "d_model": frozen_architecture.d_model,
            "encoder_layers": frozen_architecture.encoder_layers,
        },
        "frozen_sha256": verified,
        "outer_validation_labels_opened": False,
        "full_training_authorized": False,
    }


def _nanmedian(values: torch.Tensor, *, dim: int, keepdim: bool = True) -> torch.Tensor:
    result = torch.nanmedian(values, dim=dim, keepdim=keepdim).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _robust_center_scale(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale_floor: float,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nan = torch.full_like(values, float("nan"))
    finite_values = torch.where(mask, values, nan)
    location = _nanmedian(finite_values, dim=1)
    absolute_deviation = torch.where(mask, torch.abs(values - location), nan)
    scale = 1.4826 * _nanmedian(absolute_deviation, dim=1)
    fallback = torch.sqrt(
        torch.sum(
            torch.where(mask, torch.square(values - location), torch.zeros_like(values)),
            dim=1,
            keepdim=True,
        )
        / torch.clamp(mask.sum(dim=1, keepdim=True).to(values.dtype), min=1.0)
    )
    scale = torch.where(scale >= scale_floor, scale, fallback)
    scale = torch.where(
        torch.isfinite(scale) & (scale >= scale_floor), scale, torch.full_like(scale, scale_floor)
    )
    used_location = location if center else torch.zeros_like(location)
    normalized = torch.where(mask, (values - used_location) / scale, torch.zeros_like(values))
    return normalized, used_location, scale


def validate_raw_context(raw: torch.Tensor) -> None:
    if raw.ndim != 3 or raw.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError(f"raw context must have shape (batch, {CONTEXT_ROWS}, {len(RAW_COLUMNS)})")
    if not torch.isfinite(raw[:, -1, 0]).all():
        raise ValueError("current hs must be finite")
    odd_wave = raw[:, 1::2, :4]
    if torch.isfinite(odd_wave).any():
        raise ValueError("wave values on structural 10-minute rows are forbidden")


def extract_past_context(values: np.ndarray, anchor_position: int) -> np.ndarray:
    """Return exactly 48 hours ending at an anchor; later rows can never enter."""

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(RAW_COLUMNS):
        raise ValueError("values must have shape (time, 10 raw channels)")
    stop = int(anchor_position) + 1
    start = stop - CONTEXT_ROWS
    if start < 0 or stop > len(array):
        raise ValueError("anchor does not have a complete 48-hour past context")
    return np.array(array[start:stop], copy=True)


def prepare_streams(raw: torch.Tensor, config: PatchModelConfig | None = None) -> PreparedStreams:
    """Build native-cadence streams using past-only, casewise robust normalization."""

    cfg = config or PatchModelConfig()
    cfg.validate()
    validate_raw_context(raw)
    raw = raw.to(dtype=torch.float32)

    wave_raw = raw[:, ::2, :4]
    atmos_raw = raw[:, :, 4:]
    if wave_raw.shape[1] != WAVE_ROWS:
        raise AssertionError("native wave extraction did not produce 145 rows")

    current_hs = wave_raw[:, -1, 0]
    wave_continuous = wave_raw[:, :, :3]
    wave_mask = torch.isfinite(wave_raw)
    hs_delta = wave_continuous[:, :, :1] - current_hs[:, None, None]
    hs_normalized, _, hs_scale = _robust_center_scale(
        hs_delta,
        wave_mask[:, :, :1],
        scale_floor=cfg.robust_scale_floor,
        center=False,
    )
    other_wave, _, _ = _robust_center_scale(
        wave_continuous[:, :, 1:],
        wave_mask[:, :, 1:3],
        scale_floor=cfg.robust_scale_floor,
    )
    wave_angle = torch.deg2rad(wave_raw[:, :, 3])
    wave_direction = torch.stack([torch.sin(wave_angle), torch.cos(wave_angle)], dim=-1)
    wave_direction = torch.where(
        wave_mask[:, :, 3:4], wave_direction, torch.zeros_like(wave_direction)
    )
    wave_time = torch.linspace(-1.0, 0.0, WAVE_ROWS, device=raw.device, dtype=raw.dtype)
    wave_time = wave_time.view(1, WAVE_ROWS, 1).expand(len(raw), -1, -1)
    wave = torch.cat(
        [
            hs_normalized,
            other_wave,
            wave_direction,
            wave_mask.to(raw.dtype),
            wave_time,
        ],
        dim=-1,
    )

    atmos_mask = torch.isfinite(atmos_raw)
    atmos_continuous = atmos_raw[:, :, [0, 1, 3, 4, 5]]
    atmos_continuous_mask = atmos_mask[:, :, [0, 1, 3, 4, 5]]
    atmos_normalized, _, _ = _robust_center_scale(
        atmos_continuous,
        atmos_continuous_mask,
        scale_floor=cfg.robust_scale_floor,
    )
    wind_angle = torch.deg2rad(atmos_raw[:, :, 2])
    wind_direction = torch.stack([torch.sin(wind_angle), torch.cos(wind_angle)], dim=-1)
    wind_direction = torch.where(
        atmos_mask[:, :, 2:3], wind_direction, torch.zeros_like(wind_direction)
    )
    atmos_time = torch.linspace(-1.0, 0.0, ATMOS_ROWS, device=raw.device, dtype=raw.dtype)
    atmos_time = atmos_time.view(1, ATMOS_ROWS, 1).expand(len(raw), -1, -1)
    atmos = torch.cat(
        [atmos_normalized, wind_direction, atmos_mask.to(raw.dtype), atmos_time], dim=-1
    )

    if wave.shape[-1] != 10 or atmos.shape[-1] != 14:
        raise AssertionError("prepared stream channel contract changed")
    return PreparedStreams(wave, atmos, current_hs, hs_scale[:, 0, 0])


def _patch(values: torch.Tensor, *, length: int, stride: int) -> torch.Tensor:
    patches = values.unfold(dimension=1, size=length, step=stride)
    return patches.contiguous().reshape(len(values), patches.shape[1], -1)


class TwoStreamRevINPatchTransformer(nn.Module):
    """Small two-stream patch transformer with six direct residual queries."""

    def __init__(self, config: PatchModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatchModelConfig()
        self.config.validate()
        cfg = self.config
        self.wave_projection = nn.Linear(cfg.wave_patch_steps * 10, cfg.d_model)
        self.atmos_projection = nn.Linear(cfg.atmos_patch_steps * 14, cfg.d_model)
        self.wave_position = nn.Parameter(torch.zeros(1, cfg.wave_patch_count, cfg.d_model))
        self.atmos_position = nn.Parameter(torch.zeros(1, cfg.atmos_patch_count, cfg.d_model))
        self.stream_embedding = nn.Parameter(torch.zeros(2, cfg.d_model))
        self.horizon_queries = nn.Parameter(torch.zeros(1, len(LEADS), cfg.d_model))
        self.station_embedding = nn.Embedding(len(STATIONS), cfg.d_model)
        self.amplitude_projection = nn.Sequential(
            nn.Linear(2, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.encoder_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.residual_head = nn.Linear(cfg.d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.wave_position, std=0.02)
        nn.init.normal_(self.atmos_position, std=0.02)
        nn.init.normal_(self.stream_embedding, std=0.02)
        nn.init.normal_(self.horizon_queries, std=0.02)

    def forward(self, raw: torch.Tensor, station_code: torch.Tensor) -> torch.Tensor:
        if station_code.ndim != 1 or len(station_code) != len(raw):
            raise ValueError("station_code must align one-to-one with raw contexts")
        if station_code.min().item() < 0 or station_code.max().item() >= len(STATIONS):
            raise ValueError("station_code is outside [0, 2]")
        streams = prepare_streams(raw, self.config)
        wave = self.wave_projection(
            _patch(
                streams.wave,
                length=self.config.wave_patch_steps,
                stride=self.config.wave_stride_steps,
            )
        )
        atmos = self.atmos_projection(
            _patch(
                streams.atmos,
                length=self.config.atmos_patch_steps,
                stride=self.config.atmos_stride_steps,
            )
        )
        wave = wave + self.wave_position + self.stream_embedding[0]
        atmos = atmos + self.atmos_position + self.stream_embedding[1]
        amplitude = torch.stack(
            [torch.log1p(streams.current_hs.clamp_min(0.0)), torch.log(streams.hs_scale)], dim=-1
        )
        case_context = self.station_embedding(station_code) + self.amplitude_projection(amplitude)
        queries = self.horizon_queries.expand(len(raw), -1, -1) + case_context[:, None, :]
        tokens = torch.cat(
            [wave + case_context[:, None, :], atmos + case_context[:, None, :], queries], dim=1
        )
        encoded = self.encoder(tokens)
        normalized_delta = self.residual_head(self.output_norm(encoded[:, -len(LEADS) :])).squeeze(
            -1
        )
        return normalized_delta * streams.hs_scale[:, None]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def weighted_official_mse(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    case_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction_delta.shape != target_delta.shape or prediction_delta.ndim != 2:
        raise ValueError("prediction and target must share shape (cases, 6)")
    if prediction_delta.shape[1] != len(LEADS):
        raise ValueError("only the six official leads are allowed")
    if case_weight.ndim != 1 or len(case_weight) != len(prediction_delta):
        raise ValueError("case weights must align with cases")
    if not torch.isfinite(target_delta).all() or not torch.isfinite(case_weight).all():
        raise ValueError("targets and weights must be finite")
    normalized_weight = case_weight / case_weight.mean()
    return torch.mean(normalized_weight[:, None] * torch.square(prediction_delta - target_delta))


def blend_long_leads(
    incumbent: np.ndarray,
    patch_prediction: np.ndarray,
    *,
    patch_weight: float = 0.2,
) -> np.ndarray:
    base = np.asarray(incumbent, dtype=np.float64)
    candidate = np.asarray(patch_prediction, dtype=np.float64)
    if base.shape != candidate.shape or base.ndim != 2 or base.shape[1] != len(LEADS):
        raise ValueError("predictions must be aligned arrays with six lead columns")
    if patch_weight != 0.2:
        raise ValueError("P3 RevIN Patch v1 patch weight is frozen at 0.2")
    output = base.copy()
    active = [LEADS.index(lead) for lead in ACTIVE_BLEND_LEADS]
    output[:, active] = (1.0 - patch_weight) * base[:, active] + patch_weight * candidate[:, active]
    protected = [LEADS.index(lead) for lead in PROTECTED_LEADS]
    if not np.array_equal(output[:, protected], base[:, protected]):
        raise AssertionError("3/6/9h exact incumbent protection failed")
    return output


def assign_storm_episodes(anchors: pd.DataFrame) -> pd.DataFrame:
    required = {"anchor_id", "station", "anchor_time", "current_hs"}
    if not required.issubset(anchors.columns):
        raise ValueError(f"anchor metadata is missing: {sorted(required - set(anchors.columns))}")
    result = anchors.copy()
    result["anchor_time"] = pd.to_datetime(result["anchor_time"], utc=True)
    if result["current_hs"].lt(1.5).any():
        raise ValueError("episode table contains an ineligible anchor below 1.5m")
    episode = np.empty(len(result), dtype=np.int64)
    next_episode = 0
    for _, group in result.groupby("station", sort=True, observed=True):
        ordered = group.sort_values("anchor_time")
        delta = ordered["anchor_time"].diff()
        start = delta.isna() | delta.ne(pd.Timedelta(minutes=20))
        local = start.cumsum().to_numpy(dtype=np.int64) - 1 + next_episode
        episode[ordered.index.to_numpy(dtype=np.int64)] = local
        next_episode = int(local.max()) + 1
    result["episode_id"] = episode
    return result


def assign_storm_episodes_from_wave(
    anchors: pd.DataFrame,
    wave: pd.DataFrame,
) -> pd.DataFrame:
    """Attach episodes defined on the complete observed wave stream.

    Episode boundaries must not depend on whether a future six-lead target happens to be
    available. Defining them on eligible anchors alone could split one physical storm when
    target validity has a hole, so this path uses only current/past raw ``hs`` observations.
    """

    required_anchor = {"anchor_id", "station", "anchor_time", "current_hs"}
    required_wave = {"station", "time", "hs"}
    if not required_anchor.issubset(anchors.columns):
        raise ValueError(
            f"anchor metadata is missing: {sorted(required_anchor - set(anchors.columns))}"
        )
    if not required_wave.issubset(wave.columns):
        raise ValueError(f"wave data is missing: {sorted(required_wave - set(wave.columns))}")

    result = anchors.copy()
    result["anchor_time"] = pd.to_datetime(result["anchor_time"], utc=True)
    if result["current_hs"].lt(1.5).any():
        raise ValueError("episode table contains an ineligible anchor below 1.5m")

    source = wave.loc[:, ["station", "time", "hs"]].copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    if source.duplicated(["station", "time"]).any():
        raise ValueError("wave data contains duplicate station/time keys")

    high_rows: list[pd.DataFrame] = []
    next_episode = 0
    for _, group in source.groupby("station", sort=True, observed=True):
        ordered = group.sort_values("time").copy()
        high = ordered["hs"].ge(1.5) & ordered["hs"].notna()
        contiguous = ordered["time"].diff().eq(pd.Timedelta(minutes=20))
        previous_high = high.shift(fill_value=False)
        start = high & (~previous_high | ~contiguous)
        local_episode = start.cumsum().astype(np.int64) - 1 + next_episode
        selected = ordered.loc[high, ["station", "time", "hs"]].copy()
        selected["episode_id"] = local_episode.loc[high].to_numpy(dtype=np.int64)
        high_rows.append(selected)
        if high.any():
            next_episode = int(local_episode.loc[high].max()) + 1

    mapping = pd.concat(high_rows, ignore_index=True).rename(
        columns={"time": "anchor_time", "hs": "raw_current_hs"}
    )
    result = result.merge(
        mapping,
        on=["station", "anchor_time"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["episode_id"].isna().any():
        raise ValueError("eligible anchor could not be mapped to a raw-wave storm episode")
    if not np.allclose(
        result["current_hs"].to_numpy(dtype=np.float64),
        result["raw_current_hs"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("anchor cache current_hs differs from immutable raw-wave hs")
    result["episode_id"] = result["episode_id"].astype(np.int64)
    return result.drop(columns="raw_current_hs")


def event_balanced_weights(anchors: pd.DataFrame, anchor_ids: Iterable[int]) -> np.ndarray:
    ids = np.asarray(list(anchor_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("event-balanced weighting received no anchors")
    lookup = anchors.set_index("anchor_id")
    selected = lookup.loc[ids, ["station", "episode_id"]]
    event_size = selected.groupby(["station", "episode_id"], observed=True)["episode_id"].transform(
        "size"
    )
    weight = 1.0 / np.sqrt(event_size.to_numpy(dtype=np.float64))
    weight /= weight.mean()
    return weight


def _timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def select_episode_disjoint_cases(
    anchors: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    gap_hours: int = 78,
) -> np.ndarray:
    if gap_hours != 78:
        raise ValueError("P3 case gap is frozen at 78 hours")
    start_ts = _timestamp(start)
    end_ts = _timestamp(end)
    chosen: list[int] = []
    for _, group in anchors.groupby("station", sort=True, observed=True):
        eligible = group.loc[
            group["anchor_time"].ge(start_ts) & group["anchor_time"].lt(end_ts)
        ].sort_values("anchor_time")
        next_time: pd.Timestamp | None = None
        used_episodes: set[int] = set()
        for row in eligible.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            episode_id = int(row.episode_id)
            if episode_id in used_episodes:
                continue
            if next_time is None or timestamp >= next_time:
                chosen.append(int(row.anchor_id))
                used_episodes.add(episode_id)
                next_time = timestamp + pd.Timedelta(hours=gap_hours)
    return np.asarray(sorted(chosen), dtype=np.int64)


def _assert_fold_disjoint(
    anchors: pd.DataFrame, train_ids: np.ndarray, validation_ids: np.ndarray
) -> None:
    lookup = anchors.set_index("anchor_id")
    train = lookup.loc[train_ids]
    validation = lookup.loc[validation_ids]
    train_episode = set(
        zip(train["station"].astype(str), train["episode_id"].astype(int), strict=True)
    )
    validation_episode = set(
        zip(validation["station"].astype(str), validation["episode_id"].astype(int), strict=True)
    )
    if train_episode.intersection(validation_episode):
        raise AssertionError("a storm episode appears in both outer train and validation")
    embargo = pd.Timedelta(hours=78)
    for station, current in validation.groupby("station", observed=True):
        train_time = train.loc[train["station"].eq(station), "anchor_time"]
        if train_time.empty:
            continue
        for timestamp in current["anchor_time"]:
            distance = (train_time - pd.Timestamp(timestamp)).abs()
            if distance.lt(embargo).any():
                raise AssertionError("train/validation anchor distance is below 78 hours")


def build_episode_disjoint_folds(
    anchors: pd.DataFrame,
    *,
    windows: Sequence[Sequence[str]],
    embargo_hours: int = 78,
) -> tuple[EpisodeDisjointFold, ...]:
    if embargo_hours != 78:
        raise ValueError("P3 embargo is frozen at 78 hours")
    if "episode_id" not in anchors:
        raise ValueError("anchors must be assigned to storm episodes before splitting")
    folds: list[EpisodeDisjointFold] = []
    for item in windows:
        if len(item) != 3:
            raise ValueError("each validation window must contain name, start, end")
        name, start, end = map(str, item)
        validation_start = _timestamp(start)
        validation_end = _timestamp(end)
        validation_ids = select_episode_disjoint_cases(
            anchors, start=validation_start, end=validation_end, gap_hours=78
        )
        train_end = validation_start - pd.Timedelta(hours=embargo_hours)
        train_ids = anchors.loc[anchors["anchor_time"].lt(train_end), "anchor_id"].to_numpy(
            dtype=np.int64
        )
        validation_lookup = anchors.set_index("anchor_id").loc[validation_ids]
        validation_episodes = set(
            zip(
                validation_lookup["station"].astype(str),
                validation_lookup["episode_id"].astype(int),
                strict=True,
            )
        )
        train_lookup = anchors.set_index("anchor_id").loc[train_ids]
        keep = np.asarray(
            [
                (str(row.station), int(row.episode_id)) not in validation_episodes
                for row in train_lookup.itertuples()
            ],
            dtype=bool,
        )
        train_ids = train_ids[keep]
        if len(train_ids) == 0 or len(validation_ids) == 0:
            raise ValueError(f"empty train or validation cases in {name}")
        if np.intersect1d(train_ids, validation_ids).size:
            raise AssertionError(f"anchor id overlap in {name}")
        _assert_fold_disjoint(anchors, train_ids, validation_ids)
        folds.append(
            EpisodeDisjointFold(
                name=name,
                train_ids=train_ids,
                validation_ids=validation_ids,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return tuple(folds)


def build_episode_disjoint_folds_from_ids(
    anchors: pd.DataFrame,
    *,
    windows: Sequence[Sequence[str]],
    validation_ids_by_fold: Mapping[str, Iterable[int]],
    embargo_hours: int = 78,
) -> tuple[EpisodeDisjointFold, ...]:
    """Build rolling folds around pre-existing, key-only outer validation IDs.

    This is used when candidate predictions must pair exactly with a frozen incumbent OOF.
    Only its keys are consumed; prediction and target columns remain unopened.
    """

    if embargo_hours != 78:
        raise ValueError("P3 embargo is frozen at 78 hours")
    lookup = anchors.set_index("anchor_id")
    folds: list[EpisodeDisjointFold] = []
    expected_names = {str(item[0]) for item in windows}
    if set(validation_ids_by_fold) != expected_names:
        raise ValueError("frozen outer fold names do not match preregistered windows")
    for item in windows:
        if len(item) != 3:
            raise ValueError("each validation window must contain name, start, end")
        name, start, end = map(str, item)
        validation_start = _timestamp(start)
        validation_end = _timestamp(end)
        validation_ids = np.asarray(list(validation_ids_by_fold[name]), dtype=np.int64)
        if not len(validation_ids) or len(np.unique(validation_ids)) != len(validation_ids):
            raise ValueError(f"invalid frozen outer validation ids in {name}")
        validation = lookup.loc[validation_ids]
        if (
            not validation["anchor_time"].ge(validation_start).all()
            or not validation["anchor_time"].lt(validation_end).all()
        ):
            raise ValueError(f"frozen outer validation id is outside {name} window")
        for _, station_rows in validation.groupby("station", observed=True):
            gap = station_rows["anchor_time"].sort_values().diff().dropna()
            if not gap.ge(pd.Timedelta(hours=embargo_hours)).all():
                raise ValueError(f"frozen outer validation cases violate 78h spacing in {name}")

        train_end = validation_start - pd.Timedelta(hours=embargo_hours)
        train_ids = anchors.loc[anchors["anchor_time"].lt(train_end), "anchor_id"].to_numpy(
            dtype=np.int64
        )
        validation_episodes = set(
            zip(
                validation["station"].astype(str),
                validation["episode_id"].astype(int),
                strict=True,
            )
        )
        train = lookup.loc[train_ids]
        keep = np.asarray(
            [
                (str(row.station), int(row.episode_id)) not in validation_episodes
                for row in train.itertuples()
            ],
            dtype=bool,
        )
        train_ids = train_ids[keep]
        if not len(train_ids):
            raise ValueError(f"empty frozen-key outer train in {name}")
        _assert_fold_disjoint(anchors, train_ids, validation_ids)
        folds.append(
            EpisodeDisjointFold(
                name=name,
                train_ids=train_ids,
                validation_ids=np.sort(validation_ids),
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return tuple(folds)


def fold_coverage(anchors: pd.DataFrame, folds: Sequence[EpisodeDisjointFold]) -> dict[str, Any]:
    lookup = anchors.set_index("anchor_id")
    result: dict[str, Any] = {}
    for fold in folds:
        train = lookup.loc[fold.train_ids]
        validation = lookup.loc[fold.validation_ids]
        weights = event_balanced_weights(anchors, fold.train_ids)
        station_gaps: list[float] = []
        for station, current in validation.groupby("station", observed=True):
            train_time = train.loc[train["station"].eq(station), "anchor_time"]
            if train_time.empty:
                continue
            gap = (current["anchor_time"].min() - train_time.max()).total_seconds() / 3600.0
            station_gaps.append(float(gap))
        result[fold.name] = {
            "train_anchors": int(len(train)),
            "train_episodes": int(train.groupby(["station", "episode_id"], observed=True).ngroups),
            "validation_cases": int(len(validation)),
            "validation_cases_by_station": {
                str(key): int(value)
                for key, value in validation.groupby("station", observed=True).size().items()
            },
            "shared_station_episode_count": 0,
            "minimum_same_station_anchor_gap_hours": min(station_gaps),
            "event_weight_min": float(weights.min()),
            "event_weight_max": float(weights.max()),
            "event_weight_mean": float(weights.mean()),
        }
    return result


def build_inner_episode_split(
    anchors: pd.DataFrame,
    outer_train_ids: Iterable[int],
    *,
    validation_days: int = 45,
    embargo_hours: int = 78,
) -> InnerEpisodeSplit:
    """Build an epoch-selection split strictly inside one outer-training partition."""

    if validation_days != 45 or embargo_hours != 78:
        raise ValueError("P3 RevIN Patch v1 inner split is frozen at 45 days and 78 hours")
    ids = np.asarray(list(outer_train_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("outer training partition is empty")
    lookup = anchors.set_index("anchor_id")
    outer = lookup.loc[ids].sort_values("anchor_time")
    validation_end = pd.Timestamp(outer["anchor_time"].max()) + pd.Timedelta(minutes=20)
    validation_start = validation_end - pd.Timedelta(days=validation_days)
    outer_anchors = anchors.loc[anchors["anchor_id"].isin(ids)].copy()
    validation_ids = select_episode_disjoint_cases(
        outer_anchors,
        start=validation_start,
        end=validation_end,
        gap_hours=embargo_hours,
    )
    inner_train_end = validation_start - pd.Timedelta(hours=embargo_hours)
    train_ids = outer.loc[outer["anchor_time"].lt(inner_train_end)].index.to_numpy(dtype=np.int64)

    validation = lookup.loc[validation_ids]
    validation_episodes = set(
        zip(
            validation["station"].astype(str),
            validation["episode_id"].astype(int),
            strict=True,
        )
    )
    train = lookup.loc[train_ids]
    keep = np.asarray(
        [
            (str(row.station), int(row.episode_id)) not in validation_episodes
            for row in train.itertuples()
        ],
        dtype=bool,
    )
    train_ids = train_ids[keep]
    if not len(train_ids) or not len(validation_ids):
        raise ValueError("inner episode split produced an empty partition")
    _assert_fold_disjoint(anchors, train_ids, validation_ids)
    if not np.isin(train_ids, ids).all() or not np.isin(validation_ids, ids).all():
        raise AssertionError("inner split escaped its outer-training partition")
    return InnerEpisodeSplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        validation_start=validation_start,
        validation_end=validation_end,
    )


def _seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _as_cpu_tensor(values: np.ndarray | torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.from_numpy(np.asarray(values))
    return tensor.detach().to(device="cpu", dtype=dtype)


def _predict_delta_batches(
    model: TwoStreamRevINPatchTransformer,
    raw: torch.Tensor,
    station_code: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            stop = min(start + batch_size, len(raw))
            batch_raw = raw[start:stop].to(device)
            batch_station = station_code[start:stop].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch_raw, batch_station)
            outputs.append(prediction.float().cpu())
    return torch.cat(outputs, dim=0)


def _fit_epochs(
    model: TwoStreamRevINPatchTransformer,
    raw: torch.Tensor,
    station_code: torch.Tensor,
    target_delta: torch.Tensor,
    case_weight: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
    epoch_callback: Any | None = None,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    model.train()
    for epoch in range(1, int(epochs) + 1):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(len(raw), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(raw[batch].to(device), station_code[batch].to(device))
                loss = weighted_official_mse(
                    prediction,
                    target_delta[batch].to(device),
                    case_weight[batch].to(device),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite P3 RevIN Patch training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
        if epoch_callback is not None:
            epoch_callback(epoch, model)
        model.train()


def select_epoch_on_inner_split(
    inner_train_raw: np.ndarray | torch.Tensor,
    inner_train_station: np.ndarray | torch.Tensor,
    inner_train_target_delta: np.ndarray | torch.Tensor,
    inner_train_weight: np.ndarray | torch.Tensor,
    inner_validation_raw: np.ndarray | torch.Tensor,
    inner_validation_station: np.ndarray | torch.Tensor,
    inner_validation_target_delta: np.ndarray | torch.Tensor,
    *,
    seed: int,
    device: str | torch.device,
    maximum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> EpochSelection:
    """Choose an epoch using only an episode-disjoint inner validation set."""

    selected_device = torch.device(device)
    train_raw = _as_cpu_tensor(inner_train_raw, dtype=torch.float32)
    train_station = _as_cpu_tensor(inner_train_station, dtype=torch.long)
    train_target = _as_cpu_tensor(inner_train_target_delta, dtype=torch.float32)
    train_weight = _as_cpu_tensor(inner_train_weight, dtype=torch.float32)
    validation_raw = _as_cpu_tensor(inner_validation_raw, dtype=torch.float32)
    validation_station = _as_cpu_tensor(inner_validation_station, dtype=torch.long)
    validation_target = _as_cpu_tensor(inner_validation_target_delta, dtype=torch.float32)
    if not len(train_raw) or not len(validation_raw):
        raise ValueError("inner epoch selection received an empty partition")

    _seed_torch(seed, selected_device)
    model = TwoStreamRevINPatchTransformer().to(selected_device)
    history: list[float] = []
    best_rmse = float("inf")
    best_epoch = 0

    def evaluate(epoch: int, current_model: TwoStreamRevINPatchTransformer) -> None:
        nonlocal best_epoch, best_rmse
        prediction = _predict_delta_batches(
            current_model,
            validation_raw,
            validation_station,
            device=selected_device,
            batch_size=batch_size,
        )
        rmse = float(torch.sqrt(torch.mean(torch.square(prediction - validation_target))).item())
        if not np.isfinite(rmse):
            raise RuntimeError("non-finite inner validation RMSE")
        history.append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = int(epoch)
        if epoch - best_epoch >= patience:
            raise StopIteration

    try:
        _fit_epochs(
            model,
            train_raw,
            train_station,
            train_target,
            train_weight,
            device=selected_device,
            epochs=maximum_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            seed=seed,
            epoch_callback=evaluate,
        )
    except StopIteration:
        pass
    if best_epoch < 1:
        raise RuntimeError("inner epoch selection did not complete one epoch")
    return EpochSelection(best_epoch, len(history), best_rmse, tuple(history))


def refit_fixed_epoch_and_predict(
    train_raw: np.ndarray | torch.Tensor,
    train_station: np.ndarray | torch.Tensor,
    train_target_delta: np.ndarray | torch.Tensor,
    train_weight: np.ndarray | torch.Tensor,
    prediction_raw: np.ndarray | torch.Tensor,
    prediction_station: np.ndarray | torch.Tensor,
    *,
    seed: int,
    device: str | torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> np.ndarray:
    """Refit on the entire outer train for the already-selected epoch count."""

    if epochs < 1:
        raise ValueError("fixed refit epoch must be positive")
    selected_device = torch.device(device)
    raw = _as_cpu_tensor(train_raw, dtype=torch.float32)
    station = _as_cpu_tensor(train_station, dtype=torch.long)
    target = _as_cpu_tensor(train_target_delta, dtype=torch.float32)
    weight = _as_cpu_tensor(train_weight, dtype=torch.float32)
    predict_raw = _as_cpu_tensor(prediction_raw, dtype=torch.float32)
    predict_station = _as_cpu_tensor(prediction_station, dtype=torch.long)
    _seed_torch(seed, selected_device)
    model = TwoStreamRevINPatchTransformer().to(selected_device)
    _fit_epochs(
        model,
        raw,
        station,
        target,
        weight,
        device=selected_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
    )
    prediction = _predict_delta_batches(
        model,
        predict_raw,
        predict_station,
        device=selected_device,
        batch_size=batch_size,
    )
    return prediction.numpy().astype(np.float32, copy=False)


def bounded_training_protocol_smoke(seed: int = 20260821) -> dict[str, Any]:
    """Exercise inner epoch selection and fixed-epoch refit on CPU without files or labels."""

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = build_synthetic_context(batch=18, seed=seed).numpy()
        station = np.resize(np.arange(3, dtype=np.int64), 18)
        target = np.column_stack(
            [0.02 * (number + 1) * np.linspace(0.5, 1.5, 18) for number in range(6)]
        ).astype(np.float32)
        selection = select_epoch_on_inner_split(
            raw[:8],
            station[:8],
            target[:8],
            np.ones(8, dtype=np.float32),
            raw[8:12],
            station[8:12],
            target[8:12],
            seed=seed,
            device="cpu",
            maximum_epochs=2,
            patience=1,
            batch_size=4,
            learning_rate=3e-4,
            weight_decay=2e-4,
            gradient_clip_norm=1.0,
        )
        prediction = refit_fixed_epoch_and_predict(
            raw[:12],
            station[:12],
            target[:12],
            np.ones(12, dtype=np.float32),
            raw[12:],
            station[12:],
            seed=seed,
            device="cpu",
            epochs=selection.selected_epoch,
            batch_size=4,
            learning_rate=3e-4,
            weight_decay=2e-4,
            gradient_clip_norm=1.0,
        )
        return {
            "device": "cpu",
            "selected_epoch": int(selection.selected_epoch),
            "epochs_ran": int(selection.epochs_ran),
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(np.isfinite(prediction).all()),
            "checkpoint_written": False,
            "outer_labels_opened": False,
        }
    finally:
        torch.set_num_threads(previous_threads)


def build_synthetic_context(batch: int = 4, *, seed: int = 20260821) -> torch.Tensor:
    """Create a bounded structural context for CPU forward/backward smoke tests."""

    rng = np.random.default_rng(seed)
    raw = np.empty((batch, CONTEXT_ROWS, len(RAW_COLUMNS)), dtype=np.float32)
    raw[:] = np.nan
    time = np.arange(CONTEXT_ROWS, dtype=np.float32)
    for number in range(batch):
        wave_time = time[::2]
        raw[number, ::2, 0] = 2.0 + 0.2 * np.sin(wave_time / 17.0 + number)
        raw[number, ::2, 1] = 7.0 + 0.3 * np.cos(wave_time / 31.0)
        raw[number, ::2, 2] = 1.7 * raw[number, ::2, 0]
        raw[number, ::2, 3] = np.mod(20.0 + wave_time * 1.3, 360.0)
        raw[number, :, 4] = 8.0 + np.sin(time / 23.0) + rng.normal(0, 0.05, len(time))
        raw[number, :, 5] = raw[number, :, 4] + 1.5
        raw[number, :, 6] = np.mod(180.0 + time * 0.7, 360.0)
        raw[number, :, 7] = 15.0 + np.sin(time / 73.0)
        raw[number, :, 8] = 70.0 + 4.0 * np.cos(time / 41.0)
        raw[number, :, 9] = 1012.0 - 0.01 * time
    if batch:
        raw[0, 20:30, 4] = np.nan
    if batch > 1:
        raw[1, 40:45:2, 1] = np.nan
    return torch.from_numpy(raw)


def bounded_cpu_backward_smoke(seed: int = 20260821) -> dict[str, Any]:
    torch.manual_seed(seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = TwoStreamRevINPatchTransformer()
        model.train()
        raw = build_synthetic_context(batch=4, seed=seed)
        station = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        prediction = model(raw, station)
        target = torch.zeros_like(prediction)
        weight = torch.tensor([1.0, 0.75, 1.25, 1.0], dtype=torch.float32)
        loss = weighted_official_mse(prediction, target, weight)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("CPU backward smoke produced missing or non-finite gradients")
        return {
            "device": "cpu",
            "batch": 4,
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(torch.isfinite(prediction).all()),
            "loss_finite": bool(torch.isfinite(loss)),
            "gradient_tensor_count": int(len(gradients)),
            "trainable_parameters": int(model.trainable_parameter_count),
            "wave_patch_count": model.config.wave_patch_count,
            "atmos_patch_count": model.config.atmos_patch_count,
            "dense_72_step_auxiliary": False,
        }
    finally:
        torch.set_num_threads(previous_threads)


__all__ = [
    "ACTIVE_BLEND_LEADS",
    "ATMOS_ROWS",
    "CONTEXT_ROWS",
    "EpochSelection",
    "EpisodeDisjointFold",
    "FULL_AUTHORIZATION_TOKEN",
    "PatchModelConfig",
    "PreparedStreams",
    "InnerEpisodeSplit",
    "PROTECTED_LEADS",
    "RAW_COLUMNS",
    "TwoStreamRevINPatchTransformer",
    "WAVE_ROWS",
    "assign_storm_episodes",
    "assign_storm_episodes_from_wave",
    "blend_long_leads",
    "bounded_cpu_backward_smoke",
    "bounded_training_protocol_smoke",
    "build_episode_disjoint_folds",
    "build_episode_disjoint_folds_from_ids",
    "build_inner_episode_split",
    "build_synthetic_context",
    "event_balanced_weights",
    "extract_past_context",
    "fold_coverage",
    "load_preregistration",
    "prepare_streams",
    "refit_fixed_epoch_and_predict",
    "select_episode_disjoint_cases",
    "select_epoch_on_inner_split",
    "sha256_file",
    "validate_preregistration",
    "validate_raw_context",
    "weighted_official_mse",
]
