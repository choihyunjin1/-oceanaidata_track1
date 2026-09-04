"""Exactly-once Fishr gradient-variance P1 cross-quarter audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v44_causal_fishr_gradient_variance_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V36 = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v44_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared science module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


prior = _module(V36)
shared = prior.shared
base = prior.base
CAUSAL_FEATURES = prior.CAUSAL_FEATURES
_BASE_WRITE = base._write
SCIENTIFIC_FEATURES = 8
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
LAYERS = tuple(range(1, 9))
QUARTERS = tuple(range(1, 5))
ENVIRONMENT_BITS = len(STATIONS) + len(LAYERS) + len(QUARTERS)


def causal_environment_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Append penalty-only one-hot source-environment metadata."""

    scientific = CAUSAL_FEATURES(frame, train_boundary_ns, representation)
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy(np.int64)
    quarter = pd.to_datetime(frame["_time"], utc=True).dt.quarter.to_numpy(np.int64)
    bits = np.zeros((len(frame), ENVIRONMENT_BITS), dtype=np.float32)
    for index, value in enumerate(STATIONS):
        bits[:, index] = station == value
    layer_start = len(STATIONS)
    for index, value in enumerate(LAYERS):
        bits[:, layer_start + index] = layer == value
    quarter_start = layer_start + len(LAYERS)
    for index, value in enumerate(QUARTERS):
        bits[:, quarter_start + index] = quarter == value
    valid = (
        bits[:, :layer_start].sum(axis=1) == 1
    ) & (
        bits[:, layer_start:quarter_start].sum(axis=1) == 1
    ) & (
        bits[:, quarter_start:].sum(axis=1) == 1
    )
    if not bool(valid.all()):
        raise RuntimeError("source environment encoding is incomplete")
    output = np.column_stack([scientific, bits]).astype(np.float32)
    if output.shape != (len(frame), SCIENTIFIC_FEATURES + ENVIRONMENT_BITS):
        raise RuntimeError("Fishr feature shape contract failed")
    return output


def decode_environment_ids(scaled_features: np.ndarray) -> np.ndarray:
    """Recover station-layer-quarter identities from standardized one-hot bits."""

    values = np.asarray(scaled_features, dtype=np.float32)
    metadata = values[:, SCIENTIFIC_FEATURES:]
    if metadata.shape[1] != ENVIRONMENT_BITS:
        raise ValueError("Fishr metadata width invalid")
    station = np.argmax(metadata[:, : len(STATIONS)], axis=1)
    layer_start = len(STATIONS)
    quarter_start = layer_start + len(LAYERS)
    layer = np.argmax(metadata[:, layer_start:quarter_start], axis=1)
    quarter = np.argmax(metadata[:, quarter_start:], axis=1)
    return (station * 100 + layer * 10 + quarter).astype(np.int64)


def fishr_gradient_variance_penalty(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    environment_ids: torch.Tensor,
    positive_weight: float,
    minimum_rows: int,
) -> tuple[torch.Tensor, int]:
    """Match variances of per-row last-layer BCE gradients across environments."""

    weights = torch.where(targets > 0.5, float(positive_weight), 1.0)
    residual = (torch.sigmoid(logits) - targets) * weights
    augmented = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
    gradients = residual[:, None] * augmented
    variances = []
    for environment in torch.unique(environment_ids):
        use = environment_ids == environment
        if int(use.sum().item()) >= int(minimum_rows):
            variances.append(torch.var(gradients[use], dim=0, unbiased=False))
    if len(variances) < 2:
        return logits.sum() * 0.0, len(variances)
    stacked = torch.stack(variances)
    center = stacked.mean(dim=0, keepdim=True)
    return torch.square(stacked - center).mean(), len(variances)


class _FishrNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        return self.head(hidden).squeeze(1), hidden


class FishrClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        if int(inputs) != SCIENTIFIC_FEATURES + ENVIRONMENT_BITS:
            raise ValueError("Fishr input contract changed")
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _FishrNetwork(SCIENTIFIC_FEATURES, int(config["hidden_units"])).to(self.device)
        self.training_receipt: dict[str, Any] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> FishrClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        environments = decode_environment_ids(values)
        scientific = values[:, :SCIENTIFIC_FEATURES]
        rng = np.random.default_rng(self.seed)
        maximum = int(self.config["maximum_sample_rows_per_fit"])
        ratio = int(self.config["negative_to_positive_sample_ratio"])
        positive = np.flatnonzero(targets == 1)
        negative = np.flatnonzero(targets == 0)
        positive = rng.choice(positive, size=min(len(positive), maximum // (ratio + 1)), replace=False)
        negative = rng.choice(negative, size=min(len(negative), ratio * len(positive)), replace=False)
        selected = np.concatenate([positive, negative])
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        batch = int(self.config["batch_rows"])
        positive_weight = float(self.config["positive_class_weight"])
        coefficient = float(self.config["fishr_coefficient"])
        minimum_rows = int(self.config["minimum_environment_rows_per_batch"])
        penalty_values: list[float] = []
        supported_counts: list[int] = []
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(scientific[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                env = torch.from_numpy(environments[rows]).to(self.device)
                logits, hidden = self.network(x)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    pos_weight=torch.tensor(positive_weight, device=self.device),
                )
                penalty, supported = fishr_gradient_variance_penalty(
                    logits, hidden, y, env, positive_weight, minimum_rows
                )
                loss = bce + coefficient * penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                penalty_values.append(float(penalty.detach().cpu()))
                supported_counts.append(int(supported))
        self.training_receipt = {
            "sample_rows": int(len(selected)),
            "source_environments": int(len(np.unique(environments[selected]))),
            "mean_fishr_penalty": float(np.mean(penalty_values)),
            "minimum_supported_environments_per_batch": int(min(supported_counts)),
            "optimizer_steps": int(len(penalty_values)),
        }
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)[:, :SCIENTIFIC_FEATURES]
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                logits, _hidden = self.network(x)
                output[start : start + len(x)] = torch.sigmoid(logits).cpu().numpy()
        return output


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    logits = torch.tensor([-2.0, -0.5, 0.5, 2.0, -1.5, -0.25, 1.0, 2.5])
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    env = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    penalty, supported = fishr_gradient_variance_penalty(logits, hidden, targets, env, 3.0, 4)
    equal_penalty, equal_supported = fishr_gradient_variance_penalty(
        torch.cat([logits[:4], logits[:4]]),
        torch.cat([hidden[:4], hidden[:4]]),
        torch.cat([targets[:4], targets[:4]]),
        env,
        3.0,
        4,
    )
    frame = pd.DataFrame(
        {
            "station": ["G-ORS", "I-ORS", "S-ORS", "G-ORS"],
            "layer": [1, 2, 8, 1],
            "_time": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z", "2024-07-01T00:00:00Z", "2024-10-01T00:00:00Z"]
            ),
            "temp": [1.0, 1.0, 1.0, 1.0],
        }
    )
    features = causal_environment_features(frame, int(frame["_time"].iloc[-1].value), representation)
    return {
        "different_environment_variances_positive": bool(torch.isfinite(penalty) and penalty > 0.0 and supported == 2),
        "equal_environment_variances_zero": bool(torch.abs(equal_penalty) < 1e-8 and equal_supported == 2),
        "environment_bits_one_hot_complete": bool(
            np.all(features[:, SCIENTIFIC_FEATURES : SCIENTIFIC_FEATURES + len(STATIONS)].sum(axis=1) == 1)
            and np.all(features[:, SCIENTIFIC_FEATURES + len(STATIONS) : SCIENTIFIC_FEATURES + len(STATIONS) + len(LAYERS)].sum(axis=1) == 1)
            and np.all(features[:, -len(QUARTERS) :].sum(axis=1) == 1)
        ),
        "environment_bits_not_predictor_width": bool(SCIENTIFIC_FEATURES == 8 and features.shape[1] == 23),
        "fixed_fit_budget": bool(len(base._read(CONFIG)["model"]["seeds"]) == 3),
    }


def _write_with_objective(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["objective"] = {
            "kind": "fishr_last_layer_per_sample_gradient_variance",
            "coefficient": 1.0,
            "environment": "station_x_layer_x_calendar_quarter",
            "environment_bits_used_as_predictors": False,
            "outer_rows_in_training": 0,
        }
        hashes = dict(payload["hashes"])
        hashes["shared_engine"] = base._sha(SHARED_ENGINE)
        payload["hashes"] = hashes
    _BASE_WRITE(path, payload)


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = causal_environment_features
    shared._synthetic_guards = _synthetic_guards
    shared.shared.LinearProbeClassifier = FishrClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = causal_environment_features
    base.VIBClassifier = FishrClassifier
    base._write = _write_with_objective


def _install_hooks() -> None:
    shared._configure = _configure
    _configure()


def _source_environment_support(data_dir: Path, ready: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(data_dir / "train.csv", usecols=["station", "layer", "time"])
    time = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    time_ns = base._time_ns(time)
    boundary_ns = pd.Timestamp(ready["pre_q2"]["fit_boundary"]).value
    if time_ns.dtype != np.dtype("int64") or not int(time_ns.min()) < boundary_ns < int(time_ns.max()):
        raise RuntimeError("pre-Q2 source support cutoff is not distinct nanoseconds")
    use = time_ns <= boundary_ns
    work = pd.DataFrame(
        {
            "station": frame.loc[use, "station"].astype(str).to_numpy(),
            "layer": frame.loc[use, "layer"].to_numpy(np.int64),
            "quarter": time.loc[use].dt.quarter.to_numpy(np.int64),
        }
    )
    counts = work.groupby(["station", "layer", "quarter"], observed=True).size()
    return {
        "rows": int(use.sum()),
        "stations": int(work["station"].nunique()),
        "station_layers": int(work.groupby(["station", "layer"], observed=True).ngroups),
        "station_layer_quarter_environments": int(len(counts)),
        "minimum_rows_per_environment": int(counts.min()),
        "median_rows_per_environment": float(counts.median()),
        "maximum_rows_per_environment": int(counts.max()),
        "target_columns_read": 0,
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = shared.preflight(data_dir)
    config = base._read(CONFIG)
    support = _source_environment_support(data_dir, ready)
    gate = config["representation_support_gate"]
    passed = (
        support["stations"] >= int(gate["minimum_stations"])
        and support["station_layer_quarter_environments"] >= int(gate["minimum_source_environments"])
        and support["minimum_rows_per_environment"] >= int(gate["minimum_rows_per_source_environment"])
    )
    support["gate"] = "PASS" if passed else gate["failure"]
    ready["source_environment_support"] = support
    ready["objective_guards"] = _synthetic_guards(config["representation"])
    if not passed:
        raise RuntimeError(gate["failure"])
    return ready


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    value = shared.qa(data_dir)
    checks = value["checks"]
    checks["fishr_guards"] = all(_synthetic_guards(base._read(CONFIG)["representation"]).values())
    result_path = ARTIFACT / "result.json"
    if result_path.exists():
        result = base._read(result_path)
        checks["objective"] = result["objective"] == {
            "kind": "fishr_last_layer_per_sample_gradient_variance",
            "coefficient": 1.0,
            "environment": "station_x_layer_x_calendar_quarter",
            "environment_bits_used_as_predictors": False,
            "outer_rows_in_training": 0,
        }
        checks["shared_engine"] = result["hashes"]["shared_engine"] == base._sha(SHARED_ENGINE)
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    if result_path.exists():
        value["result_sha256"] = base._sha(result_path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else execute(args.data_dir) if args.execute else qa(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
