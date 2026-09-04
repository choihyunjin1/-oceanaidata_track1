"""Exactly-once causal temporal-order-verification P1 cross-quarter audit."""

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
EXPERIMENT_ID = "p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v37_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V34)
base = shared.base


def temporal_order_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Return prefix-normalized, exact-cadence backward lag sequences."""

    lags = [int(value) for value in representation["lag_rows"]]
    output = np.zeros((len(frame), len(lags) + 1), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = base._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if not len(prefix):
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        lagged = np.zeros((len(values), len(lags)), dtype=np.float64)
        supported = np.ones((len(values), len(lags)), dtype=bool)
        for column, lag in enumerate(lags):
            if lag == 0:
                lagged[:, column] = values
                supported[:, column] = np.isfinite(raw)
                continue
            supported[:, column] = False
            rows = np.arange(lag, len(values), dtype=np.int64)
            exact = times[rows] - times[rows - lag] == lag * CADENCE_NS
            rows = rows[exact]
            lagged[rows, column] = values[rows - lag]
            supported[rows, column] = np.isfinite(raw[rows - lag])
        full_support = supported.all(axis=1)
        lagged[~full_support] = 0.0
        features = np.column_stack([lagged, full_support.astype(np.float64)])
        if not np.isfinite(features).all():
            raise RuntimeError("temporal-order features are nonfinite")
        output[positions] = features.astype(np.float32)
    return output


class _TemporalOrderNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.order_head = nn.Linear(hidden, 1)
        self.anomaly_head = nn.Linear(hidden, 1)


class TemporalOrderClassifier:
    """Sklearn-shaped order pretrainer followed by a frozen-encoder head."""

    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _TemporalOrderNetwork(inputs, int(config["hidden_units"])).to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> TemporalOrderClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        rng = np.random.default_rng(self.seed)
        maximum = min(len(values), int(self.config["maximum_sample_rows_per_fit"]))
        supported = np.flatnonzero(values[:, -1] > 0.5)
        if not len(supported):
            raise RuntimeError("no supported temporal-order sequences")
        unlabeled = rng.choice(supported, size=min(len(supported), maximum), replace=False)
        optimizer = torch.optim.AdamW(
            list(self.network.encoder.parameters()) + list(self.network.order_head.parameters()),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        batch = int(self.config["batch_rows"])
        self.network.train()
        for _epoch in range(int(self.config["pretext_epochs"])):
            order = unlabeled[rng.permutation(len(unlabeled))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                chronological = torch.from_numpy(values[rows]).to(self.device)
                reversed_sequence = chronological.clone()
                reversed_sequence[:, :-1] = torch.flip(chronological[:, :-1], dims=[1])
                x = torch.cat([chronological, reversed_sequence], dim=0)
                y = torch.cat(
                    [torch.ones(len(rows), device=self.device), torch.zeros(len(rows), device=self.device)]
                )
                logits = self.network.order_head(self.network.encoder(x)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.order_head.parameters():
            parameter.requires_grad_(False)
        ratio = int(self.config["negative_to_positive_sample_ratio"])
        positive = np.flatnonzero(targets == 1)
        negative = np.flatnonzero(targets == 0)
        positive = rng.choice(positive, size=min(len(positive), maximum // (ratio + 1)), replace=False)
        negative = rng.choice(negative, size=min(len(negative), ratio * len(positive)), replace=False)
        selected = np.concatenate([positive, negative])
        optimizer = torch.optim.AdamW(
            self.network.anomaly_head.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        positive_weight = torch.tensor(float(self.config["positive_class_weight"]), device=self.device)
        for _epoch in range(int(self.config["head_epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    encoded = self.network.encoder(x)
                logits = self.network.anomaly_head(encoded).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                logits = self.network.anomaly_head(self.network.encoder(x)).squeeze(1)
                output[start : start + len(x)] = torch.sigmoid(logits).cpu().numpy()
        return output


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = temporal_order_features
    shared._synthetic_guards = _synthetic_guards
    shared.shared.LinearProbeClassifier = TemporalOrderClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = temporal_order_features
    base.VIBClassifier = TemporalOrderClassifier


def _install_hooks() -> None:
    shared._configure = _configure
    _configure()


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    rows = 160
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    values = np.sin(np.arange(rows, dtype=np.float64) / 8.0)
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], rows),
            "layer": np.repeat(["L1", "L2"], rows),
            "_time": np.tile(times, 2),
            "temp": np.tile(values, 2),
        }
    )
    boundary = int(times[79].value)
    original = temporal_order_features(frame, boundary, representation)
    perturbed_frame = frame.copy()
    future = base._time_ns(perturbed_frame["_time"]) > boundary
    perturbed_frame.loc[future, "temp"] += np.linspace(100.0, 500.0, int(future.sum()))
    perturbed = temporal_order_features(perturbed_frame, boundary, representation)
    first_group_support = original[:rows, -1]
    second_group_support = original[rows:, -1]
    example = torch.tensor([[0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 1.0]])
    reversed_example = example.clone()
    reversed_example[:, :-1] = torch.flip(example[:, :-1], dims=[1])
    return {
        "fixed_reverse_is_involution": bool(torch.equal(torch.flip(reversed_example[:, :-1], dims=[1]), example[:, :-1])),
        "ordered_and_reverse_distinct": bool(not torch.equal(example, reversed_example)),
        "station_layer_group_reset": bool(
            np.all(first_group_support[:36] == 0.0)
            and np.all(second_group_support[:36] == 0.0)
            and first_group_support[36] == second_group_support[36] == 1.0
        ),
        "prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
        "ns_cutoff_distinct": bool(
            base._time_ns(times).dtype == np.dtype("int64")
            and int(times[78].value) < boundary < int(times[80].value)
        ),
        "shape_finite": bool(original.shape == (2 * rows, 9) and np.isfinite(original).all()),
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.preflight(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.qa(data_dir)


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
