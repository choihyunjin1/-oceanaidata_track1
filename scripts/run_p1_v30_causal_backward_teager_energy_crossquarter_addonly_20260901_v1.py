"""Exactly-once causal backward-Teager-energy P1 cross-quarter audit."""

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
EXPERIMENT_ID = "p1_v30_causal_backward_teager_energy_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V29 = ROOT / "scripts/run_p1_v29_causal_variational_information_bottleneck_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v30_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


base = _module(V29)


def backward_teager(values: np.ndarray, times_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return psi[x] at t using only x[t-2:t+1], representing center t-1."""

    values = np.asarray(values, dtype=np.float64)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    energy = np.zeros(len(values), dtype=np.float64)
    supported = np.zeros(len(values), dtype=bool)
    if len(values) < 3:
        return energy, supported
    rows = np.arange(2, len(values))
    exact = (times_ns[rows] - times_ns[rows - 1] == CADENCE_NS) & (
        times_ns[rows - 1] - times_ns[rows - 2] == CADENCE_NS
    )
    rows = rows[exact]
    energy[rows] = np.square(values[rows - 1]) - values[rows - 2] * values[rows]
    supported[rows] = True
    return energy, supported


def _segment_rolling(values: np.ndarray, times_ns: np.ndarray, window: int) -> np.ndarray:
    segments = np.cumsum(np.r_[True, np.diff(times_ns) != CADENCE_NS])
    series = pd.Series(values)
    return (
        series.groupby(segments, sort=False)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy(np.float64)
    )


def teager_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    output = np.zeros((len(frame), 10), dtype=np.float32)
    windows = [int(value) for value in representation["rolling_rows"]]
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
        energy, supported = backward_teager(values, times)
        prefix_energy = energy[(times <= train_boundary_ns) & supported]
        energy_center = float(np.median(prefix_energy)) if len(prefix_energy) else 0.0
        energy_scale = float(1.4826 * np.median(np.abs(prefix_energy - energy_center))) if len(prefix_energy) else 1.0
        if not np.isfinite(energy_scale) or energy_scale < 1e-6:
            energy_scale = float(np.std(prefix_energy)) if len(prefix_energy) else 1.0
        if not np.isfinite(energy_scale) or energy_scale < 1e-6:
            energy_scale = 1.0
        normalized = np.clip((energy - energy_center) / energy_scale, -12.0, 12.0)
        normalized[~supported] = 0.0
        means = [_segment_rolling(normalized, times, window) for window in windows]
        previous = np.r_[0.0, normalized[:-1]]
        change = normalized - previous
        change[~supported] = 0.0
        feature = np.column_stack(
            [
                values,
                normalized,
                np.abs(normalized),
                np.maximum(normalized, 0.0),
                np.maximum(-normalized, 0.0),
                *means,
                change,
                supported.astype(np.float64),
            ]
        )
        if not np.isfinite(feature).all():
            raise RuntimeError("Teager features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


class LinearProbeClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = nn.Linear(inputs, 1).to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LinearProbeClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
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
        positive_weight = torch.tensor(float(self.config["positive_class_weight"]), device=self.device)
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                logits = self.network(x).squeeze(1)
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
                output[start : start + len(x)] = torch.sigmoid(self.network(x).squeeze(1)).cpu().numpy()
        return output


def _configure() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = teager_features
    base.VIBClassifier = LinearProbeClassifier


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return base.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    return base.qa(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _configure()
    return base.execute(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else qa(args.data_dir) if args.qa else execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
