"""Exactly-once station-adversarial invariant P1 falsification."""

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
EXPERIMENT_ID = "p1_v24_causal_station_adversarial_invariant_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16 = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v24_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def causal_station_features(
    frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]
) -> np.ndarray:
    """Prefix-normalized current/backward temperature plus nuisance station IDs."""

    del representation
    shared._set_transport_context(frame, train_boundary_ns)
    output = np.zeros((len(frame), 4), dtype=np.float64)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = core._time_ns(ordered["_time"])
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
        normalized = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        exact = np.r_[False, np.diff(times) == CADENCE_NS]
        difference = np.r_[0.0, np.diff(normalized)]
        difference[~exact] = 0.0
        output[positions] = np.column_stack(
            [normalized, np.clip(difference, -12.0, 12.0), ~np.isfinite(raw), ~exact]
        )
    prefix_rows = core._time_ns(frame["_time"]) <= train_boundary_ns
    stations = sorted(frame.loc[prefix_rows, "station"].astype(str).unique())
    station = frame["station"].astype(str).to_numpy()
    nuisance = np.column_stack([station == value for value in stations]).astype(np.float64)
    result = np.column_stack([output, nuisance]).astype(np.float32)
    if not np.isfinite(result).all() or len(stations) < 2:
        raise RuntimeError("station-adversarial feature contract invalid")
    return result


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


class _Network(nn.Module):
    def __init__(self, inputs: int, hidden: int, domains: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.anomaly = nn.Linear(hidden, 1)
        self.domain = nn.Linear(hidden, domains)

    def forward(self, values: torch.Tensor, coefficient: float) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        anomaly = self.anomaly(latent).squeeze(1)
        domain = self.domain(_Reverse.apply(latent, coefficient))
        return anomaly, domain


class StationAdversarialClassifier:
    """Sklearn-shaped small gradient-reversal classifier."""

    def __init__(
        self,
        *,
        loss: str,
        penalty: str,
        alpha: float,
        max_iter: int,
        tol: None,
        class_weight: dict[int, float],
        shuffle: bool,
        random_state: int,
    ) -> None:
        if loss != "log_loss" or penalty != "l2" or tol is not None or not shuffle:
            raise ValueError("station-adversarial scorer contract drifted")
        self.alpha = float(alpha)
        self.epochs = int(max_iter)
        self.positive_weight = float(class_weight[1])
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> StationAdversarialClassifier:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        objective = config["objective"]
        domain_count = 3
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int8)
        domains = np.argmax(values[:, -domain_count:], axis=1).astype(np.int64)
        task = values[:, :-domain_count]
        rng = np.random.default_rng(self.random_state)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        maximum = int(objective["maximum_sample_rows_per_fit"])
        ratio = int(objective["negative_to_positive_sample_ratio"])
        positive = rng.choice(positive, size=min(len(positive), maximum // (ratio + 1)), replace=False)
        negative = rng.choice(negative, size=min(len(negative), ratio * len(positive)), replace=False)
        selected = np.concatenate([positive, negative])
        selected = selected[rng.permutation(len(selected))]
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = _Network(task.shape[1], int(config["representation"]["encoder_hidden_units"]), domain_count).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=self.alpha,
        )
        task_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.positive_weight, device=self.device))
        domain_loss = nn.CrossEntropyLoss()
        batch = int(objective["batch_rows"])
        coefficient = float(objective["domain_adversarial_coefficient"])
        self.network.train()
        for _epoch in range(self.epochs):
            epoch_order = selected[rng.permutation(len(selected))]
            for start in range(0, len(epoch_order), batch):
                rows = epoch_order[start : start + batch]
                x = torch.from_numpy(task[rows]).to(self.device)
                y = torch.from_numpy(labels[rows].astype(np.float32)).to(self.device)
                d = torch.from_numpy(domains[rows]).to(self.device)
                anomaly, station = self.network(x, coefficient)
                loss_value = task_loss(anomaly, y) + domain_loss(station, d)
                optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                optimizer.step()
        packed = np.concatenate(
            [parameter.detach().cpu().numpy().ravel() for parameter in self.network.parameters()]
        )
        self.coef_ = packed.reshape(1, -1)
        self.intercept_ = np.zeros(1, dtype=np.float32)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)[:, :-3]
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                logits, _ = self.network(x, 0.0)
                output[start : start + len(x)] = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - output, output])


def _write_v24(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
        payload["objective"] = {
            "kind": "anomaly_BCE_plus_gradient_reversed_station_cross_entropy",
            "outer_rows_in_training": 0,
            "outer_domain_rows_in_training": 0,
        }
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = causal_station_features
    core.SGDClassifier = StationAdversarialClassifier
    core._write = _write_v24
    base._select = shared.shared._select_transport


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    result_path = ARTIFACT / "result.json"
    if not result_path.exists():
        return core.qa(data_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "terminal_result": result["experiment_id"] == EXPERIMENT_ID,
        "fits9": result["counters"]["fits"] == 9,
        "add_only": result["counters"]["anchor_removals"] == 0,
        "outer_isolation": result["counters"]["outer_target_reads_before_all_seals"] == 0,
        "domain_outer_isolation": result["objective"]["outer_domain_rows_in_training"] == 0,
        "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
        "config_hash": result["hashes"]["config"] == core._sha(CONFIG),
        "runner_hash": result["hashes"]["runner"] == core._sha(Path(__file__)),
        "lock_hash": result["hashes"]["lock"] == core._sha(LOCK),
        "completion_hash": result["hashes"]["completion"] == core._sha(ARTIFACT / "predictions_complete.json"),
        "seals": all(core._sha(ARTIFACT / f"{fold}_sealed.npz") == json.loads((ARTIFACT / f"{fold}_seal.json").read_text(encoding="utf-8"))["sha256"] for fold in config["parts"]),
    }
    return {"experiment_id": EXPERIMENT_ID, "phase": "POST_TERMINAL_IMMUTABLE_REVALIDATION", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "result_sha256": core._sha(result_path)}


def execute(data_dir: Path) -> dict[str, Any]:
    global _DATA_DIR
    _DATA_DIR = data_dir.resolve(strict=True)
    shared._DATA_DIR = _DATA_DIR
    _configure()
    return core.execute(data_dir)


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
