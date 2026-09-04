"""Exactly-once causal Barlow-redundancy P1 falsification."""

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
EXPERIMENT_ID = "p1_v26_causal_barlow_redundancy_reduction_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16 = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v26_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def causal_multilag_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    """Prefix-normalized levels and differences at fixed backward lags."""

    shared._set_transport_context(frame, train_boundary_ns)
    lags = [int(value) for value in representation["lag_rows"]]
    output = np.zeros((len(frame), len(lags) + len(lags) - 1 + 1), dtype=np.float32)
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
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        lagged = np.zeros((len(values), len(lags)), dtype=np.float64)
        supported = np.zeros((len(values), len(lags)), dtype=bool)
        for column, lag in enumerate(lags):
            if lag == 0:
                lagged[:, column] = values
                supported[:, column] = np.isfinite(raw)
                continue
            valid = np.arange(lag, len(values))
            exact = times[valid] - times[valid - lag] == lag * CADENCE_NS
            rows = valid[exact]
            lagged[rows, column] = values[rows - lag]
            supported[rows, column] = True
        differences = values[:, None] - lagged[:, 1:]
        differences[~supported[:, 1:]] = 0.0
        gap = (~supported[:, 1:].all(axis=1)).astype(np.float64)
        output[positions] = np.column_stack([lagged, differences, gap]).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("causal multilag features are nonfinite")
    return output


def barlow_loss(first: torch.Tensor, second: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Cross-correlation identity loss with fixed off-diagonal weight."""

    first = (first - first.mean(0)) / first.std(0, unbiased=False).clamp_min(1e-4)
    second = (second - second.mean(0)) / second.std(0, unbiased=False).clamp_min(1e-4)
    correlation = first.T @ second / len(first)
    diagonal = torch.diagonal(correlation)
    off_diagonal = correlation - torch.diag_embed(diagonal)
    return ((diagonal - 1.0) ** 2).sum() + coefficient * (off_diagonal**2).sum()


class _BarlowNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int, projection: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.projector = nn.Linear(hidden, projection, bias=False)
        self.head = nn.Linear(hidden, 1)


class BarlowFrozenHeadClassifier:
    """Sklearn-shaped label-free pretrainer plus frozen-encoder linear head."""

    def __init__(self, *, loss: str, penalty: str, alpha: float, max_iter: int, tol: None, class_weight: dict[int, float], shuffle: bool, random_state: int) -> None:
        if loss != "log_loss" or penalty != "l2" or tol is not None or not shuffle:
            raise ValueError("Barlow scorer contract drifted")
        self.alpha = float(alpha)
        self.head_epochs = int(max_iter)
        self.positive_weight = float(class_weight[1])
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> BarlowFrozenHeadClassifier:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        objective, representation = config["objective"], config["representation"]
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int8)
        rng = np.random.default_rng(self.random_state)
        maximum = min(len(values), int(objective["maximum_sample_rows_per_fit"]))
        unlabeled = rng.choice(len(values), size=maximum, replace=False)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        positive = rng.choice(positive, size=min(len(positive), maximum // 4), replace=False)
        negative = rng.choice(negative, size=min(len(negative), 3 * len(positive)), replace=False)
        supervised = np.concatenate([positive, negative])
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = _BarlowNetwork(values.shape[1], int(representation["encoder_hidden_units"]), int(representation["projector_units"])).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=self.alpha)
        batch = int(objective["batch_rows"])
        dropout = float(objective["view_feature_dropout"])
        noise = float(objective["view_gaussian_noise_std"])
        coefficient = float(objective["off_diagonal_coefficient"])
        self.network.train()
        for _epoch in range(int(objective["pretrain_epochs"])):
            order = unlabeled[rng.permutation(len(unlabeled))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                if len(rows) < 2:
                    continue
                x = torch.from_numpy(values[rows]).to(self.device)
                first = x * (torch.rand_like(x) >= dropout) + noise * torch.randn_like(x)
                second = x * (torch.rand_like(x) >= dropout) + noise * torch.randn_like(x)
                loss_value = barlow_loss(self.network.projector(self.network.encoder(first)), self.network.projector(self.network.encoder(second)), coefficient)
                optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                optimizer.step()
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.projector.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(self.network.head.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=self.alpha)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.positive_weight, device=self.device))
        for _epoch in range(self.head_epochs):
            order = supervised[rng.permutation(len(supervised))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(labels[rows].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    latent = self.network.encoder(x)
                logits = self.network.head(latent).squeeze(1)
                loss_value = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                optimizer.step()
        packed = np.concatenate([parameter.detach().cpu().numpy().ravel() for parameter in self.network.parameters()])
        self.coef_, self.intercept_ = packed.reshape(1, -1), np.zeros(1, dtype=np.float32)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                logits = self.network.head(self.network.encoder(x)).squeeze(1)
                output[start : start + len(x)] = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - output, output])


def _select_amended(scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
    legacy = shared.shared._select_transport(scores, labels, selection)
    eligible = []
    evaluated = []
    contract = selection["transport_stability"]
    for candidate in legacy["candidates"]:
        item = dict(candidate)
        stability = dict(item["transport_stability"])
        environments = stability["supported_environments"]
        station_layers = {(value["station"], value["layer"]) for value in environments}
        stations = {value["station"] for value in environments}
        stability["legacy_passed"] = stability["passed"]
        stability["distinct_station_layer_identities"] = len(station_layers)
        stability["distinct_stations"] = len(stations)
        stability["passed"] = bool(stability["passed"] and len(station_layers) >= contract["minimum_distinct_station_layer_identities"] and len(stations) >= contract["minimum_distinct_stations"])
        item["transport_stability"] = stability
        evaluated.append(item)
        if item["count"] >= selection["minimum_additions"] and item["precision_lcb"] >= selection["wilson90_lcb_minimum"] and stability["passed"]:
            eligible.append(item)
    eligible.sort(key=lambda item: (item["quantile"], item["precision_lcb"]), reverse=True)
    return {"candidates": evaluated, "chosen": eligible[0] if eligible else None, "transport_contract": contract}


def _write_v26(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
        payload["objective"] = {"kind": "barlow_twins_cross_correlation_identity", "labels_in_pretraining": 0, "outer_rows_in_training": 0}
        payload["transport_guard_amendment_sha256"] = core._sha(ROOT / "configs/experiments/p1_v26_transport_guard_amendment_20260901_v1.json")
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = causal_multilag_features
    core.SGDClassifier = BarlowFrozenHeadClassifier
    core._write = _write_v26
    base._select = _select_amended


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
    checks = {"terminal_result": result["experiment_id"] == EXPERIMENT_ID, "fits9": result["counters"]["fits"] == 9, "add_only": result["counters"]["anchor_removals"] == 0, "outer_isolation": result["counters"]["outer_target_reads_before_all_seals"] == 0, "pretrain_label_free": result["objective"]["labels_in_pretraining"] == 0, "amendment_hash": result["transport_guard_amendment_sha256"] == config["transport_guard_amendment"]["sha256"], "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0, "config_hash": result["hashes"]["config"] == core._sha(CONFIG), "runner_hash": result["hashes"]["runner"] == core._sha(Path(__file__)), "lock_hash": result["hashes"]["lock"] == core._sha(LOCK), "completion_hash": result["hashes"]["completion"] == core._sha(ARTIFACT / "predictions_complete.json"), "seals": all(core._sha(ARTIFACT / f"{fold}_sealed.npz") == json.loads((ARTIFACT / f"{fold}_seal.json").read_text(encoding="utf-8"))["sha256"] for fold in config["parts"])}
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
