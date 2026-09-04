"""Exactly-once causal Dirichlet-evidential P1 falsification."""

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
EXPERIMENT_ID = "p1_v27_causal_dirichlet_evidential_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16 = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v27_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def causal_evidential_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    """Return fixed causal state features with prefix-only normalization."""

    shared._set_transport_context(frame, train_boundary_ns)
    lags = [int(value) for value in representation["lag_rows"]]
    output = np.zeros((len(frame), 8), dtype=np.float32)
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
            valid = np.arange(lag, len(values))
            exact = times[valid] - times[valid - lag] == lag * CADENCE_NS
            rows = valid[exact]
            lagged[rows, column] = values[rows - lag]
            supported[rows, column] = True
        diff = values - lagged[:, 0]
        diff[~supported[:, 0]] = 0.0
        previous_diff = np.r_[0.0, diff[:-1]]
        acceleration = diff - previous_diff
        acceleration[~supported[:, 0]] = 0.0
        gap = (~supported.all(axis=1)).astype(np.float64)
        output[positions] = np.column_stack([values, lagged, diff, np.abs(diff), acceleration, gap]).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("evidential features are nonfinite")
    return output


def evidential_loss(logits: torch.Tensor, labels: torch.Tensor, kl_coefficient: float) -> torch.Tensor:
    """Two-class Dirichlet evidential loss with wrong-class uniform KL."""

    evidence = torch.nn.functional.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    target = torch.nn.functional.one_hot(labels, num_classes=2).to(alpha.dtype)
    expected = alpha / strength
    mse = ((target - expected) ** 2).sum(dim=1)
    variance = (alpha * (strength - alpha) / (strength * strength * (strength + 1.0))).sum(dim=1)
    adjusted = target + (1.0 - target) * alpha
    adjusted_strength = adjusted.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(adjusted_strength).squeeze(1)
        - torch.lgamma(adjusted).sum(dim=1)
        - torch.lgamma(torch.tensor(2.0, device=alpha.device))
        + ((adjusted - 1.0) * (torch.digamma(adjusted) - torch.digamma(adjusted_strength))).sum(dim=1)
    )
    return mse + variance + kl_coefficient * kl


class _EvidentialNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EvidentialClassifier:
    """Sklearn-shaped Dirichlet-evidence classifier."""

    def __init__(self, *, loss: str, penalty: str, alpha: float, max_iter: int, tol: None, class_weight: dict[int, float], shuffle: bool, random_state: int) -> None:
        if loss != "log_loss" or penalty != "l2" or tol is not None or not shuffle:
            raise ValueError("evidential classifier contract drifted")
        self.alpha = float(alpha)
        self.epochs = int(max_iter)
        self.positive_weight = float(class_weight[1])
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> EvidentialClassifier:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        objective = config["objective"]
        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int8)
        rng = np.random.default_rng(self.random_state)
        maximum = int(objective["maximum_sample_rows_per_fit"])
        ratio = int(objective["negative_to_positive_sample_ratio"])
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        positive = rng.choice(positive, size=min(len(positive), maximum // (ratio + 1)), replace=False)
        negative = rng.choice(negative, size=min(len(negative), ratio * len(positive)), replace=False)
        selected = np.concatenate([positive, negative])
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = _EvidentialNetwork(values.shape[1], int(config["representation"]["encoder_hidden_units"])).to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=self.alpha)
        batch = int(objective["batch_rows"])
        kl = float(objective["kl_coefficient"])
        self.network.train()
        for _epoch in range(self.epochs):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(labels[rows].astype(np.int64)).to(self.device)
                loss_rows = evidential_loss(self.network(x), y, kl)
                weights = torch.where(y == 1, self.positive_weight, 1.0)
                loss_value = (loss_rows * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                loss_value.backward()
                optimizer.step()
        packed = np.concatenate([parameter.detach().cpu().numpy().ravel() for parameter in self.network.parameters()])
        self.coef_, self.intercept_ = packed.reshape(1, -1), np.zeros(1, dtype=np.float32)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        output = np.empty((len(values), 2), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                alpha = torch.nn.functional.softplus(self.network(x)) + 1.0
                output[start : start + len(x)] = (alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy()
        return output


def _select_amended(scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
    legacy = shared.shared._select_transport(scores, labels, selection)
    eligible, evaluated = [], []
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


def _write_v27(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
        payload["objective"] = {"kind": "dirichlet_expected_mse_variance_plus_wrong_class_uniform_kl", "uncertainty_use": "diagnostic_only", "outer_rows_in_training": 0}
        payload["transport_guard_amendment_sha256"] = core._sha(ROOT / "configs/experiments/p1_v26_transport_guard_amendment_20260901_v1.json")
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = causal_evidential_features
    core.SGDClassifier = EvidentialClassifier
    core._write = _write_v27
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
    checks = {"terminal_result": result["experiment_id"] == EXPERIMENT_ID, "fits9": result["counters"]["fits"] == 9, "add_only": result["counters"]["anchor_removals"] == 0, "outer_isolation": result["counters"]["outer_target_reads_before_all_seals"] == 0, "uncertainty_diagnostic_only": result["objective"]["uncertainty_use"] == "diagnostic_only", "amendment_hash": result["transport_guard_amendment_sha256"] == config["transport_guard_amendment"]["sha256"], "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0, "config_hash": result["hashes"]["config"] == core._sha(CONFIG), "runner_hash": result["hashes"]["runner"] == core._sha(Path(__file__)), "lock_hash": result["hashes"]["lock"] == core._sha(LOCK), "completion_hash": result["hashes"]["completion"] == core._sha(ARTIFACT / "predictions_complete.json"), "seals": all(core._sha(ARTIFACT / f"{fold}_sealed.npz") == json.loads((ARTIFACT / f"{fold}_seal.json").read_text(encoding="utf-8"))["sha256"] for fold in config["parts"])}
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
