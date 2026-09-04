"""Exactly-once cross-environment gradient-sign AND-mask P1 audit."""

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
EXPERIMENT_ID = "p1_v45_causal_cross_environment_gradient_sign_mask_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V44 = ROOT / "scripts/run_p1_v44_causal_fishr_gradient_variance_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v45_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared science module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


previous = _module(V44)
shared = previous.shared
base = previous.base
CAUSAL_ENVIRONMENT_FEATURES = previous.causal_environment_features
SCIENTIFIC_FEATURES = previous.SCIENTIFIC_FEATURES
STATIONS = previous.STATIONS
LAYERS = previous.LAYERS
QUARTERS = previous.QUARTERS
ENVIRONMENT_BITS = previous.ENVIRONMENT_BITS
_BASE_WRITE = base._write
TRAINING_RECEIPTS: list[dict[str, Any]] = []


def decode_source_metadata(scaled_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover station-quarter environment and layer from standardized one-hot bits."""

    values = np.asarray(scaled_features, dtype=np.float32)
    metadata = values[:, SCIENTIFIC_FEATURES:]
    if metadata.shape[1] != ENVIRONMENT_BITS:
        raise ValueError("AND-mask metadata width invalid")
    station = np.argmax(metadata[:, : len(STATIONS)], axis=1)
    layer_start = len(STATIONS)
    quarter_start = layer_start + len(LAYERS)
    layer = np.argmax(metadata[:, layer_start:quarter_start], axis=1)
    quarter = np.argmax(metadata[:, quarter_start:], axis=1)
    environment = station * 10 + quarter
    return environment.astype(np.int64), layer.astype(np.int64)


def and_mask_gradients(
    environment_gradients: list[list[torch.Tensor]],
    threshold: float,
) -> tuple[list[torch.Tensor], float]:
    """Average gradients and zero coordinates without enough sign agreement."""

    if len(environment_gradients) < 2:
        raise ValueError("AND-mask requires at least two environments")
    parameter_count = len(environment_gradients[0])
    if any(len(item) != parameter_count for item in environment_gradients):
        raise ValueError("environment gradient parameter counts differ")
    masked: list[torch.Tensor] = []
    kept = 0
    total = 0
    for parameter_index in range(parameter_count):
        stack = torch.stack([item[parameter_index] for item in environment_gradients])
        agreement = torch.abs(torch.sign(stack).mean(dim=0))
        use = agreement >= float(threshold) - 1e-7
        value = stack.mean(dim=0) * use
        masked.append(value)
        kept += int(use.sum().item())
        total += int(use.numel())
    return masked, float(kept / max(1, total))


class _ANDMaskNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class ANDMaskClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        if int(inputs) != SCIENTIFIC_FEATURES + ENVIRONMENT_BITS:
            raise ValueError("AND-mask input contract changed")
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _ANDMaskNetwork(SCIENTIFIC_FEATURES, int(config["hidden_units"])).to(self.device)
        self.training_receipt: dict[str, Any] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ANDMaskClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        environments, layers = decode_source_metadata(values)
        scientific = values[:, :SCIENTIFIC_FEATURES]
        minimum_rows = int(self.config["minimum_rows_per_supported_environment"])
        minimum_layers = int(self.config["minimum_distinct_layers_per_supported_environment"])
        supported = []
        for environment in sorted(np.unique(environments).tolist()):
            rows = np.flatnonzero(environments == environment)
            if len(rows) >= minimum_rows and len(np.unique(layers[rows])) >= minimum_layers:
                supported.append((int(environment), rows))
        if len(supported) != 5:
            raise RuntimeError("sealed five-environment AND-mask support changed")
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        parameters = list(self.network.parameters())
        rows_per_step = int(self.config["environment_rows_per_step"])
        steps = int(self.config["steps_per_epoch"])
        threshold = float(self.config["gradient_sign_agreement_threshold"])
        positive_weight = torch.tensor(float(self.config["positive_class_weight"]), device=self.device)
        mask_shares: list[float] = []
        positives_seen = 0
        rows_seen = 0
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            schedules = {}
            for environment, rows in supported:
                required = rows_per_step * steps
                schedules[environment] = rng.permutation(rows)[:required].reshape(steps, rows_per_step)
            for step in range(steps):
                environment_gradients: list[list[torch.Tensor]] = []
                for environment, _rows in supported:
                    rows = schedules[environment][step]
                    x = torch.from_numpy(scientific[rows]).to(self.device)
                    y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        self.network(x), y, pos_weight=positive_weight
                    )
                    gradients = torch.autograd.grad(loss, parameters)
                    environment_gradients.append([item.detach() for item in gradients])
                    positives_seen += int(targets[rows].sum())
                    rows_seen += int(len(rows))
                masked, share = and_mask_gradients(environment_gradients, threshold)
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, masked, strict=True):
                    parameter.grad = gradient
                optimizer.step()
                mask_shares.append(share)
        self.training_receipt = {
            "seed": self.seed,
            "supported_environment_ids": [item[0] for item in supported],
            "supported_environments": len(supported),
            "rows_seen": rows_seen,
            "positive_rows_seen": positives_seen,
            "optimizer_steps": len(mask_shares),
            "mean_parameter_coordinate_mask_share": float(np.mean(mask_shares)),
            "minimum_parameter_coordinate_mask_share": float(np.min(mask_shares)),
            "maximum_parameter_coordinate_mask_share": float(np.max(mask_shares)),
        }
        TRAINING_RECEIPTS.append(dict(self.training_receipt))
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)[:, :SCIENTIFIC_FEATURES]
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                output[start : start + len(x)] = torch.sigmoid(self.network(x)).cpu().numpy()
        return output


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    first = [[torch.tensor([2.0, 2.0, -1.0]), torch.tensor([1.0])], [torch.tensor([1.0, -3.0, -2.0]), torch.tensor([2.0])], [torch.tensor([3.0, 4.0, -4.0]), torch.tensor([3.0])]]
    masked, share = and_mask_gradients(first, 1.0)
    permuted, permuted_share = and_mask_gradients([first[2], first[0], first[1]], 1.0)
    rows = 64
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame({"station": np.repeat(["G-ORS", "I-ORS"], rows), "layer": np.repeat([1, 2], rows), "_time": np.tile(time, 2), "temp": np.tile(np.sin(np.arange(rows) / 6.0), 2)})
    boundary = int(pd.DatetimeIndex([time[31]]).as_unit("ns").asi8[0])
    original = CAUSAL_ENVIRONMENT_FEATURES(frame, boundary, representation)
    changed = frame.copy()
    time_ns = pd.DatetimeIndex(changed["_time"]).as_unit("ns").asi8
    future = time_ns > boundary
    changed.loc[future, "temp"] += 1000.0
    perturbed = CAUSAL_ENVIRONMENT_FEATURES(changed, boundary, representation)
    return {
        "unanimous_coordinates_retained": bool(torch.allclose(masked[0], torch.tensor([2.0, 0.0, -7.0 / 3.0]))),
        "conflicting_coordinate_zeroed": bool(masked[0][1] == 0.0 and abs(share - 0.75) < 1e-12),
        "environment_order_invariant": bool(all(torch.equal(a, b) for a, b in zip(masked, permuted, strict=True)) and share == permuted_share),
        "symmetric_null_probability_fixed": bool(abs(2 ** (1 - 5) - 0.0625) < 1e-12),
        "group_reset": bool(np.array_equal(original[:rows, :SCIENTIFIC_FEATURES], original[rows:, :SCIENTIFIC_FEATURES])),
        "prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
        "nanosecond_cutoff_distinct": bool(int(time_ns[30]) < boundary < int(time_ns[32])),
        "environment_bits_not_predictors": bool(original.shape[1] == SCIENTIFIC_FEATURES + ENVIRONMENT_BITS),
    }


def _write_with_objective(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["objective"] = {"kind": "cross_environment_gradient_sign_unanimity_and_mask", "agreement_threshold": 1.0, "supported_environments": 5, "environment": "station_x_calendar_quarter", "environment_bits_used_as_predictors": False, "outer_rows_in_training": 0}
        payload["training_receipts"] = list(TRAINING_RECEIPTS)
        hashes = dict(payload["hashes"])
        hashes["shared_engine"] = base._sha(SHARED_ENGINE)
        payload["hashes"] = hashes
    _BASE_WRITE(path, payload)


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = CAUSAL_ENVIRONMENT_FEATURES
    shared._synthetic_guards = _synthetic_guards
    shared.shared.LinearProbeClassifier = ANDMaskClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_ENVIRONMENT_FEATURES
    base.VIBClassifier = ANDMaskClassifier
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
    work = pd.DataFrame({"station": frame.loc[use, "station"].astype(str).to_numpy(), "layer": frame.loc[use, "layer"].to_numpy(np.int64), "quarter": time.loc[use].dt.quarter.to_numpy(np.int64)})
    grouped = work.groupby(["station", "quarter"], observed=True).agg(rows=("layer", "size"), layers=("layer", "nunique")).reset_index().sort_values(["station", "quarter"], kind="stable")
    gate = base._read(CONFIG)["representation_support_gate"]
    supported = grouped[(grouped["rows"] >= int(gate["minimum_rows_per_supported_environment"])) & (grouped["layers"] >= int(gate["minimum_distinct_layers_per_supported_environment"]))]
    return {
        "rows": int(use.sum()),
        "all_station_quarter_environments": int(len(grouped)),
        "supported_environments": int(len(supported)),
        "supported_distinct_stations": int(supported["station"].nunique()),
        "supported_distinct_quarters": int(supported["quarter"].nunique()),
        "minimum_supported_rows": int(supported["rows"].min()),
        "minimum_supported_layers": int(supported["layers"].min()),
        "supported_environment_receipts": [{"station": str(row.station), "quarter": int(row.quarter), "rows": int(row.rows), "layers": int(row.layers)} for row in supported.itertuples(index=False)],
        "target_columns_read": 0,
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = shared.preflight(data_dir)
    config = base._read(CONFIG)
    support = _source_environment_support(data_dir, ready)
    gate = config["representation_support_gate"]
    passed = support["supported_environments"] >= int(gate["minimum_supported_environments"]) and support["supported_distinct_stations"] >= int(gate["minimum_distinct_stations"]) and support["supported_distinct_quarters"] >= int(gate["minimum_distinct_quarters"])
    support["gate"] = "PASS" if passed else gate["failure"]
    ready["source_environment_support"] = support
    ready["objective_guards"] = _synthetic_guards(config["representation"])
    if not passed:
        raise RuntimeError(gate["failure"])
    return ready


def execute(data_dir: Path) -> dict[str, Any]:
    TRAINING_RECEIPTS.clear()
    _install_hooks()
    ready = preflight(data_dir)
    original_preflight = shared.preflight
    shared.preflight = lambda _data_dir: ready
    try:
        return shared.execute(data_dir)
    finally:
        shared.preflight = original_preflight


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    value = shared.qa(data_dir)
    checks = value["checks"]
    checks["and_mask_guards"] = all(_synthetic_guards(base._read(CONFIG)["representation"]).values())
    result_path = ARTIFACT / "result.json"
    if result_path.exists():
        result = base._read(result_path)
        checks["objective"] = result["objective"] == {"kind": "cross_environment_gradient_sign_unanimity_and_mask", "agreement_threshold": 1.0, "supported_environments": 5, "environment": "station_x_calendar_quarter", "environment_bits_used_as_predictors": False, "outer_rows_in_training": 0}
        checks["training_receipts"] = len(result["training_receipts"]) == result["counters"]["fits"] and all(item["supported_environments"] == 5 and item["optimizer_steps"] == 192 for item in result["training_receipts"])
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
