"""Exactly-once causal spectral-decoupling P1 audit."""

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
EXPERIMENT_ID = "p1_v46_causal_spectral_decoupling_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V36 = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v46_shared", path)
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
TRAINING_RECEIPTS: list[dict[str, Any]] = []


def spectral_decoupling_penalty(logits: torch.Tensor, coefficient: float) -> torch.Tensor:
    return 0.5 * float(coefficient) * torch.square(logits).mean()


class _SpectralDecouplingNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class SpectralDecouplingClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        if int(inputs) != 8:
            raise ValueError("spectral-decoupling input width changed")
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _SpectralDecouplingNetwork(inputs, int(config["hidden_units"])).to(self.device)
        self.training_receipt: dict[str, Any] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> SpectralDecouplingClassifier:
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
        coefficient = float(self.config["spectral_decoupling_coefficient"])
        bce_values: list[float] = []
        penalty_values: list[float] = []
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                logits = self.network(x)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=positive_weight
                )
                penalty = spectral_decoupling_penalty(logits, coefficient)
                optimizer.zero_grad(set_to_none=True)
                (bce + penalty).backward()
                optimizer.step()
                bce_values.append(float(bce.detach().cpu()))
                penalty_values.append(float(penalty.detach().cpu()))
        self.training_receipt = {
            "seed": self.seed,
            "sample_rows": int(len(selected)),
            "optimizer_steps": int(len(penalty_values)),
            "mean_weighted_bce": float(np.mean(bce_values)),
            "mean_spectral_decoupling_penalty": float(np.mean(penalty_values)),
            "maximum_spectral_decoupling_penalty": float(np.max(penalty_values)),
        }
        TRAINING_RECEIPTS.append(dict(self.training_receipt))
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                output[start : start + len(x)] = torch.sigmoid(self.network(x)).cpu().numpy()
        return output


def _synthetic_guards(_representation: dict[str, Any]) -> dict[str, bool]:
    logits = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    penalty = spectral_decoupling_penalty(logits, 0.01)
    penalty.backward()
    expected_gradient = 0.01 * logits.detach() / len(logits)
    zero = spectral_decoupling_penalty(torch.zeros(5), 0.01)
    reversed_penalty = spectral_decoupling_penalty(-logits.detach(), 0.01)
    return {
        "zero_logits_zero_penalty": bool(zero == 0.0),
        "sign_symmetric": bool(torch.equal(penalty.detach(), reversed_penalty)),
        "positive_for_nonzero_logits": bool(torch.isfinite(penalty) and penalty > 0.0),
        "gradient_matches_formula": bool(torch.allclose(logits.grad, expected_gradient, atol=1e-9, rtol=1e-7)),
        "single_fixed_coefficient": bool(base._read(CONFIG)["model"]["spectral_decoupling_coefficient"] == 0.01),
        "weight_decay_replaced": bool(base._read(CONFIG)["model"]["weight_decay"] == 0.0),
    }


def _write_with_objective(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["objective"] = {"kind": "spectral_decoupling_squared_logit_penalty", "coefficient": 0.01, "formula": "weighted_BCE + 0.5 * 0.01 * mean(logit^2)", "weight_decay": 0.0, "outer_rows_in_training": 0}
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
    shared.dfa_features = CAUSAL_FEATURES
    shared._synthetic_guards = _synthetic_guards
    shared.shared.LinearProbeClassifier = SpectralDecouplingClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_FEATURES
    base.VIBClassifier = SpectralDecouplingClassifier
    base._write = _write_with_objective


def _install_hooks() -> None:
    shared._configure = _configure
    _configure()


def _target_free_conditioning(data_dir: Path, ready: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(data_dir / "train.csv", usecols=["station", "layer", "time", "temp"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    time_ns = base._time_ns(frame["_time"])
    boundary_ns = pd.Timestamp(ready["pre_q2"]["fit_boundary"]).value
    if not int(time_ns.min()) < boundary_ns < int(time_ns.max()):
        raise RuntimeError("conditioning cutoff is not distinct nanoseconds")
    features = CAUSAL_FEATURES(frame, boundary_ns, base._read(CONFIG)["representation"])
    use = time_ns <= boundary_ns
    scaler = base.StandardScaler().fit(features[use])
    standardized = scaler.transform(features[use]).astype(np.float32)
    covariance = np.cov(standardized, rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    initial_variances = []
    initial_mean_squares = []
    model_config = base._read(CONFIG)["model"]
    for seed in model_config["seeds"]:
        torch.manual_seed(int(seed))
        network = _SpectralDecouplingNetwork(standardized.shape[1], int(model_config["hidden_units"]))
        values = []
        with torch.no_grad():
            for start in range(0, len(standardized), 32768):
                values.append(network(torch.from_numpy(standardized[start : start + 32768])).numpy())
        logits = np.concatenate(values)
        initial_variances.append(float(np.var(logits)))
        initial_mean_squares.append(float(np.mean(np.square(logits))))
    return {
        "prefix_rows": int(use.sum()),
        "feature_count": int(standardized.shape[1]),
        "finite": bool(np.isfinite(standardized).all()),
        "minimum_covariance_eigenvalue": float(eigenvalues.min()),
        "maximum_covariance_eigenvalue": float(eigenvalues.max()),
        "covariance_condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "initial_logit_variances": initial_variances,
        "initial_mean_squared_logits": initial_mean_squares,
        "target_columns_read": 0,
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = shared.preflight(data_dir)
    config = base._read(CONFIG)
    support = _target_free_conditioning(data_dir, ready)
    gate = config["representation_support_gate"]
    passed = support["prefix_rows"] >= int(gate["minimum_prefix_rows"]) and support["feature_count"] == int(gate["required_feature_count"]) and support["finite"] and support["minimum_covariance_eigenvalue"] >= float(gate["minimum_covariance_eigenvalue"]) and support["covariance_condition_number"] <= float(gate["maximum_covariance_condition_number"]) and min(support["initial_logit_variances"]) >= float(gate["minimum_initial_logit_variance"]) and max(support["initial_logit_variances"]) <= float(gate["maximum_initial_logit_variance"])
    support["gate"] = "PASS" if passed else gate["failure"]
    ready["target_free_conditioning"] = support
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
    checks["spectral_decoupling_guards"] = all(_synthetic_guards(base._read(CONFIG)["representation"]).values())
    result_path = ARTIFACT / "result.json"
    if result_path.exists():
        result = base._read(result_path)
        checks["objective"] = result["objective"] == {"kind": "spectral_decoupling_squared_logit_penalty", "coefficient": 0.01, "formula": "weighted_BCE + 0.5 * 0.01 * mean(logit^2)", "weight_decay": 0.0, "outer_rows_in_training": 0}
        checks["training_receipts"] = len(result["training_receipts"]) == result["counters"]["fits"] and all(item["optimizer_steps"] > 0 and item["mean_spectral_decoupling_penalty"] > 0.0 for item in result["training_receipts"])
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
