"""Exactly-once R-Drop-consistency P1 cross-quarter audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v39_causal_rdrop_consistency_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V36 = ROOT / "scripts/run_p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v39_shared", path)
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


def symmetric_bernoulli_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first = torch.sigmoid(first_logits).clamp(1e-6, 1.0 - 1e-6)
    second = torch.sigmoid(second_logits).clamp(1e-6, 1.0 - 1e-6)
    first_to_second = first * torch.log(first / second) + (1.0 - first) * torch.log((1.0 - first) / (1.0 - second))
    second_to_first = second * torch.log(second / first) + (1.0 - second) * torch.log((1.0 - second) / (1.0 - first))
    return 0.5 * (first_to_second + second_to_first)


class _RDropNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.hidden(values))).squeeze(1)


class RDropClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _RDropNetwork(inputs, int(config["hidden_units"]), float(config["dropout_probability"])).to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> RDropClassifier:
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
        coefficient = float(self.config["consistency_coefficient"])
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                first_logits = self.network(x)
                second_logits = self.network(x)
                first_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    first_logits, y, pos_weight=positive_weight, reduction="none"
                )
                second_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    second_logits, y, pos_weight=positive_weight, reduction="none"
                )
                consistency = symmetric_bernoulli_kl(first_logits, second_logits)
                loss = (0.5 * (first_loss + second_loss) + coefficient * consistency).mean()
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
                output[start : start + len(x)] = torch.sigmoid(self.network(x)).cpu().numpy()
        return output


def _synthetic_guards(_representation: dict[str, Any]) -> dict[str, bool]:
    equal = symmetric_bernoulli_kl(torch.tensor([-2.0, 0.0, 2.0]), torch.tensor([-2.0, 0.0, 2.0]))
    forward = symmetric_bernoulli_kl(torch.tensor([-2.0, 0.5, 2.0]), torch.tensor([1.0, -0.5, -1.0]))
    reverse = symmetric_bernoulli_kl(torch.tensor([1.0, -0.5, -1.0]), torch.tensor([-2.0, 0.5, 2.0]))
    network = _RDropNetwork(4, 8, 0.1)
    sample = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    network.train()
    first = network(sample)
    second = network(sample)
    network.eval()
    deterministic_first = network(sample)
    deterministic_second = network(sample)
    return {
        "equal_distributions_zero": bool(torch.max(torch.abs(equal)) < 1e-7),
        "symmetric": bool(torch.allclose(forward, reverse, atol=1e-7, rtol=1e-7)),
        "mismatch_positive_finite": bool(torch.all(forward > 0.0) and torch.isfinite(forward).all()),
        "training_dropout_distinct": bool(not torch.equal(first, second)),
        "inference_deterministic": bool(torch.equal(deterministic_first, deterministic_second)),
    }


def _write_with_objective(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["objective"] = {"kind": "rdrop_bidirectional_bernoulli_kl", "dropout": 0.1, "coefficient": 1.0, "outer_rows_in_training": 0}
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
    shared.shared.LinearProbeClassifier = RDropClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_FEATURES
    base.VIBClassifier = RDropClassifier
    base._write = _write_with_objective


def _install_hooks() -> None:
    shared._configure = _configure
    _configure()


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = shared.preflight(data_dir)
    ready["objective_guards"] = _synthetic_guards(base._read(CONFIG)["representation"])
    return ready


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return shared.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    value = shared.qa(data_dir)
    checks = value["checks"]
    result_path = ARTIFACT / "result.json"
    checks["rdrop_guards"] = all(_synthetic_guards(base._read(CONFIG)["representation"]).values())
    if result_path.exists():
        result = base._read(result_path)
        checks["objective"] = result["objective"] == {"kind": "rdrop_bidirectional_bernoulli_kl", "dropout": 0.1, "coefficient": 1.0, "outer_rows_in_training": 0}
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
