"""Exactly-once generalized-cross-entropy P1 cross-quarter audit."""

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
EXPERIMENT_ID = "p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v36_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V34)
base = shared.base
CAUSAL_FEATURES = base.shared.causal_evidential_features


def generalized_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, q: float) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target_probability = torch.where(targets > 0.5, probability, 1.0 - probability).clamp(1e-7, 1.0)
    return -torch.expm1(q * torch.log(target_probability)) / q


class _GCENetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class GCEClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _GCENetwork(inputs, int(config["hidden_units"])).to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> GCEClassifier:
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
        q = float(self.config["gce_q"])
        batch = int(self.config["batch_rows"])
        positive_weight = float(self.config["positive_class_weight"])
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                loss_rows = generalized_cross_entropy(self.network(x), y, q)
                weights = torch.where(y > 0.5, positive_weight, 1.0)
                loss = (loss_rows * weights).mean()
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


def _objective_guards(_representation: dict[str, Any]) -> dict[str, bool]:
    logits = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    positive = torch.ones(4)
    loss = generalized_cross_entropy(logits, positive, 0.7)
    loss.sum().backward()
    probabilities = torch.tensor([0.05, 0.2, 0.8, 0.95])
    direct = (1.0 - probabilities.pow(0.7)) / 0.7
    ce_limit = generalized_cross_entropy(torch.tensor([-1.5, 0.0, 1.5]), torch.ones(3), 1e-6)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-1.5, 0.0, 1.5]), torch.ones(3), reduction="none"
    )
    return {
        "fixed_q_in_open_interval": 0.0 < 0.7 < 1.0,
        "loss_finite": bool(torch.isfinite(loss).all()),
        "gradient_finite_negative": bool(torch.isfinite(logits.grad).all() and torch.all(logits.grad < 0.0)),
        "confidence_monotone": bool(torch.all(direct[:-1] > direct[1:])),
        "cross_entropy_limit": bool(torch.allclose(ce_limit, ce, atol=2e-4, rtol=2e-4)),
    }


def _configure() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.CONFIG = CONFIG
    shared.ARTIFACT = ARTIFACT
    shared.LOCK = LOCK
    shared.dfa_features = CAUSAL_FEATURES
    shared._synthetic_guards = _objective_guards
    shared.shared.LinearProbeClassifier = GCEClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_FEATURES
    base.VIBClassifier = GCEClassifier


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return shared.preflight(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _configure()
    return shared.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
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
