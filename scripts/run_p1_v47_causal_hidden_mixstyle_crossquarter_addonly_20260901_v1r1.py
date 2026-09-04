"""Fresh-ID science-neutral recovery of the causal hidden MixStyle P1 audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1r1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V47 = ROOT / "scripts/run_p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v47r1_frozen_science", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen v47 science module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


frozen = _module(V47)
base = frozen.base


def resolve_support_thresholds(config: dict[str, Any]) -> tuple[int, int]:
    """Resolve only the repaired support-gate namespace; old model keys are forbidden."""

    model = config["model"]
    forbidden = {
        "minimum_rows_per_supported_environment",
        "minimum_distinct_layers_per_supported_environment",
    }
    if forbidden.intersection(model):
        raise RuntimeError("legacy support thresholds must not be injected into model config")
    gate = config["representation_support_gate"]
    required = {
        "minimum_rows_per_donor_environment",
        "minimum_distinct_layers_per_donor_environment",
    }
    missing = required.difference(gate)
    if missing:
        raise KeyError(f"representation_support_gate missing {sorted(missing)}")
    return (
        int(gate["minimum_rows_per_donor_environment"]),
        int(gate["minimum_distinct_layers_per_donor_environment"]),
    )


class MixStyleClassifier(frozen.MixStyleClassifier):
    """Frozen v47 classifier with the sole support-namespace repair."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> MixStyleClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        environments, layers = frozen.decode_source_metadata(values)
        scientific = values[:, : frozen.SCIENTIFIC_FEATURES]
        minimum_rows, minimum_layers = resolve_support_thresholds(base._read(CONFIG))
        supported_pools: dict[int, np.ndarray] = {}
        for environment in sorted(np.unique(environments).tolist()):
            rows = np.flatnonzero(environments == environment)
            if len(rows) >= minimum_rows and len(np.unique(layers[rows])) >= minimum_layers:
                supported_pools[int(environment)] = rows
        if len(supported_pools) != 5:
            raise RuntimeError("sealed five-environment MixStyle support changed")
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
        alpha = float(self.config["beta_alpha"])
        probability = float(self.config["mix_probability"])
        positive_weight = torch.tensor(float(self.config["positive_class_weight"]), device=self.device)
        optimizer_steps = 0
        mixstyle_steps = 0
        mixed_rows = 0
        same_environment_donors = 0
        lambda_values: list[float] = []
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(scientific[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                if rng.random() < probability:
                    partner_rows = frozen.choose_partner_rows(environments[rows], supported_pools, rng)
                    same_environment_donors += int(
                        np.sum(environments[partner_rows] == environments[rows])
                    )
                    donor = torch.from_numpy(scientific[partner_rows]).to(self.device)
                    sampled = rng.beta(alpha, alpha, size=(len(rows), 1)).astype(np.float32)
                    lambdas = torch.from_numpy(sampled).to(self.device)
                    logits = self.network(x, donor, lambdas)
                    mixstyle_steps += 1
                    mixed_rows += len(rows)
                    lambda_values.extend(sampled[:, 0].tolist())
                else:
                    logits = self.network(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=positive_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
        self.training_receipt = {
            "seed": self.seed,
            "supported_environment_ids": sorted(supported_pools),
            "supported_environments": len(supported_pools),
            "sample_rows": int(len(selected)),
            "optimizer_steps": optimizer_steps,
            "mixstyle_steps": mixstyle_steps,
            "mixed_rows": int(mixed_rows),
            "same_environment_donor_rows": same_environment_donors,
            "empirical_mix_probability": float(mixstyle_steps / optimizer_steps),
            "mean_lambda": float(np.mean(lambda_values)),
            "lambda_standard_deviation": float(np.std(lambda_values)),
            "labels_mixed": False,
        }
        frozen.TRAINING_RECEIPTS.append(dict(self.training_receipt))
        return self


def _configure_namespace() -> None:
    frozen.EXPERIMENT_ID = EXPERIMENT_ID
    frozen.CONFIG = CONFIG
    frozen.ARTIFACT = ARTIFACT
    frozen.LOCK = LOCK
    frozen.__file__ = str(Path(__file__).resolve())
    frozen.MixStyleClassifier = MixStyleClassifier
    frozen._install_hooks()


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure_namespace()
    ready = frozen.preflight(data_dir)
    ready["science_neutral_recovery"] = {
        "parent": "p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1",
        "change": "support thresholds resolved from representation_support_gate",
        "thresholds": list(resolve_support_thresholds(base._read(CONFIG))),
    }
    return ready


def execute(data_dir: Path) -> dict[str, Any]:
    _configure_namespace()
    return frozen.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure_namespace()
    value = frozen.qa(data_dir)
    config = base._read(CONFIG)
    checks = value["checks"]
    checks["support_namespace_repaired"] = resolve_support_thresholds(config) == (4096, 2)
    checks["legacy_model_support_keys_absent"] = not {
        "minimum_rows_per_supported_environment",
        "minimum_distinct_layers_per_supported_environment",
    }.intersection(config["model"])
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
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
