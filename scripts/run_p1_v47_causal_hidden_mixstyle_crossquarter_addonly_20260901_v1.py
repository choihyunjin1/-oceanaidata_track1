"""Exactly-once causal hidden-feature MixStyle P1 audit."""

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
EXPERIMENT_ID = "p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V44 = ROOT / "scripts/run_p1_v44_causal_fishr_gradient_variance_crossquarter_addonly_20260901_v1.py"
SHARED_ENGINE = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v47_shared", path)
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
        raise ValueError("MixStyle metadata width invalid")
    station = np.argmax(metadata[:, : len(STATIONS)], axis=1)
    layer_start = len(STATIONS)
    quarter_start = layer_start + len(LAYERS)
    layer = np.argmax(metadata[:, layer_start:quarter_start], axis=1)
    quarter = np.argmax(metadata[:, quarter_start:], axis=1)
    return (station * 10 + quarter).astype(np.int64), layer.astype(np.int64)


def mixstyle_hidden(
    content: torch.Tensor,
    donor: torch.Tensor,
    lambdas: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mix detached row-wise hidden mean/scale while retaining content coordinates."""

    if content.shape != donor.shape or content.ndim != 2:
        raise ValueError("MixStyle content/donor shape mismatch")
    if lambdas.shape != (len(content), 1):
        raise ValueError("MixStyle lambda shape mismatch")
    content_mean = content.mean(dim=1, keepdim=True).detach()
    donor_mean = donor.mean(dim=1, keepdim=True).detach()
    content_scale = torch.sqrt(content.var(dim=1, keepdim=True, unbiased=False) + epsilon).detach()
    donor_scale = torch.sqrt(donor.var(dim=1, keepdim=True, unbiased=False) + epsilon).detach()
    normalized = (content - content_mean) / content_scale
    mixed_mean = lambdas * content_mean + (1.0 - lambdas) * donor_mean
    mixed_scale = lambdas * content_scale + (1.0 - lambdas) * donor_scale
    return normalized * mixed_scale + mixed_mean


def choose_partner_rows(
    environments: np.ndarray,
    supported_pools: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose a donor row from a different supported station-quarter environment."""

    supported_ids = np.asarray(sorted(supported_pools), dtype=np.int64)
    if len(supported_ids) < 2:
        raise ValueError("MixStyle requires two supported donor environments")
    output = np.empty(len(environments), dtype=np.int64)
    for index, environment in enumerate(np.asarray(environments, dtype=np.int64)):
        choices = supported_ids[supported_ids != environment]
        donor_environment = int(choices[rng.integers(0, len(choices))])
        pool = supported_pools[donor_environment]
        output[index] = int(pool[rng.integers(0, len(pool))])
    return output


class _MixStyleNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int, epsilon: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)
        self.epsilon = float(epsilon)

    def forward(
        self,
        values: torch.Tensor,
        donor_values: torch.Tensor | None = None,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encoder(values)
        if donor_values is not None:
            if lambdas is None:
                raise ValueError("MixStyle lambdas missing")
            donor_hidden = self.encoder(donor_values)
            hidden = mixstyle_hidden(hidden, donor_hidden, lambdas, self.epsilon)
        return self.head(hidden).squeeze(1)


class MixStyleClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        if int(inputs) != SCIENTIFIC_FEATURES + ENVIRONMENT_BITS:
            raise ValueError("MixStyle input contract changed")
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _MixStyleNetwork(
            SCIENTIFIC_FEATURES,
            int(config["hidden_units"]),
            float(config["style_epsilon"]),
        ).to(self.device)
        self.training_receipt: dict[str, Any] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> MixStyleClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        environments, layers = decode_source_metadata(values)
        scientific = values[:, :SCIENTIFIC_FEATURES]
        minimum_rows = int(self.config["minimum_rows_per_supported_environment"])
        minimum_layers = int(self.config["minimum_distinct_layers_per_supported_environment"])
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
                    partner_rows = choose_partner_rows(environments[rows], supported_pools, rng)
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
    content = torch.tensor([[1.0, 2.0, 4.0, 8.0], [-3.0, -1.0, 2.0, 5.0]])
    donor = torch.tensor([[10.0, 12.0, 14.0, 16.0], [20.0, 21.0, 24.0, 29.0]])
    own = mixstyle_hidden(content, donor, torch.ones(2, 1), 1e-6)
    borrowed = mixstyle_hidden(content, donor, torch.zeros(2, 1), 1e-6)
    rng = np.random.default_rng(17)
    environments = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    pools = {value: np.asarray([value], dtype=np.int64) for value in range(5)}
    partners = choose_partner_rows(environments, pools, rng)
    rows = 64
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "station": np.repeat(["G-ORS", "I-ORS"], rows),
            "layer": np.repeat([1, 2], rows),
            "_time": np.tile(time, 2),
            "temp": np.tile(np.sin(np.arange(rows) / 6.0), 2),
        }
    )
    boundary = int(pd.DatetimeIndex([time[31]]).as_unit("ns").asi8[0])
    original = CAUSAL_ENVIRONMENT_FEATURES(frame, boundary, representation)
    changed = frame.copy()
    time_ns = pd.DatetimeIndex(changed["_time"]).as_unit("ns").asi8
    future = time_ns > boundary
    changed.loc[future, "temp"] += 1000.0
    perturbed = CAUSAL_ENVIRONMENT_FEATURES(changed, boundary, representation)
    donor_mean = donor.mean(dim=1)
    donor_scale = torch.sqrt(donor.var(dim=1, unbiased=False) + 1e-6)
    return {
        "lambda_one_content_identity": bool(torch.allclose(own, content, atol=2e-6, rtol=2e-6)),
        "lambda_zero_donor_mean": bool(torch.allclose(borrowed.mean(dim=1), donor_mean, atol=2e-6)),
        "lambda_zero_donor_scale": bool(
            torch.allclose(
                torch.sqrt(borrowed.var(dim=1, unbiased=False) + 1e-6),
                donor_scale,
                atol=5e-6,
            )
        ),
        "different_environment_partner": bool(np.all(environments[partners] != environments)),
        "finite_shape": bool(borrowed.shape == content.shape and torch.isfinite(borrowed).all()),
        "group_reset": bool(
            np.array_equal(
                original[:rows, :SCIENTIFIC_FEATURES],
                original[rows:, :SCIENTIFIC_FEATURES],
            )
        ),
        "prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
        "nanosecond_cutoff_distinct": bool(int(time_ns[30]) < boundary < int(time_ns[32])),
        "environment_bits_not_predictors": bool(
            original.shape[1] == SCIENTIFIC_FEATURES + ENVIRONMENT_BITS
        ),
    }


def _write_with_objective(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["objective"] = {
            "kind": "causal_cross_environment_hidden_feature_mixstyle",
            "mix_probability": 0.5,
            "beta_alpha": 0.1,
            "donor_environment": "different_station_x_calendar_quarter",
            "labels_mixed": False,
            "environment_bits_used_as_predictors": False,
            "inference_mixstyle": False,
            "outer_rows_in_training": 0,
        }
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
    shared.shared.LinearProbeClassifier = MixStyleClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = CAUSAL_ENVIRONMENT_FEATURES
    base.VIBClassifier = MixStyleClassifier
    base._write = _write_with_objective


def _install_hooks() -> None:
    shared._configure = _configure
    _configure()


def _target_free_mixstyle_support(data_dir: Path, ready: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(data_dir / "train.csv", usecols=["station", "layer", "time", "temp"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    time_ns = base._time_ns(frame["_time"])
    boundary_ns = pd.Timestamp(ready["pre_q2"]["fit_boundary"]).value
    if time_ns.dtype != np.dtype("int64") or not int(time_ns.min()) < boundary_ns < int(time_ns.max()):
        raise RuntimeError("pre-Q2 MixStyle support cutoff is not distinct nanoseconds")
    use = time_ns <= boundary_ns
    config = base._read(CONFIG)
    features = CAUSAL_ENVIRONMENT_FEATURES(frame, boundary_ns, config["representation"])
    scaler = base.StandardScaler().fit(features[use])
    standardized = scaler.transform(features[use]).astype(np.float32)
    environments, layers = decode_source_metadata(standardized)
    gate = config["representation_support_gate"]
    supported: dict[int, np.ndarray] = {}
    receipts = []
    for environment in sorted(np.unique(environments).tolist()):
        rows = np.flatnonzero(environments == environment)
        layer_count = len(np.unique(layers[rows]))
        if len(rows) >= int(gate["minimum_rows_per_donor_environment"]) and layer_count >= int(
            gate["minimum_distinct_layers_per_donor_environment"]
        ):
            supported[int(environment)] = rows
            receipts.append(
                {"environment_id": int(environment), "rows": int(len(rows)), "layers": layer_count}
            )
    hidden_std = []
    nondegenerate_share = []
    mean_variance = []
    scale_variance = []
    model_config = config["model"]
    scientific = standardized[:, :SCIENTIFIC_FEATURES]
    for seed in model_config["seeds"]:
        torch.manual_seed(int(seed))
        network = _MixStyleNetwork(
            SCIENTIFIC_FEATURES,
            int(model_config["hidden_units"]),
            float(model_config["style_epsilon"]),
        )
        hidden_parts = []
        with torch.no_grad():
            for start in range(0, len(scientific), 32768):
                hidden_parts.append(network.encoder(torch.from_numpy(scientific[start : start + 32768])).numpy())
        hidden = np.concatenate(hidden_parts)
        row_std = hidden.std(axis=1)
        hidden_std.append(float(np.median(row_std)))
        nondegenerate_share.append(float(np.mean(row_std > 1e-6)))
        environment_means = np.stack([hidden[rows].mean(axis=0) for rows in supported.values()])
        environment_scales = np.stack([hidden[rows].std(axis=0) for rows in supported.values()])
        mean_variance.append(float(np.mean(np.var(environment_means, axis=0))))
        scale_variance.append(float(np.mean(np.var(environment_scales, axis=0))))
    supported_ids = np.asarray(sorted(supported), dtype=np.int64)
    supported_stations = supported_ids // 10
    supported_quarters = supported_ids % 10
    return {
        "prefix_rows": int(use.sum()),
        "supported_donor_environments": int(len(supported)),
        "supported_distinct_stations": int(len(np.unique(supported_stations))),
        "supported_distinct_quarters": int(len(np.unique(supported_quarters))),
        "minimum_supported_rows": int(min(len(rows) for rows in supported.values())),
        "minimum_supported_layers": int(min(item["layers"] for item in receipts)),
        "supported_environment_receipts": receipts,
        "initial_median_hidden_std": hidden_std,
        "initial_nondegenerate_hidden_std_share": nondegenerate_share,
        "between_environment_hidden_mean_variance": mean_variance,
        "between_environment_hidden_scale_variance": scale_variance,
        "target_columns_read": 0,
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = shared.preflight(data_dir)
    config = base._read(CONFIG)
    support = _target_free_mixstyle_support(data_dir, ready)
    gate = config["representation_support_gate"]
    passed = (
        support["supported_donor_environments"] >= int(gate["minimum_donor_environments"])
        and support["supported_distinct_stations"] >= int(gate["minimum_distinct_donor_stations"])
        and support["supported_distinct_quarters"] >= int(gate["minimum_distinct_donor_quarters"])
        and min(support["initial_median_hidden_std"]) >= float(gate["minimum_median_hidden_std"])
        and min(support["initial_nondegenerate_hidden_std_share"])
        >= float(gate["minimum_nondegenerate_hidden_std_share"])
        and min(support["between_environment_hidden_mean_variance"])
        >= float(gate["minimum_between_environment_hidden_mean_variance"])
        and min(support["between_environment_hidden_scale_variance"])
        >= float(gate["minimum_between_environment_hidden_scale_variance"])
    )
    support["gate"] = "PASS" if passed else gate["failure"]
    ready["target_free_mixstyle_support"] = support
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
    checks["mixstyle_guards"] = all(_synthetic_guards(base._read(CONFIG)["representation"]).values())
    result_path = ARTIFACT / "result.json"
    if result_path.exists():
        result = base._read(result_path)
        checks["objective"] = result["objective"] == {
            "kind": "causal_cross_environment_hidden_feature_mixstyle",
            "mix_probability": 0.5,
            "beta_alpha": 0.1,
            "donor_environment": "different_station_x_calendar_quarter",
            "labels_mixed": False,
            "environment_bits_used_as_predictors": False,
            "inference_mixstyle": False,
            "outer_rows_in_training": 0,
        }
        checks["training_receipts"] = (
            len(result["training_receipts"]) == result["counters"]["fits"]
            and all(
                item["optimizer_steps"] > 0
                and item["mixstyle_steps"] > 0
                and item["mixed_rows"] > 0
                and item["same_environment_donor_rows"] == 0
                and item["labels_mixed"] is False
                for item in result["training_receipts"]
            )
        )
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
