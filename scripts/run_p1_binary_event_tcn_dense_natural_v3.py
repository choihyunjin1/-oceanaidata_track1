"""Run the append-only P1 dense-natural binary-event TCN Gen3 curve once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p1_qc.binary_event_tcn import (
    BinaryEventModelConfig,
    DenseNaturalTrainingConfig,
    fit_fixed_step_binary_event_model,
    load_fitted_binary_event_model,
    predict_binary_event_probability,
    save_fitted_binary_event_model,
)

_SHARED_RUNNER = PROJECT_ROOT / "scripts/run_p1_station_layer_temporal_convolution_event_v2.py"
_SHARED_SPEC = importlib.util.spec_from_file_location(
    "p1_gen3_shared_curve_engine", _SHARED_RUNNER
)
if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
    raise ImportError("failed to load pinned P1 Gen2 curve engine")
shared = importlib.util.module_from_spec(_SHARED_SPEC)
sys.modules[_SHARED_SPEC.name] = shared
_SHARED_SPEC.loader.exec_module(shared)

EXPECTED_CONFIG_SHA256 = "e52694f1f0ab5bc10f216645f18a2eaac7ea882a2e7ca9ce659d915c7a6ddbb0"
EXPECTED_CONFIG_DEEP_SHA256 = "eeaf64cb70b34213c2115215970904df60cb08453932231caa5f0d445f06fdc6"
CANONICAL_CONFIG = "configs/experiments/p1_binary_event_tcn_dense_natural_v3.json"
CANONICAL_ARTIFACT = "artifacts/p1_binary_event_tcn_dense_natural_v3"
CANONICAL_LOCK = "artifacts/p1_binary_event_tcn_dense_natural_v3.ATTEMPT_LOCK.json"
HYPOTHESIS = "station_layer_binary_event_tcn_aux_boundaries_dense_natural"
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
_SHARED_JSON_NEW = shared._json_new


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "artifact": root / CANONICAL_ARTIFACT,
        "lock": root / CANONICAL_LOCK,
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2": root / "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
    }


def _model_config(
    config: dict[str, Any], feature_count: int, group_count: int
) -> BinaryEventModelConfig:
    model = config["model"]
    result = BinaryEventModelConfig(
        input_feature_count=feature_count,
        group_count=group_count,
        width=int(model["width"]),
        group_embedding_width=int(model["group_embedding_width"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        dropout=float(model["dropout"]),
        norm_groups=int(model["norm_groups"]),
    )
    result.validate()
    if result.receptive_field_rows != model["receptive_field_rows"]:
        raise ValueError("registered receptive field differs from implementation")
    return result


def _training_config(config: dict[str, Any]) -> DenseNaturalTrainingConfig:
    training = config["training"]
    result = DenseNaturalTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_cell"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        auxiliary_loss_weight=float(training["auxiliary_loss_weight"]),
        boundary_band_rows=int(training["boundary_band_rows"]),
    )
    result.validate()
    return result


def authorize_entry(
    *, root: Path, data_dir: Path, requested_config: Path, requested_artifact: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    """Bind Gen3 canonical paths, bytes, structural contract, code, and inputs."""

    root = root.resolve(strict=True)
    paths = _paths(root)
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical config path is forbidden")
    if requested_artifact.resolve(strict=False) != paths["artifact"].resolve(strict=False):
        raise PermissionError("non-canonical artifact path is forbidden")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if shared._deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config deep JSON differs")
    if config["experiment_id"] != "p1_binary_event_tcn_dense_natural_v3":
        raise PermissionError("experiment identity differs")
    if config.get("comparison_mode") != "EXACT_OFFICIAL_PREFIX_REFIT":
        raise PermissionError("P1 Gen3 comparison mode must remain exact")
    if config["canonical_paths"] != {
        "config": CANONICAL_CONFIG,
        "base_config": "configs/p1.toml",
        "goal_contract": "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1_artifact": "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2_artifact": "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "frozen_oof": "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "artifact": CANONICAL_ARTIFACT,
        "attempt_lock": CANONICAL_LOCK,
    }:
        raise PermissionError("canonical path contract differs")
    if [item["id"] for item in config["hypotheses"]] != [HYPOTHESIS]:
        raise PermissionError("single registered hypothesis differs")
    if tuple(config["prefix_fractions"]) != FRACTIONS or tuple(config["seeds"]) != SEEDS:
        raise PermissionError("prefix or seed contract differs")
    training = config["training"]
    if not (
        training["optimizer_steps_per_cell"] == 120
        and training["batch_size"] == 4096
        and training["expected_curve_fit_cells"] == 45
        and training["expected_curve_optimizer_steps"] == 5400
        and training["main_event_loss"]
        == "unweighted_BCEWithLogits_on_dense_natural_prefix_rows"
        and training["phase_balanced_resampling"] is False
        and training["natural_prior_probability_correction"] is False
        and training["hyperparameter_search"] is False
    ):
        raise PermissionError("fixed dense-natural training contract differs")
    model = config["model"]
    if not (
        model["probability_rule"] == "sigmoid(binary_event_logit)_only"
        and model["auxiliary_probability_union_forbidden"] is True
        and model["inference_head"] == "binary_event"
        and model["auxiliary_heads"] == ["onset", "offset"]
    ):
        raise PermissionError("binary event / auxiliary-only head contract differs")
    if not all(value is True for value in config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")
    implementations = {
        "binary_event_module": root / "src/p1_qc/binary_event_tcn.py",
        "shared_temporal_layout_module": root / "src/p1_qc/temporal_event_tcn.py",
        "shared_gen2_runner": _SHARED_RUNNER,
        "gen1_runner": root / "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        "base_config": paths["base_config"],
        "pipeline": root / "src/p1_qc/pipeline.py",
        "validation": root / "src/p1_qc/validation.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
    }
    for name, path in implementations.items():
        if shared._sha(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    pins = shared._verify_input_pins(root, data_dir, config)
    return config, paths, pins


def _json_new(path: Path, value: Any) -> None:
    if path.name == "learning_curve_evidence.json" and isinstance(value, dict):
        value = {**value, "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT"}
    _SHARED_JSON_NEW(path, value)


def _patch_shared_engine() -> None:
    """Bind the pinned generic curve engine to the sealed Gen3 implementation."""

    shared.__file__ = str(Path(__file__).resolve())
    shared.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    shared.EXPECTED_CONFIG_DEEP_SHA256 = EXPECTED_CONFIG_DEEP_SHA256
    shared.CANONICAL_CONFIG = CANONICAL_CONFIG
    shared.CANONICAL_ARTIFACT = CANONICAL_ARTIFACT
    shared.CANONICAL_LOCK = CANONICAL_LOCK
    shared.HYPOTHESIS = HYPOTHESIS
    shared._paths = _paths
    shared._json_new = _json_new
    shared.authorize_entry = authorize_entry
    shared._model_config = _model_config
    shared._training_config = _training_config
    shared.TemporalEventModelConfig = BinaryEventModelConfig
    shared.FixedStepTrainingConfig = DenseNaturalTrainingConfig
    shared.fit_fixed_step_temporal_event_model = fit_fixed_step_binary_event_model
    shared.predict_temporal_event_probability = predict_binary_event_probability
    shared.save_fitted_temporal_event_model = save_fitted_binary_event_model
    shared.load_fitted_temporal_event_model = load_fitted_binary_event_model
    shared.evaluate_learning_curve = evaluate_learning_curve
    shared.load_contract = load_contract


_patch_shared_engine()


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    result = shared.check_only(root=root, data_dir=data_dir)
    config = json.loads((root / CANONICAL_CONFIG).read_text(encoding="utf-8"))
    return {
        **result,
        "experiment_id": config["experiment_id"],
        "comparison_mode": config["comparison_mode"],
        "main_event_loss": config["training"]["main_event_loss"],
        "binary_event_probability_only": True,
        "auxiliary_probability_union": False,
        "dense_natural_batch_size": config["training"]["batch_size"],
    }


def seal(*, root: Path, data_dir: Path) -> dict[str, Any]:
    return shared.seal(root=root, data_dir=data_dir)


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    return shared.run_experiment(root=root, data_dir=data_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--seal-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    if args.check_only:
        result = check_only(root=root, data_dir=data_dir)
    elif args.seal_only:
        result = seal(root=root, data_dir=data_dir)
    else:
        result = run_experiment(root=root, data_dir=data_dir)
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
