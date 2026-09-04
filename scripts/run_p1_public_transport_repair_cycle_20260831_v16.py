"""Two-fit robust GCE affine residual experiment for P1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_incumbent_preserving_mstcn_asrf_v2 as mstcn  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v16"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
MSTCN_CONFIG_PATH = ROOT / "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID


class ContractError(RuntimeError):
    """Frozen v16 contract violation."""


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(base.CALIBRATION_PATH.read_text(encoding="utf-8"))
    smooth_penalty = max(
        float(item["adverse_penalty_points"])
        for item in calibration["observed_pairs"]
        if item["tier_id"] == "SMOOTH_LEARNED_PROFILE"
    )
    model = config["model"]
    checks = {
        "calibration_sha": config["transport_family"]["calibration_sha256"]
        == base.sha256_file(base.CALIBRATION_PATH),
        "family": config["transport_family"]["family_id"] == "P1_GCE_SMOOTH_LINEAR_RESIDUAL",
        "penalty": np.isclose(config["decision_policy"]["transport_penalty_points"], smooth_penalty),
        "raw": np.isclose(
            config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            smooth_penalty + calibration["minimum_calibrated_expected_points_delta"],
        ),
        "width": config["features"]["numeric_feature_count"] == 74
        and config["features"]["encoded_feature_count"] == 165,
        "gce": model["gce_q"] == 0.7 and model["l2"] == 0.001,
        "optimizer": model["optimizer"] == "L-BFGS-B"
        and model["optimizer_max_iterations"] == 500
        and model["optimizer_ftol"] == 1e-12,
        "threshold": model["probability_threshold_inclusive"] == 0.95,
        "no_delete": model["row_deletion"] is False and model["validation_weighting"] is False,
        "fits": config["fit_budget"]["maximum"] == 2,
        "no_tuning": config["validation"]["outer_result_based_tuning"] is False,
    }
    if not all(checks.values()):
        raise ContractError(f"v16 contract mismatch: {checks}")
    return config


def load_feature_surface() -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...], dict[str, Any]]:
    source_config = json.loads(MSTCN_CONFIG_PATH.read_text(encoding="utf-8"))
    metadata_path = ROOT / source_config["immutable_inputs"]["feature_metadata"]["path"]
    cache_path = ROOT / source_config["immutable_inputs"]["feature_cache"]["path"]
    key_path = ROOT / source_config["immutable_inputs"]["feature_key_sidecar"]["path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    numeric_names, projected, dependency = mstcn._feature_dependency_audit(metadata, source_config)
    if len(numeric_names) != 74:
        raise ContractError("mature numeric projection is not 74 columns")
    features = pd.read_parquet(cache_path, columns=list(projected))
    keys = pd.read_parquet(key_path, columns=["ordinal", "station", "year", "layer", "time"])
    if len(features) != len(keys) or not np.array_equal(
        keys["ordinal"].to_numpy(np.int64), np.arange(len(keys), dtype=np.int64)
    ):
        raise ContractError("mature feature/key alignment changed")
    feature_frame = keys.drop(columns="ordinal").copy()
    feature_frame["time"] = pd.to_datetime(feature_frame["time"], utc=True)
    for column in projected:
        feature_frame[column] = features[column].to_numpy()
    anchor_frame = pd.read_parquet(base.ANCHOR_PATH)
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    historical, _ = base.attach_truth(anchor_frame, anchor)
    historical["__position"] = np.arange(len(historical), dtype=np.int64)
    merged = historical.merge(
        feature_frame,
        on=["station", "year", "layer", "time"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    ).sort_values("__position", kind="stable")
    if len(merged) != len(historical) or not np.array_equal(
        merged["__position"].to_numpy(np.int64), np.arange(len(historical), dtype=np.int64)
    ):
        raise ContractError("mature projection does not align to OOF surface")
    return merged.drop(columns="__position"), anchor, numeric_names, dependency


@dataclass(frozen=True)
class PrefixEncoder:
    center: np.ndarray
    scale: np.ndarray
    station_vocab: tuple[str, ...]
    layer_vocab: tuple[str, ...]
    depth_thresholds: tuple[float, float]
    depth_vocab: tuple[str, ...]
    numeric_names: tuple[str, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_mask: np.ndarray, numeric_names: tuple[str, ...]) -> PrefixEncoder:
        values = frame.loc[train_mask, list(numeric_names)].to_numpy(np.float64)
        center = np.nanmedian(values, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        station_vocab = tuple(sorted(frame.loc[train_mask, "station"].astype(str).unique()))
        layer_vocab = tuple(sorted(frame.loc[train_mask, "layer_category"].astype(str).unique()))
        depth = frame.loc[train_mask, "depth_raw"].to_numpy(np.float64)
        finite_depth = depth[np.isfinite(depth)]
        if not len(finite_depth):
            raise ContractError("training prefix has no finite depth")
        lower, upper = np.quantile(finite_depth, [1.0 / 3.0, 2.0 / 3.0])
        depth_vocab = ("deep", "mid", "missing", "shallow")
        return cls(
            center=center,
            scale=scale,
            station_vocab=station_vocab,
            layer_vocab=layer_vocab,
            depth_thresholds=(float(lower), float(upper)),
            depth_vocab=depth_vocab,
            numeric_names=numeric_names,
        )

    def transform(self, frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = frame.loc[mask, list(self.numeric_names)].to_numpy(np.float64)
        finite = np.isfinite(values)
        scaled = np.where(finite, (values - self.center) / self.scale, 0.0)
        scaled = np.clip(scaled, -10.0, 10.0)
        missing = (~finite).astype(np.float64)
        row_valid = np.ones((len(values), 1), dtype=np.float64)
        gap_index = self.numeric_names.index("has_gap_before")
        gap = np.where(np.isfinite(values[:, gap_index]), values[:, gap_index], 0.0)[:, None]

        def one_hot(raw: np.ndarray, vocab: tuple[str, ...]) -> np.ndarray:
            mapping = {value: index for index, value in enumerate(vocab)}
            output = np.zeros((len(raw), len(vocab)), dtype=np.float64)
            for row, value in enumerate(raw.astype(str)):
                index = mapping.get(value)
                if index is not None:
                    output[row, index] = 1.0
            return output

        station = one_hot(frame.loc[mask, "station"].astype(str).to_numpy(), self.station_vocab)
        layer = one_hot(frame.loc[mask, "layer_category"].astype(str).to_numpy(), self.layer_vocab)
        depth = frame.loc[mask, "depth_raw"].to_numpy(np.float64)
        lower, upper = self.depth_thresholds
        tokens = np.full(len(depth), "missing", dtype=object)
        finite_depth = np.isfinite(depth)
        tokens[finite_depth & (depth <= lower)] = "shallow"
        tokens[finite_depth & (depth > lower) & (depth <= upper)] = "mid"
        tokens[finite_depth & (depth > upper)] = "deep"
        depth_one_hot = one_hot(tokens.astype(str), self.depth_vocab)
        encoded = np.column_stack([scaled, missing, row_valid, gap, station, layer, depth_one_hot])
        if encoded.shape[1] != 165 or not np.isfinite(encoded).all():
            raise ContractError(
                f"encoded mature projection is not finite width165: {encoded.shape}, "
                f"vocab=({len(self.station_vocab)},{len(self.layer_vocab)},{len(self.depth_vocab)})"
            )
        return encoded, scaled


def leverage_weights(scaled: np.ndarray) -> np.ndarray:
    radius = np.max(np.abs(scaled), axis=1)
    denominator = np.maximum(radius, 8.0)
    return np.maximum(0.25, np.minimum(1.0, 8.0 / denominator))


def gce_objective_gradient(
    parameters: np.ndarray,
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    q: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    coefficients = parameters[:-1]
    intercept = parameters[-1]
    probability = expit(design @ coefficients + intercept)
    target_probability = np.where(labels == 1, probability, 1.0 - probability)
    target_probability = np.clip(target_probability, 1e-12, 1.0)
    denominator = float(weights.sum())
    loss = float(np.dot(weights, (1.0 - target_probability**q) / q) / denominator)
    loss += 0.5 * l2 * float(np.dot(coefficients, coefficients))
    gradient_logit = weights * (probability - labels) * target_probability**q / denominator
    gradient = np.empty_like(parameters)
    gradient[:-1] = design.T @ gradient_logit + l2 * coefficients
    gradient[-1] = gradient_logit.sum()
    return loss, gradient


def fit_gce(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> OptimizeResult:
    model = config["model"]
    initial = np.zeros(design.shape[1] + 1, dtype=np.float64)
    return minimize(
        lambda parameters: gce_objective_gradient(
            parameters,
            design,
            labels,
            weights,
            q=float(model["gce_q"]),
            l2=float(model["l2"]),
        ),
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(model["optimizer_max_iterations"]),
            "ftol": float(model["optimizer_ftol"]),
        },
    )


def train_nested(
    frame: pd.DataFrame,
    anchor: np.ndarray,
    numeric_names: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = frame["label_base"].to_numpy(np.int8)
    candidate = anchor.copy()
    probability = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    threshold = float(config["model"]["probability_threshold_inclusive"])
    for fit_number, spec in enumerate(config["validation"]["nested_fits"], start=1):
        train_mask = frame["fold"].isin(spec["train_folds"]).to_numpy() & (anchor == 0)
        validation_mask = frame["fold"].eq(spec["validation_fold"]).to_numpy()
        validation_negative = validation_mask & (anchor == 0)
        encoder = PrefixEncoder.fit(frame, train_mask, numeric_names)
        train_design, train_scaled = encoder.transform(frame, train_mask)
        validation_design, _ = encoder.transform(frame, validation_negative)
        weights = leverage_weights(train_scaled)
        result = fit_gce(train_design, truth[train_mask], weights, config)
        parameters = result.x.astype(np.float64)
        current_probability = expit(validation_design @ parameters[:-1] + parameters[-1])
        probability[validation_negative] = current_probability
        proposed = np.zeros(len(frame), dtype=bool)
        proposed[validation_negative] = current_probability >= threshold
        candidate[proposed] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": list(spec["train_folds"]),
                "validation_fold": spec["validation_fold"],
                "train_rows": int(train_mask.sum()),
                "train_positives": int(truth[train_mask].sum()),
                "validation_rows": int(validation_mask.sum()),
                "validation_anchor_negative_rows": int(validation_negative.sum()),
                "sealed_additions": int(proposed.sum()),
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "optimizer_message": str(result.message),
                "optimizer_iterations": int(result.nit),
                "optimizer_function_evaluations": int(result.nfev),
                "optimizer_final_objective": float(result.fun),
                "minimum_train_weight": float(weights.min()),
                "downweighted_train_rows": int((weights < 1.0).sum()),
                "parameters_sha256": sha256_array(parameters),
                "probability_sha256": sha256_array(current_probability),
                "candidate_bits_sha256": base.sha256_bool(candidate[validation_mask]),
                "outer_target_reads_before_prediction_seal": 0,
                "encoder_center_sha256": sha256_array(encoder.center),
                "encoder_scale_sha256": sha256_array(encoder.scale),
            }
        )
    return candidate, probability, receipts


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "exactly_two_fits": result["fit_count"] == 2 and len(result["nested_fit_receipts"]) == 2,
        "encoded_width165": result["feature_contract"]["encoded_feature_count"] == 165,
        "fixed_gce": result["model_contract"]["gce_q"] == 0.7,
        "fixed_threshold": result["model_contract"]["probability_threshold_inclusive"] == 0.95,
        "no_row_deletion": result["model_contract"]["row_deletion"] is False,
        "outer_targets_zero_before_seals": all(
            item["outer_target_reads_before_prediction_seal"] == 0 for item in result["nested_fit_receipts"]
        ),
        "add_only": result["candidate"]["anchor_removals"] == 0,
        "official_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "upload_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    load_contract()
    for path in (MSTCN_CONFIG_PATH, base.ANCHOR_PATH, base.CALIBRATION_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": base.sha256_file(CONFIG_PATH),
        "runner_sha256": base.sha256_file(Path(__file__)),
        "candidate_count": 1,
        "fit_budget": 2,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    base.write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "fit_budget": 2,
            "official_covariate_reads": 0,
        },
    )
    base.write_json(ARTIFACT / "progress.json", {"phase": "load_frozen_165_projection", "fit_count": 0}, exclusive=False)
    frame, anchor, numeric_names, dependency = load_feature_surface()
    candidate, probability, receipts = train_nested(frame, anchor, numeric_names, config)
    prediction_path = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(prediction_path, candidate=candidate, probability=probability)
    prediction_seal = {
        "candidate_sha256": base.sha256_bool(candidate),
        "probability_sha256": sha256_array(probability),
        "npz_sha256": base.sha256_file(prediction_path),
        "q3_outer_target_reads_before_seal": 0,
        "q4_outer_target_reads_before_seal": 0,
    }
    base.write_json(ARTIFACT / "prediction_seal.json", prediction_seal)
    base.write_json(ARTIFACT / "progress.json", {"phase": "prediction_sealed", "fit_count": 2}, exclusive=False)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = "P1_1_CAUSAL_GCE_LINEAR_RESIDUAL"
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v16.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "feature_contract": config["features"],
        "model_contract": config["model"],
        "safety_contract": config["safety"],
        "decision_policy": config["decision_policy"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 2,
        "candidate": record,
        "nested_fit_receipts": receipts,
        "prediction_seal": prediction_seal,
        "source_feature_dependency_receipt": dependency,
        "outputs": [],
        "adaptive_surface_disclaimer": "This is newly preregistered development evidence on reused historical Q3/Q4 surfaces, not an independent confirmation.",
        "operations": {
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": base.sha256_file(CONFIG_PATH),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "mstcn_source_config_sha256": base.sha256_file(MSTCN_CONFIG_PATH),
            "anchor_sha256": base.sha256_file(base.ANCHOR_PATH),
            "root_calibration_sha256": base.sha256_file(base.CALIBRATION_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    base.write_json(ARTIFACT / "result.json", result)
    base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]},
        exclusive=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
