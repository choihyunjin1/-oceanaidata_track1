"""Run the preregistered nested P1 direct interval-set PyTorch experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p1_qc.direct_interval_set_torch import (
    DirectIntervalConfig,
    DirectIntervalSetPredictor,
    interval_metrics,
    interval_set_loss,
    predict,
    synthetic_feasibility_smoke,
)
from p1_qc.long_event_change_point_rescue import binary_metrics

EXPERIMENT_ID = "p1_direct_interval_set_torch_nested_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
SOURCE_RUNNER = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
SOURCE_CONFIG = ROOT / "configs" / "experiments" / "p1_incumbent_preserving_mstcn_asrf_v2.json"
SOURCE_ARTIFACT = ROOT / "artifacts" / "p1_incumbent_preserving_mstcn_asrf_v2"
CHECKPOINT_ARTIFACT = ROOT / "artifacts" / "p1_mstcn_checkpoint_diagnostic_20260827_v2"


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config["anchor"]["positive_removal_allowed"] or not config["anchor"]["exact_zero_add_no_op_arm"]:
        raise ContractError("anchor/no-op contract changed")
    return config


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_e150_anchor(source: Any, surfaces: Any, config: dict[str, Any]) -> dict[str, np.ndarray]:
    anchor = config["anchor"]
    grid = source.load_sealed_q2_grid(SOURCE_ARTIFACT / "q2_qualification_grid_receipt.json")
    capacity = np.flatnonzero((grid.widths == int(anchor["width"])) & (grid.epochs == int(anchor["epoch"])))
    threshold = np.flatnonzero(np.isclose(grid.thresholds, float(anchor["threshold"]), atol=0.0, rtol=0.0))
    if len(capacity) != 1 or len(threshold) != 1:
        raise ContractError("Q2 e150 anchor is not unique")
    result = {"q2": grid.candidate[int(capacity[0]), int(threshold[0])].astype(np.int8)}
    for phase in ("q3", "q4"):
        receipt = json.loads((CHECKPOINT_ARTIFACT / f"{phase}_blind_checkpoint_curve_receipt.json").read_text(encoding="utf-8"))
        score_path = CHECKPOINT_ARTIFACT / receipt["score_path"]
        if score_path.stat().st_size != int(receipt["score_bytes"]) or _sha256(score_path) != receipt["score_sha256"]:
            raise ContractError(f"{phase} checkpoint surface changed")
        with np.load(score_path, allow_pickle=False) as archive:
            positions = np.flatnonzero(archive["epochs"] == int(anchor["epoch"]))
            if len(positions) != 1:
                raise ContractError(f"{phase} e150 anchor is not unique")
            result[phase] = archive["candidate"][int(positions[0])].astype(np.int8)
    expected = {"q2": surfaces.q2.rows, "q3": surfaces.q3.rows, "q4": surfaces.q4.rows}
    for phase, rows in expected.items():
        if result[phase].shape != (rows,) or not np.isin(result[phase], [0, 1]).all():
            raise ContractError(f"{phase} e150 anchor invalid")
    return result


def _subset_bundle(engine: Any, bundle: Any, mask: np.ndarray) -> Any:
    indices = np.flatnonzero(mask)
    return engine.TrainingBundle(
        bundle.features[indices.tolist()],
        bundle.keys[indices.tolist()],
        bundle.labels[indices.tolist()],
    )


def _time_array(engine: Any, keys: Any) -> np.ndarray:
    return np.asarray([engine._to_datetime(value) for value in keys.get_column("time")], dtype=object)


def _build_labeled_windows(engine: Any, bundle: Any, config: dict[str, Any], *, censored_at: datetime | None) -> list[Any]:
    segments = engine.exact_cadence_segments(bundle.keys)
    events = engine.eligible_target_events(
        segments,
        bundle.labels,
        minimum_original_rows=int(config["target"]["minimum_original_rows"]),
        right_censor_cutoff=censored_at,
    )
    spec = config["windowing"]
    return engine.build_windows(
        segments,
        events,
        window_length=int(spec["rows"]),
        stride=int(spec["stride_rows"]),
        max_queries=int(spec["queries"]),
    )


def _training_sample(engine: Any, windows: list[Any], config: dict[str, Any]) -> list[Any]:
    return engine.deterministic_training_sample(
        windows,
        seed=int(config["training"]["seed"]),
        empty_ratio=int(config["training"]["negative_to_positive_window_ratio"]),
    )


def _new_model(config: dict[str, Any], input_features: int, device: str) -> DirectIntervalSetPredictor:
    window = config["windowing"]
    model = config["model"]
    torch.manual_seed(int(config["training"]["seed"]))
    return DirectIntervalSetPredictor(
        DirectIntervalConfig(
            input_features=input_features,
            window_rows=int(window["rows"]),
            patch_rows=int(window["patch_rows"]),
            d_model=int(model["d_model"]),
            heads=int(model["heads"]),
            encoder_layers=int(model["encoder_layers"]),
            queries=int(window["queries"]),
            dropout=float(model["dropout"]),
        )
    ).to(device)


def _fit_restore_inner_best(
    config: dict[str, Any],
    train_features: np.ndarray,
    train_targets: list[np.ndarray],
    inner_features: np.ndarray,
    inner_targets: list[np.ndarray],
    *,
    device: str,
) -> tuple[DirectIntervalSetPredictor, dict[str, Any]]:
    training = config["training"]
    model = _new_model(config, train_features.shape[2], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(training["seed"]))
    batch_size = int(training["batch_size"])
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(train_features))
        weighted, seen = 0.0, 0
        for offset in range(0, len(order), batch_size):
            index = order[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, intervals, patch_logits = model(torch.as_tensor(train_features[index], device=device))
            loss = interval_set_loss(
                logits,
                intervals,
                patch_logits,
                [train_targets[int(value)] for value in index],
                positive_weight=float(training["positive_weight"]),
                endpoint_weight=float(training["endpoint_weight"]),
                iou_weight=float(training["iou_weight"]),
                actionness_weight=float(training["actionness_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise ContractError("non-finite training loss")
            weighted += value * len(index)
            seen += len(index)
        predicted, scores = predict(model, inner_features, batch_size=batch_size)
        metrics = interval_metrics(predicted, scores, inner_targets, threshold=float(config["model"]["fixed_query_threshold"]), iou_cutoff=0.7)
        empty = max(1, sum(len(value) == 0 for value in inner_targets))
        selection_score = float(metrics["target_recall"]) + 0.25 * float(metrics["median_matched_iou"]) - 0.01 * float(metrics["negative_window_fp"]) / empty
        history.append({"epoch": epoch, "loss": weighted / seen, "selection_score": selection_score, **metrics})
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise ContractError("inner checkpoint was never selected")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {"best_epoch": best_epoch, "best_selection_score": best_score, "best_inner_metrics": history[best_epoch - 1], "epochs_ran": len(history)}


@dataclass
class BlindPrediction:
    phase: str
    keys: Any
    anchor: np.ndarray
    proposal: np.ndarray
    confidence: np.ndarray
    training_receipt: dict[str, Any]


def _fit_predict_phase(
    engine: Any,
    inputs: Any,
    truth_oof: Any,
    numeric_features: Sequence[str],
    config: dict[str, Any],
    phase: str,
    anchor: np.ndarray,
    *,
    device: str,
) -> BlindPrediction:
    outer_cutoff = _parse_time(config["chronological_protocol"][phase]["training_rows_time_before"])
    window = config["windowing"]
    bundle = engine.load_training_prefix_bundle(inputs, numeric_features, cutoff=outer_cutoff)
    times = _time_array(engine, bundle.keys)
    train_end = outer_cutoff - timedelta(days=int(window["inner_train_end_days_before_outer"]))
    inner_start = outer_cutoff - timedelta(days=int(window["inner_validation_start_days_before_outer"]))
    inner_end = outer_cutoff - timedelta(days=int(window["inner_validation_end_days_before_outer"]))
    train_bundle = _subset_bundle(engine, bundle, np.asarray([value < train_end for value in times], dtype=bool))
    inner_bundle = _subset_bundle(engine, bundle, np.asarray([inner_start <= value < inner_end for value in times], dtype=bool))
    train_windows = _training_sample(engine, _build_labeled_windows(engine, train_bundle, config, censored_at=train_end), config)
    inner_windows = _build_labeled_windows(engine, inner_bundle, config, censored_at=inner_end)
    if not train_windows or not inner_windows or not any(len(value.targets) for value in inner_windows):
        raise ContractError(f"{phase}: insufficient nested train/inner windows")
    train_indices = np.arange(train_bundle.rows, dtype=np.int64)
    preprocessor = engine.RobustPreprocessor.fit(
        train_bundle,
        train_indices,
        numeric_features,
        train_end=max(_time_array(engine, train_bundle.keys)),
    )
    train_features = engine.materialize_windows(train_bundle, train_windows, preprocessor, window_length=int(window["rows"]))
    inner_features = engine.materialize_windows(inner_bundle, inner_windows, preprocessor, window_length=int(window["rows"]))
    model, checkpoint = _fit_restore_inner_best(
        config,
        train_features,
        [value.targets for value in train_windows],
        inner_features,
        [value.targets for value in inner_windows],
        device=device,
    )
    del train_features, inner_features
    validation = engine.load_validation_feature_bundle(inputs, truth_oof, numeric_features, fold=f"2025_{phase}")
    validation_windows = engine.build_windows(
        engine.exact_cadence_segments(validation.keys),
        (),
        window_length=int(window["rows"]),
        stride=int(window["stride_rows"]),
        max_queries=int(window["queries"]),
    )
    validation_features = engine.materialize_windows(validation, validation_windows, preprocessor, window_length=int(window["rows"]))
    intervals, scores = predict(model, validation_features, batch_size=int(config["training"]["batch_size"]))
    confidence, proposal = engine.stitch_proposals(
        validation_windows,
        intervals,
        scores,
        total_rows=validation.rows,
        threshold=float(config["model"]["fixed_query_threshold"]),
        minimum_decoded_rows=int(config["model"]["minimum_decoded_rows"]),
        coordinate_length=int(window["rows"]),
    )
    if anchor.shape != proposal.shape:
        raise ContractError(f"{phase}: anchor/proposal shape differs")
    return BlindPrediction(
        phase,
        validation.keys,
        anchor,
        proposal,
        confidence,
        {
            **checkpoint,
            "training_windows": len(train_windows),
            "inner_windows": len(inner_windows),
            "outer_windows": len(validation_windows),
            "train_end": train_end.isoformat(),
            "inner_start": inner_start.isoformat(),
            "inner_end": inner_end.isoformat(),
        },
    )


def _key_sha(keys: Any) -> str:
    digest = hashlib.sha256()
    for row in keys.select(["station", "year", "layer", "time"]).cast({name: keys.schema[name] for name in keys.columns if name in {"station", "year", "layer", "time"}}).iter_rows():
        digest.update("\x1f".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _commit_blind(blind: BlindPrediction) -> Path:
    path = ARTIFACT_DIR / f"{blind.phase}_blind_prediction.npz"
    np.savez_compressed(path, confidence=blind.confidence.astype(np.float32), proposal=blind.proposal.astype(np.int8), anchor=blind.anchor.astype(np.int8))
    _atomic_json(
        ARTIFACT_DIR / f"{blind.phase}_blind_prediction_receipt.json",
        {
            "phase": blind.phase,
            "score_path": path.name,
            "score_bytes": path.stat().st_size,
            "score_sha256": _sha256(path),
            "ordered_key_sha256": _key_sha(blind.keys),
            "truth_columns_read_before_receipt": 0,
            "training": blind.training_receipt,
        },
    )
    return path


def _load_truth(engine: Any, truth_oof: Any, blind: BlindPrediction) -> np.ndarray:
    receipt = json.loads((ARTIFACT_DIR / f"{blind.phase}_blind_prediction_receipt.json").read_text(encoding="utf-8"))
    if receipt["ordered_key_sha256"] != _key_sha(blind.keys):
        raise ContractError("blind key receipt changed")
    truth = engine.load_validation_fold_truth(truth_oof, f"2025_{blind.phase}")
    for name in engine.KEY_COLUMNS:
        if not np.array_equal(truth.get_column(name).cast(engine.pl.String).to_numpy(), blind.keys.get_column(name).cast(engine.pl.String).to_numpy()):
            raise ContractError(f"{blind.phase}: truth key binding differs")
    return truth.get_column("label").to_numpy().astype(np.int8)


def _score(blind: BlindPrediction, truth: np.ndarray, *, use_model: bool) -> dict[str, Any]:
    additions = blind.proposal if use_model else np.zeros_like(blind.proposal)
    candidate = np.maximum(blind.anchor, additions).astype(np.int8)
    if np.any((blind.anchor == 1) & (candidate == 0)):
        raise ContractError("anchor positive was removed")
    base = binary_metrics(truth, blind.anchor)
    new = binary_metrics(truth, candidate)
    changed = (blind.anchor == 0) & (candidate == 1)
    added_tp = int(np.sum(changed & (truth == 1)))
    added_fp = int(np.sum(changed & (truth == 0)))
    return {
        "anchor": base,
        "candidate": new,
        "delta_f1": float(new["f1"] - base["f1"]),
        "delta_recall": float(new["recall"] - base["recall"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 1.0,
        "anchor_positive_removed_rows": int(np.sum((blind.anchor == 1) & (candidate == 0))),
        "candidate_array": candidate,
    }


def smoke() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = synthetic_feasibility_smoke(device=device, seed=20260828)
    return {"experiment_id": EXPERIMENT_ID, "device": device, "smoke": metrics, "result": "PASS" if metrics["passed"] else "FAIL"}


def check_only() -> dict[str, Any]:
    _load_config()
    required = [SOURCE_RUNNER, SOURCE_CONFIG, SOURCE_ARTIFACT / "q2_qualification_grid_receipt.json", CHECKPOINT_ARTIFACT / "q3_blind_checkpoint_curve_receipt.json", CHECKPOINT_ARTIFACT / "q4_blind_checkpoint_curve_receipt.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"missing immutable dependency: {missing}")
    if ARTIFACT_DIR.exists():
        raise ContractError("append-only artifact namespace already exists")
    return {"experiment_id": EXPERIMENT_ID, "config_sha256": _sha256(CONFIG_PATH), "module_sha256": _sha256(ROOT / "src" / "p1_qc" / "direct_interval_set_torch.py"), "runner_sha256": _sha256(Path(__file__)), "official_test_reads": 0, "result": "PASS"}


def execute() -> dict[str, Any]:
    preflight = check_only()
    smoke_result = smoke()
    if smoke_result["result"] != "PASS":
        return {"experiment_id": EXPERIMENT_ID, "status": "NO_GO_FEASIBILITY_SMOKE", **smoke_result}
    config = _load_config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    _atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
    _atomic_json(ARTIFACT_DIR / "feasibility_smoke.json", smoke_result)
    engine = _load_module(ROOT / "src" / "p1_qc" / "tetad_lite_experiment.py", f"{EXPERIMENT_ID}_engine")
    source = _load_module(SOURCE_RUNNER, f"{EXPERIMENT_ID}_source")
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    surfaces = source.load_blind_surfaces(source_config, root=ROOT)
    anchors = _load_e150_anchor(source, surfaces, config)
    inputs, truth_oof, _parts = engine.canonical_frozen_inputs(ROOT)
    numeric_features = tuple(json.loads((ROOT / "configs" / "experiments" / "p1_tetad_lite_direct_interval_set_v1.json").read_text(encoding="utf-8"))["selected_numeric_features"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q2 = _fit_predict_phase(engine, inputs, truth_oof, numeric_features, config, "q2", anchors["q2"], device=device)
    _commit_blind(q2)
    q2_truth = _load_truth(engine, truth_oof, q2)
    q2_model = _score(q2, q2_truth, use_model=True)
    q2_noop = _score(q2, q2_truth, use_model=False)
    gates = config["qualification"]["q2_choose_non_noop_if_all"]
    use_model = bool(q2_model["delta_f1"] > float(gates["delta_f1_strictly_above"]) and q2_model["added_precision"] >= float(gates["added_precision_gte"]) and q2_model["added_fp"] <= int(gates["added_false_positive_rows_lte"]))
    _atomic_json(ARTIFACT_DIR / "q2_arm_selection.json", {"selected_arm": "MODEL_UNION" if use_model else "ZERO_ADD_NO_OP", "model": {key: value for key, value in q2_model.items() if key != "candidate_array"}, "no_op": {key: value for key, value in q2_noop.items() if key != "candidate_array"}, "threshold": config["model"]["fixed_query_threshold"]})
    if not use_model:
        terminal = {"experiment_id": EXPERIMENT_ID, "status": "NO_GO_Q2_EXACT_NO_OP_SELECTED", "official_test_reads": 0, "submission_files_created": 0, "uploads": 0}
        _atomic_json(ARTIFACT_DIR / "terminal_result.json", terminal)
    else:
        fold_scores: dict[str, dict[str, Any]] = {}
        candidates: dict[str, np.ndarray] = {}
        for phase in ("q3", "q4"):
            blind = _fit_predict_phase(engine, inputs, truth_oof, numeric_features, config, phase, anchors[phase], device=device)
            _commit_blind(blind)
            truth = _load_truth(engine, truth_oof, blind)
            scored = _score(blind, truth, use_model=True)
            candidates[phase] = scored.pop("candidate_array")
            fold_scores[phase] = scored
        pooled_truth = np.concatenate([_load_truth(engine, truth_oof, _fit) for _fit in []]) if False else None
        pooled_anchor = np.concatenate([anchors["q3"], anchors["q4"]])
        truth_parts = [engine.load_validation_fold_truth(truth_oof, f"2025_{phase}").get_column("label").to_numpy().astype(np.int8) for phase in ("q3", "q4")]
        pooled_truth = np.concatenate(truth_parts)
        pooled_candidate = np.concatenate([candidates["q3"], candidates["q4"]])
        pooled_base = binary_metrics(pooled_truth, pooled_anchor)
        pooled_new = binary_metrics(pooled_truth, pooled_candidate)
        changed = (pooled_anchor == 0) & (pooled_candidate == 1)
        tp = int(np.sum(changed & (pooled_truth == 1)))
        fp = int(np.sum(changed & (pooled_truth == 0)))
        outer = {
            "folds": fold_scores,
            "pooled": {
                "anchor": pooled_base,
                "candidate": pooled_new,
                "delta_f1": float(pooled_new["f1"] - pooled_base["f1"]),
                "delta_recall": float(pooled_new["recall"] - pooled_base["recall"]),
                "added_rows": int(changed.sum()),
                "added_tp": tp,
                "added_fp": fp,
                "added_precision": tp / (tp + fp) if tp + fp else 1.0,
                "anchor_positive_removed_rows": int(np.sum((pooled_anchor == 1) & (pooled_candidate == 0))),
            },
        }
        blind_gates = config["qualification"]["blind_go_if_all"]
        checks = {
            "pooled_delta": outer["pooled"]["delta_f1"] >= float(blind_gates["pooled_delta_f1_gte"]),
            "each_fold": all(fold_scores[phase]["delta_f1"] >= float(blind_gates["each_fold_delta_f1_gte"]) for phase in ("q3", "q4")),
            "added_precision": outer["pooled"]["added_precision"] >= float(blind_gates["added_precision_gte"]),
            "anchor_preserved": outer["pooled"]["anchor_positive_removed_rows"] == int(blind_gates["anchor_positive_removed_rows_eq"]),
        }
        outer["gate_checks"] = checks
        outer["decision"] = "GO_LOCAL_BLIND_GATE" if all(checks.values()) else "NO_GO_LOCAL_BLIND_GATE"
        _atomic_json(ARTIFACT_DIR / "outer_metrics.json", outer)
        terminal = {"experiment_id": EXPERIMENT_ID, "status": "COMPLETE_LOCAL_BLIND_SCREEN", "decision": outer["decision"], "official_test_reads": 0, "submission_files_created": 0, "uploads": 0}
        _atomic_json(ARTIFACT_DIR / "terminal_result.json", terminal)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "module_sha256": _sha256(ROOT / "src" / "p1_qc" / "direct_interval_set_torch.py"),
        "runner_sha256": _sha256(Path(__file__)),
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(ARTIFACT_DIR.iterdir()) if path.is_file() and path.name != "manifest.json"},
    }
    _atomic_json(ARTIFACT_DIR / "manifest.json", manifest)
    return terminal | {"manifest_sha256": _sha256(ARTIFACT_DIR / "manifest.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = smoke() if args.smoke else check_only() if args.check_only else execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
