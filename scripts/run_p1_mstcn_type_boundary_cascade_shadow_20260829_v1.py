"""Run a one-shot Q3/Q4 shadow screen of type/boundary-aware refinement.

The script reads only already-verified local training/OOF assets.  Both Q3 and
Q4 predictions are trained and sealed before either fold truth is opened.  It
never reads the official P1 test/sample/submission paths and never creates a
deployable CSV.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_type_boundary_cascade_shadow_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
BASE_RUNNER_PATH = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"


class ShadowError(RuntimeError):
    """Raised when a preregistered shadow boundary changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("p1_mstcn_shadow_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ShadowError("cannot load the pinned base runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ShadowError("shadow config identity changed")
    if config["operation_boundaries"] != {
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_csv_creation": False,
        "upload": False,
        "q3_q4_results_may_not_change_this_run": True,
    }:
        raise ShadowError("operation boundaries changed")
    return config


def _checkpoint_state(
    torch: Any, config: dict[str, Any], phase: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    pin = config["warm_start_checkpoints"][phase]
    path = (ROOT / pin["path"]).resolve()
    if _sha256(path) != pin["sha256"]:
        raise ShadowError(f"{phase} warm-start checkpoint identity changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "phase": phase,
        "width": int(config["width"]),
        "seed": int(config["fixed_seed"]),
        "epoch": int(config["baseline_epoch"]),
        "input_features": 165,
    }
    observed = {name: checkpoint.get(name) for name in expected}
    if observed != expected:
        raise ShadowError(f"{phase} checkpoint semantic identity changed: {observed}")
    return checkpoint["state_dict"], {
        "path": pin["path"],
        "sha256": pin["sha256"],
        **expected,
    }


def _decoder_candidate(runner: Any, bundle: Any, encoded: Any, base_config: dict[str, Any], shadow: dict[str, Any]) -> tuple[Any, Any]:
    score = runner._decoder_row_probability(bundle, base_config)  # noqa: SLF001
    decoder = shadow["decoder"]
    proposal = runner.decode_long_event_segments(
        score,
        bundle.boundary_probability,
        encoded.layout,
        high_threshold=float(decoder["high_threshold"]),
        snap_radius=int(decoder["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(decoder["minimum_added_segment_rows"]),
        maximum_rows=None,
    )
    if encoded.surface.anchor is None:
        raise ShadowError("holdout anchor is absent")
    candidate = runner.anchor_preserving_union(encoded.surface.anchor, proposal)
    return proposal, candidate


def _phase_fit(
    *,
    phase: str,
    runner: Any,
    baseline_api: Any,
    cascade_api: Any,
    torch: Any,
    surfaces: Any,
    base_config: dict[str, Any],
    shadow: dict[str, Any],
    device: Any,
    artifact_dir: Path,
    window_selector: Any | None = None,
) -> dict[str, Any]:
    _encoder, training, holdout, split_receipt = runner._prepare_phase_surfaces(  # noqa: SLF001
        surfaces, base_config, phase, root=ROOT
    )
    state_dict, checkpoint_receipt = _checkpoint_state(torch, shadow, phase)
    model_config = baseline_api.MSTCNASRFConfig(
        input_feature_count=int(training.features.shape[1]),
        width=int(shadow["width"]),
        generator_dilations=tuple(int(v) for v in base_config["architecture"]["dual_dilations"]),
        refinement_stages=int(base_config["architecture"]["refinement_stages"]),
        refinement_dilations=tuple(int(v) for v in base_config["architecture"]["dual_dilations"]),
        dropout=float(base_config["architecture"]["dropout"]),
    )
    baseline_model = baseline_api.MSTCNASRF(model_config).to(device)
    baseline_model.load_state_dict(state_dict, strict=True)
    cascade_model = cascade_api.MSTCNASRF(model_config).to(device)
    warm_start = cascade_model.initialize_from_baseline_state_dict(state_dict)
    batch_size = int(base_config["architecture"]["batch_size_by_width"][str(shadow["width"])])
    holdout_windows = runner._all_windows(holdout, base_config)  # noqa: SLF001
    baseline_blind = runner.predict_encoded(
        baseline_model,
        holdout,
        holdout_windows,
        batch_size=batch_size,
        device=device,
    )
    zero_blind = runner.predict_encoded(
        cascade_model,
        holdout,
        holdout_windows,
        batch_size=batch_size,
        device=device,
    )
    import numpy as np

    zero_step = {
        "row_max_abs": float(np.max(np.abs(baseline_blind.row_probability - zero_blind.row_probability))),
        "boundary_max_abs": float(np.max(np.abs(baseline_blind.boundary_probability - zero_blind.boundary_probability))),
        "type_max_abs": float(np.max(np.abs(baseline_blind.type_probability - zero_blind.type_probability))),
    }
    if max(zero_step.values()) > float(shadow["evaluation"]["information_gate"]["zero_step_max_abs_difference"]):
        raise ShadowError(f"{phase} warm start does not exactly reproduce baseline: {zero_step}")

    capacity = runner._config_for_capacity(  # noqa: SLF001
        base_config, width=int(shadow["width"]), seed=int(shadow["fixed_seed"])
    )
    fine_tune = shadow["fine_tune"]
    capacity["training"]["maximum_epochs"] = int(fine_tune["epochs"])
    capacity["training"]["warmup_epochs"] = int(fine_tune["warmup_epochs"])
    capacity["training"]["learning_rate"] = float(fine_tune["learning_rate"])
    capacity["training"]["weight_decay"] = float(fine_tune["weight_decay"])
    capacity["training"]["gradient_clip_norm"] = float(fine_tune["gradient_clip_norm"])
    optimizer = torch.optim.AdamW(
        cascade_model.parameters(),
        lr=float(fine_tune["learning_rate"]),
        weight_decay=float(fine_tune["weight_decay"]),
    )
    if window_selector is None:
        windows = runner._selected_windows(training, capacity)  # noqa: SLF001
        window_selection_receipt = {"mode": "base_runner_selected_windows"}
    else:
        windows, window_selection_receipt = window_selector(
            runner, training, capacity, shadow, phase
        )
    positive_weight = float(
        fine_tune.get(
            "positive_weight",
            runner._positive_weight(training.surface.labels),  # noqa: SLF001
        )
    )
    _steps, total_steps, _warmup = runner._schedule_geometry(  # noqa: SLF001
        capacity, window_count=len(windows)
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, int(fine_tune["epochs"]) + 1):
        started = time.perf_counter()
        telemetry, global_step, learning_rate = runner._train_epoch(  # noqa: SLF001
            cascade_model,
            optimizer,
            training,
            windows,
            config=capacity,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        record = runner._history_record(  # noqa: SLF001
            epoch=epoch,
            telemetry=telemetry,
            global_step=global_step,
            learning_rate=learning_rate,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(record)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "loss": record["total_loss"],
                    "seconds": record["epoch_wall_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    cascade_blind = runner.predict_encoded(
        cascade_model,
        holdout,
        holdout_windows,
        batch_size=batch_size,
        device=device,
    )
    baseline_proposal, baseline_candidate = _decoder_candidate(
        runner, baseline_blind, holdout, base_config, shadow
    )
    cascade_proposal, cascade_candidate = _decoder_candidate(
        runner, cascade_blind, holdout, base_config, shadow
    )
    blind_path = artifact_dir / f"{phase}_blind.npz"
    _atomic_npz(
        blind_path,
        baseline_row_probability=baseline_blind.row_probability,
        baseline_boundary_probability=baseline_blind.boundary_probability,
        baseline_type_probability=baseline_blind.type_probability,
        baseline_proposal=baseline_proposal,
        baseline_candidate=baseline_candidate,
        cascade_row_probability=cascade_blind.row_probability,
        cascade_boundary_probability=cascade_blind.boundary_probability,
        cascade_type_probability=cascade_blind.type_probability,
        cascade_proposal=cascade_proposal,
        cascade_candidate=cascade_candidate,
    )
    history_path = artifact_dir / f"{phase}_history.json"
    _atomic_json(history_path, history)
    state_path = artifact_dir / f"{phase}_cascade_state.pt"
    torch.save(
        {
            "schema_version": "p1.mstcn_type_boundary_cascade.state.v1",
            "phase": phase,
            "seed": int(shadow["fixed_seed"]),
            "baseline_epoch": int(shadow["baseline_epoch"]),
            "fine_tune_epochs": int(fine_tune["epochs"]),
            "state_dict": {k: v.detach().cpu() for k, v in cascade_model.state_dict().items()},
        },
        state_path,
    )
    receipt = {
        "schema_version": "p1.mstcn_type_boundary_cascade.blind.v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "fold": base_config["phase_protocols"][phase]["fold"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": _sha256(CONFIG_PATH),
        "model_sha256": _sha256(ROOT / "src" / "p1_qc" / "ms_tcn_type_boundary_cascade.py"),
        "base_runner_sha256": _sha256(BASE_RUNNER_PATH),
        "ordered_holdout_key_sha256": runner._ordered_key_sha(holdout.surface.keys),  # noqa: SLF001
        "blind_path": blind_path.name,
        "blind_bytes": blind_path.stat().st_size,
        "blind_sha256": _sha256(blind_path),
        "history_sha256": _sha256(history_path),
        "state_sha256": _sha256(state_path),
        "split_receipt": split_receipt,
        "checkpoint": checkpoint_receipt,
        "warm_start": warm_start,
        "window_selection": window_selection_receipt,
        "positive_weight": positive_weight,
        "zero_step": zero_step,
        "baseline_positive_rows": int(np.sum(baseline_candidate)),
        "cascade_positive_rows": int(np.sum(cascade_candidate)),
        "holdout_truth_columns_opened_before_receipt": 0,
        "official_test_rows_read": 0,
        "submission_created": False,
        "upload_performed": False,
    }
    receipt_path = artifact_dir / f"{phase}_blind_receipt.json"
    _atomic_json(receipt_path, receipt)
    return {
        "phase": phase,
        "holdout": holdout,
        "receipt_path": receipt_path,
        "blind_path": blind_path,
    }


def _verify_blind(run: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads(run["receipt_path"].read_text(encoding="utf-8"))
    if receipt["experiment_id"] != EXPERIMENT_ID:
        raise ShadowError("blind receipt identity changed")
    if receipt["blind_sha256"] != _sha256(run["blind_path"]):
        raise ShadowError("blind payload changed after sealing")
    if receipt["holdout_truth_columns_opened_before_receipt"] != 0:
        raise ShadowError("blind prediction was not truth-blind")
    return receipt


def _read_fold_truth(runner: Any, base_config: dict[str, Any], holdout: Any, fold: str) -> Any:
    import pyarrow.dataset as dataset

    with runner._verified_immutable_read(  # noqa: SLF001
        base_config, "frozen_truth_and_folds", root=ROOT
    ) as oof_path:
        truth = (
            dataset.dataset(oof_path, format="parquet")
            .scanner(
                columns=[*runner.KEY_COLUMNS, "label", "anomaly_type", "fold"],
                filter=dataset.field("fold") == fold,
                use_threads=True,
            )
            .to_table()
            .to_pandas()
            .reset_index(drop=True)
        )
    truth, _receipt = runner._validate_registered_holdout_membership(  # noqa: SLF001
        truth, base_config, fold=fold
    )
    if not runner._keys_equal(holdout.surface.keys, truth):  # noqa: SLF001
        raise ShadowError("truth keys differ from sealed holdout")
    return truth


def _evaluate_phase(runner: Any, run: dict[str, Any], truth: Any) -> dict[str, Any]:
    import numpy as np

    with np.load(run["blind_path"], allow_pickle=False) as archive:
        baseline = archive["baseline_candidate"].astype(np.int8)
        cascade = archive["cascade_candidate"].astype(np.int8)
        baseline_proposal = archive["baseline_proposal"].astype(np.int8)
        cascade_proposal = archive["cascade_proposal"].astype(np.int8)
    labels = truth["label"].to_numpy(dtype=np.int8)
    baseline_metrics = runner.binary_metrics(labels, baseline)
    cascade_metrics = runner.binary_metrics(labels, cascade)
    added = (cascade == 1) & (baseline == 0)
    removed = (cascade == 0) & (baseline == 1)
    by_station: dict[str, Any] = {}
    for station in sorted(truth["station"].astype(str).unique()):
        mask = truth["station"].astype(str).to_numpy() == station
        left = runner.binary_metrics(labels[mask], baseline[mask])
        right = runner.binary_metrics(labels[mask], cascade[mask])
        by_station[station] = {
            "baseline": left,
            "cascade": right,
            "delta_f1": float(right["f1"] - left["f1"]),
            "candidate_added_rows": int(added[mask].sum()),
            "candidate_added_true_rows": int(labels[mask & added].sum()),
            "candidate_removed_rows": int(removed[mask].sum()),
            "candidate_removed_true_rows": int(labels[mask & removed].sum()),
        }
    return {
        "baseline": baseline_metrics,
        "cascade": cascade_metrics,
        "delta_f1": float(cascade_metrics["f1"] - baseline_metrics["f1"]),
        "baseline_proposal_rows": int(baseline_proposal.sum()),
        "cascade_proposal_rows": int(cascade_proposal.sum()),
        "candidate_added_rows": int(added.sum()),
        "candidate_added_true_rows": int(labels[added].sum()),
        "candidate_added_precision": float(labels[added].mean()) if bool(added.any()) else None,
        "candidate_removed_rows": int(removed.sum()),
        "candidate_removed_true_rows": int(labels[removed].sum()),
        "by_station": by_station,
    }


def _resume_q3_run(
    *, runner: Any, surfaces: Any, base_config: dict[str, Any], shadow: dict[str, Any]
) -> dict[str, Any]:
    receipt_path = ARTIFACT_DIR / "q3_blind_receipt.json"
    blind_path = ARTIFACT_DIR / "q3_blind.npz"
    preregistration_path = ARTIFACT_DIR / "preregistration.json"
    if not all(path.is_file() for path in (receipt_path, blind_path, preregistration_path)):
        raise ShadowError("Q3 recovery requires its complete sealed blind namespace")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    original = json.loads(preregistration_path.read_text(encoding="utf-8"))
    corrected = copy.deepcopy(original)
    corrected["warm_start_checkpoints"]["q4"]["sha256"] = shadow[
        "warm_start_checkpoints"
    ]["q4"]["sha256"]
    if corrected != shadow:
        raise ShadowError("recovery changed more than the verified Q4 SHA pin typo")
    if receipt.get("holdout_truth_columns_opened_before_receipt") != 0:
        raise ShadowError("Q3 receipt does not attest a truth-blind prediction")
    if receipt.get("blind_sha256") != _sha256(blind_path):
        raise ShadowError("sealed Q3 payload changed before recovery")
    _encoder, _training, holdout, _split = runner._prepare_phase_surfaces(  # noqa: SLF001
        surfaces, base_config, "q3", root=ROOT
    )
    amendment = {
        "schema_version": "p1.mstcn_type_boundary_cascade.amendment.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reason": "Q4 checkpoint SHA-256 transcription omitted one character; verified file and semantic checkpoint metadata were unchanged.",
        "changed_field": "warm_start_checkpoints.q4.sha256",
        "old_value": original["warm_start_checkpoints"]["q4"]["sha256"],
        "new_value": shadow["warm_start_checkpoints"]["q4"]["sha256"],
        "q3_retrained": False,
        "q3_truth_reads": 0,
        "scientific_hyperparameters_changed": False,
    }
    _atomic_json(ARTIFACT_DIR / "preregistration_amendment.json", amendment)
    return {
        "phase": "q3",
        "holdout": holdout,
        "receipt_path": receipt_path,
        "blind_path": blind_path,
    }


def execute(
    *,
    resume_q4_after_pin_correction: bool = False,
    candidate_module_name: str = "p1_qc.ms_tcn_type_boundary_cascade",
    window_selector: Any | None = None,
) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() and not resume_q4_after_pin_correction:
        raise FileExistsError(f"one-shot artifact namespace already exists: {ARTIFACT_DIR}")
    if not ARTIFACT_DIR.exists() and resume_q4_after_pin_correction:
        raise FileNotFoundError("Q4 recovery requires the sealed Q3 artifact namespace")
    shadow = _read_config()
    runner = _load_runner()
    base_config = runner._canonical_config()  # noqa: SLF001
    runner.verify_immutable_inputs(base_config, root=ROOT)
    np, pd, torch, baseline_api, data_api = runner._load_scientific()  # noqa: SLF001
    cascade_api = importlib.import_module(candidate_module_name)

    runner._load_scientific = lambda: (np, pd, torch, cascade_api, data_api)  # noqa: SLF001
    if not torch.cuda.is_available():
        raise ShadowError("CUDA is required for the fixed shadow screen")
    device = torch.device("cuda")
    if not resume_q4_after_pin_correction:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
        _atomic_json(ARTIFACT_DIR / "preregistration.json", shadow)
    surfaces = runner.load_blind_surfaces(base_config, root=ROOT)
    runs: list[dict[str, Any]] = []
    phases = list(shadow["phases"])
    if resume_q4_after_pin_correction:
        runs.append(
            _resume_q3_run(
                runner=runner,
                surfaces=surfaces,
                base_config=base_config,
                shadow=shadow,
            )
        )
        phases = ["q4"]
    for phase in phases:
        runs.append(
            _phase_fit(
                phase=phase,
                runner=runner,
                baseline_api=baseline_api,
                cascade_api=cascade_api,
                torch=torch,
                surfaces=surfaces,
                base_config=base_config,
                shadow=shadow,
                device=device,
                artifact_dir=ARTIFACT_DIR,
                window_selector=window_selector,
            )
        )

    receipts = [_verify_blind(run) for run in runs]
    if len(receipts) != 2:
        raise ShadowError("both Q3 and Q4 must be sealed before truth access")
    metrics: dict[str, Any] = {}
    pooled_labels: list[Any] = []
    pooled_baseline: list[Any] = []
    pooled_cascade: list[Any] = []
    for run, receipt in zip(runs, receipts, strict=True):
        truth = _read_fold_truth(
            runner,
            base_config,
            run["holdout"],
            receipt["fold"],
        )
        metrics[run["phase"]] = _evaluate_phase(runner, run, truth)
        with np.load(run["blind_path"], allow_pickle=False) as archive:
            pooled_baseline.append(archive["baseline_candidate"].astype(np.int8))
            pooled_cascade.append(archive["cascade_candidate"].astype(np.int8))
        pooled_labels.append(truth["label"].to_numpy(dtype=np.int8))
    labels = np.concatenate(pooled_labels)
    baseline = np.concatenate(pooled_baseline)
    cascade = np.concatenate(pooled_cascade)
    pooled_baseline_metrics = runner.binary_metrics(labels, baseline)
    pooled_cascade_metrics = runner.binary_metrics(labels, cascade)
    pooled_delta = float(pooled_cascade_metrics["f1"] - pooled_baseline_metrics["f1"])
    phase_floor = float(shadow["evaluation"]["performance_gate"]["each_phase_delta_f1_min"])
    performance_checks = {
        "pooled_delta_f1": pooled_delta
        >= float(shadow["evaluation"]["performance_gate"]["pooled_delta_f1_min"]),
        "each_phase_delta_f1": all(value["delta_f1"] >= phase_floor for value in metrics.values()),
        "anchor_positive_removed_rows": True,
    }
    max_zero = max(
        value
        for receipt in receipts
        for value in receipt["zero_step"].values()
    )
    information_checks = {
        "zero_step_equivalence": max_zero
        <= float(shadow["evaluation"]["information_gate"]["zero_step_max_abs_difference"]),
        "new_pathways_changed_predictions": any(
            value["candidate_added_rows"] + value["candidate_removed_rows"] > 0
            for value in metrics.values()
        ),
        "nonfinite_count": all(
            math.isfinite(value["cascade"]["f1"]) for value in metrics.values()
        ),
    }
    status = (
        "GO_FULL_ENSEMBLE_REPLICATION_NOT_SUBMISSION_AUTHORIZED"
        if all(performance_checks.values())
        else "NO_GO_SINGLE_SEED_STRUCTURAL_SCREEN"
    )
    result = {
        "schema_version": "p1.mstcn_type_boundary_cascade.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "metrics": metrics,
        "pooled": {
            "baseline": pooled_baseline_metrics,
            "cascade": pooled_cascade_metrics,
            "delta_f1": pooled_delta,
        },
        "performance_checks": performance_checks,
        "information_checks": information_checks,
        "truth_reads": {"q3": 1, "q4": 1, "before_both_blind_seals": 0},
        "official_test_rows_read": 0,
        "submission_created": False,
        "upload_performed": False,
        "claim_limit": "Exposed historical Q3/Q4 single-seed shadow screen; no official transport or score-gain claim.",
    }
    _atomic_json(ARTIFACT_DIR / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-q4-after-pin-correction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "CHECK_ONLY",
                    "config_sha256": _sha256(CONFIG_PATH),
                    "official_test_reads": 0,
                    "submission_created": False,
                    "upload_performed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    execute(resume_q4_after_pin_correction=args.resume_q4_after_pin_correction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
