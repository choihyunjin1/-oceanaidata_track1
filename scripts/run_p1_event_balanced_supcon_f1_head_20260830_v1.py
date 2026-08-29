"""One-shot low-fidelity P1 real-event SupCon plus F1 top-k screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p1_qc.event_balanced_supcon_f1 import (  # noqa: E402
    apply_cell_topk,
    build_real_events,
    calibrate_cell_topk_rates,
    event_balanced_windows,
    pool_shared_hidden,
    soft_f1_loss,
    supervised_contrastive_loss,
)
from p1_qc.mstcn_group_dro import changed_row_concentration  # noqa: E402

EXPERIMENT_ID = "p1_event_balanced_supcon_f1_head_20260830_v1"
CONFIG_PATH = ROOT / "configs/experiments/p1_event_balanced_supcon_f1_head_20260830_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SOBOL_PATH = ROOT / "scripts/run_p1_mstcn_sobol_hpo_20260829_v1.py"

SPEC = importlib.util.spec_from_file_location("p1_mstcn_sobol_frozen_for_supcon", SOBOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen Sobol runner")
SOBOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOBOL
SPEC.loader.exec_module(SOBOL)


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment id changed")
    if config.get("status") != "PREREGISTERED_ROOT_AUTHORIZED_ONE_SHOT_LOW_FIDELITY":
        raise ContractError("one-shot authorization changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ContractError("every prohibition must remain active")
    training = config["training_contract"]
    epochs = int(training["epochs"])
    if not int(training["permitted_epochs_min"]) <= epochs <= int(
        training["permitted_epochs_max"]
    ):
        raise ContractError("low-fidelity epoch budget changed")
    if int(training["maximum_lifetime_historical_fit_count"]) != 3:
        raise ContractError("fit budget changed")
    if training["phase_order"] != ["q2", "q3", "q4"]:
        raise ContractError("phase order changed")
    if config["proposal_head"]["global_threshold"] is not None:
        raise ContractError("global threshold is prohibited")
    if config["event_balance_contract"]["synthetic_anomaly_generation"]:
        raise ContractError("synthetic anomalies are prohibited")
    return config


def _verify_pins(config: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, record in config["source_pins"].items():
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ContractError(f"source pin changed: {name}")
        observed[name] = {"path": record["path"], "sha256": record["sha256"]}
    prereg = json.loads(
        (ROOT / config["source_pins"]["deep_research_preregistration"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    record = next(
        item for item in prereg["experiments"] if item["id"] == EXPERIMENT_ID
    )
    if record["initial_seeds"] != 1 or not record[
        "three_seed_confirmation_requires_separate_authorization"
    ]:
        raise ContractError("Deep Research P1 preregistration changed")
    preconfirm = json.loads(
        (ROOT / config["source_pins"]["sobol_preconfirm"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    selected = preconfirm["selected_recipe"]["trial"]
    for key, value in config["selected_source_trial"].items():
        if selected.get(key) != value:
            raise ContractError(f"selected Sobol trial changed: {key}")
    return observed


def _candidate_config(base: Any, source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidate = SOBOL._trial_config(
        base,
        source,
        config["selected_source_trial"],
        int(config["training_contract"]["seed"]),
    )
    candidate["training"]["weight_decay"] = float(
        config["training_contract"]["strong_weight_decay"]
    )
    if int(candidate["training"]["maximum_epochs"]) != int(
        config["training_contract"]["source_schedule_horizon_epochs"]
    ):
        raise ContractError("source LR horizon changed")
    return candidate


def _event_support_checks(config: dict[str, Any], receipt: dict[str, Any]) -> dict[str, bool]:
    contract = config["event_balance_contract"]
    return {
        "real_events_present": int(receipt["real_event_count"]) > 0,
        "typed_event_support": int(receipt["supported_type_count"])
        >= int(contract["minimum_supported_type_count"]),
        "station_layer_season_support": int(receipt["supported_station_layer_season_cells"])
        >= int(contract["minimum_supported_station_layer_season_cells"]),
        "event_cell_concentration": float(receipt["maximum_event_cell_share"])
        <= float(contract["maximum_event_cell_share"]),
        "synthetic_event_count_eq_0": int(receipt["synthetic_event_count"]) == 0,
    }


def _synthetic_objective_smoke(
    base: Any, source: dict[str, Any], config: dict[str, Any], torch: Any
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = _candidate_config(base, source, config)
    model = base._new_model(165, candidate, device)
    projection = torch.nn.Sequential(
        torch.nn.Linear(int(config["selected_source_trial"]["width"]), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
    ).to(device)
    values = torch.randn((6, 64, 165), generator=torch.Generator(device=device).manual_seed(7), device=device)
    valid = torch.ones((6, 64), dtype=torch.bool, device=device)
    event = torch.zeros((6, 64), dtype=torch.float32, device=device)
    event[0:4, 12:20] = 1.0
    kinds = torch.zeros((6, 64, 5), dtype=torch.float32, device=device)
    kinds[0:2, 12:20, 0] = 1.0
    kinds[2:4, 12:20, 1] = 1.0
    classes = torch.tensor([0, 0, 1, 1, 5, 5], dtype=torch.long, device=device)
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = output[1]

    handle = model.prediction_generator.register_forward_hook(hook)
    output = model(values, valid_mask=valid)
    handle.remove()
    pooled = pool_shared_hidden(captured["hidden"], event, kinds, valid, classes)
    objective = supervised_contrastive_loss(
        projection(pooled), classes, temperature=float(config["training_contract"]["supcon_temperature"])
    ) + soft_f1_loss(output.final_logits, event, valid)
    if not bool(torch.isfinite(objective)):
        raise ContractError("synthetic objective is non-finite")
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in (*model.parameters(), *projection.parameters())
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise ContractError("synthetic objective gradients are invalid")
    result = {
        "device": str(device),
        "shape": [6, 64, 165],
        "loss_finite": True,
        "gradient_finite": True,
        "backbone_parameter_count": int(model.trainable_parameter_count),
        "projection_parameter_count": sum(parameter.numel() for parameter in projection.parameters()),
        "historical_fit_count": 0,
    }
    del model, projection, output, objective
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def check_only() -> dict[str, Any]:
    config = _config()
    pins = _verify_pins(config)
    base = SOBOL._load_base(root=ROOT)
    source = base._canonical_config()
    immutable = base.verify_immutable_inputs(source, root=ROOT)
    _np, _pd, torch, _model, _data = base._load_scientific()
    SOBOL._configure_torch_threads(torch, {"training_contract": config["training_contract"]})
    smoke = _synthetic_objective_smoke(base, source, config, torch)
    surfaces = base.load_blind_surfaces(source, root=ROOT)
    supports: dict[str, Any] = {}
    split_receipts: dict[str, Any] = {}
    all_checks: dict[str, bool] = {}
    candidate = _candidate_config(base, source, config)
    for phase in config["training_contract"]["phase_order"]:
        _encoder, training, holdout, split = base._prepare_phase_surfaces(
            surfaces, source, phase, root=ROOT
        )
        _events, support = build_real_events(training)
        checks = _event_support_checks(config, support)
        checks.update(
            {
                "chronological_feature_gap_positive": float(split["feature_non_overlap_slack_hours"]) > 0.0,
                "split_before_windowing": bool(split["split_before_windowing"]),
                "cross_split_window_count_eq_0": int(split["cross_split_window_count"]) == 0,
                "holdout_truth_columns_read_eq_0": int(split["holdout_truth_columns_read"]) == 0,
                "holdout_anchor_present": holdout.surface.anchor is not None,
            }
        )
        # Materialize only metadata/sampling receipts; this performs no fit and
        # does not open the blind fold truth.
        _windows, _class_ids, _is_event, balance = event_balanced_windows(
            training,
            window_size=int(candidate["windowing"]["rows"]),
            stride=int(candidate["windowing"]["stride_rows"]),
            seed=int(config["training_contract"]["seed"]),
        )
        supports[phase] = {"support": support, "balance": balance, "checks": checks}
        split_receipts[phase] = split
        all_checks.update({f"{phase}_{name}": value for name, value in checks.items()})
    anchor = base.anchor_preserving_union(
        [1, 0, 1, 0], [0, 1, 0, 0]
    )
    all_checks["anchor_decoder_immutable"] = list(anchor) == [1, 1, 1, 0]
    if not all(all_checks.values()):
        raise ContractError("P1 event-support or split preflight failed")
    return {
        "schema_version": "p1.event_balanced_supcon_f1_head.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "helper_sha256": _sha256(ROOT / "src/p1_qc/event_balanced_supcon_f1.py"),
        "source_pins": pins,
        "immutable_inputs": immutable,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "synthetic_objective_smoke": smoke,
        "phase_event_support": supports,
        "phase_split_receipts": split_receipts,
        "checks": all_checks,
        "artifact_namespace_available": not ARTIFACT_DIR.exists() and not ATTEMPT_LOCK.exists(),
        "historical_fit_count": 0,
        "holdout_truth_rows_read": 0,
        "official_interface_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def _index_batches(count: int, batch_size: int, *, seed: int) -> list[list[int]]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def _fit_phase(
    base: Any,
    training: Any,
    holdout: Any,
    *,
    source: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    device: Any,
    run_started: float,
) -> tuple[Any, Any, dict[str, Any]]:
    np, _pd, torch, model_api, data_api = base._load_scientific()
    seed = int(config["training_contract"]["seed"])
    candidate = _candidate_config(base, source, config)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base._reset_cuda_peak_memory(torch, device)
    model = base._new_model(training.features.shape[1], candidate, device)
    expected = int(
        source["architecture"]["exact_parameter_count_by_width_at_input_165"][
            str(config["selected_source_trial"]["width"])
        ]
    )
    if int(model.trainable_parameter_count) != expected:
        raise ContractError("backbone parameter count changed")
    projection = torch.nn.Sequential(
        torch.nn.Linear(
            int(config["selected_source_trial"]["width"]),
            int(config["training_contract"]["projection_hidden"]),
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            int(config["training_contract"]["projection_hidden"]),
            int(config["training_contract"]["projection_output"]),
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *projection.parameters()],
        lr=float(candidate["training"]["learning_rate"]),
        weight_decay=float(candidate["training"]["weight_decay"]),
    )
    windows, class_ids, _is_event, balance_receipt = event_balanced_windows(
        training,
        window_size=int(candidate["windowing"]["rows"]),
        stride=int(candidate["windowing"]["stride_rows"]),
        seed=seed,
    )
    support_checks = _event_support_checks(config, balance_receipt)
    if not all(support_checks.values()):
        raise ContractError(f"{phase} event support changed after lock")
    positive_weight = base._positive_weight(training.surface.labels)
    _steps, total_steps, warmup_steps = base._schedule_geometry(
        candidate, window_count=len(windows)
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    history_path = ARTIFACT_DIR / "histories" / f"{phase}.json"
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = output[1]

    handle = model.prediction_generator.register_forward_hook(hook)
    try:
        for epoch in range(1, int(config["training_contract"]["epochs"]) + 1):
            epoch_started = time.perf_counter()
            batches = _index_batches(
                len(windows), int(candidate["training"]["batch_size"]), seed=seed + epoch
            )
            optimizer.zero_grad(set_to_none=True)
            sums = {"total": 0.0, "base": 0.0, "supcon": 0.0, "soft_f1": 0.0}
            observed = 0
            grad_norms: list[float] = []
            steps_before = global_step
            model.train()
            projection.train()
            for batch_index, indices in enumerate(batches):
                batch_windows = tuple(windows[index] for index in indices)
                values, feature_valid = data_api.materialize_windows(
                    training.features, batch_windows
                )
                targets = base._materialize_target_batch(training.targets, batch_windows)
                tensor = torch.from_numpy(values).to(device, non_blocking=True)
                event, boundary, kinds, valid = base._move_targets(targets, device)
                feature_valid_tensor = torch.from_numpy(feature_valid).to(
                    device, dtype=torch.bool, non_blocking=True
                )
                classes = torch.from_numpy(class_ids[np.asarray(indices)]).to(
                    device, dtype=torch.long, non_blocking=True
                )
                if not bool(torch.equal(feature_valid_tensor, valid)):
                    raise ContractError("feature and target valid masks differ")
                captured.clear()
                with base._autocast(device):
                    output = model(tensor, valid_mask=feature_valid_tensor)
                    base_loss = model_api.compute_ms_tcn_asrf_loss(
                        output,
                        event,
                        boundary,
                        kinds,
                        valid_mask=valid,
                        config=base._loss_config(candidate, positive_weight),
                    )
                    pooled = pool_shared_hidden(
                        captured["hidden"], event, kinds, valid, classes
                    )
                    supcon = supervised_contrastive_loss(
                        projection(pooled),
                        classes,
                        temperature=float(config["training_contract"]["supcon_temperature"]),
                    )
                    f1_loss = soft_f1_loss(output.final_logits, event, valid)
                    total = (
                        base_loss.total
                        + float(config["training_contract"]["supcon_weight"]) * supcon
                        + float(config["training_contract"]["soft_f1_weight"]) * f1_loss
                    )
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("non-finite event-balanced objective")
                total.backward()
                weight = len(indices)
                for name, value in (
                    ("total", total),
                    ("base", base_loss.total),
                    ("supcon", supcon),
                    ("soft_f1", f1_loss),
                ):
                    scalar = float(value.detach().float().cpu())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"non-finite {name} loss")
                    sums[name] += scalar * weight
                observed += weight
                lr = base._lr_at_step(
                    global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    maximum_lr=float(candidate["training"]["learning_rate"]),
                    minimum_lr=3e-6,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                norm = torch.nn.utils.clip_grad_norm_(
                    [*model.parameters(), *projection.parameters()],
                    float(candidate["training"]["gradient_clip_norm"]),
                )
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("non-finite gradient norm")
                grad_norms.append(float(norm.detach().float().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                _ = batch_index
            history.append(
                {
                    "epoch": epoch,
                    **{name: value / max(1, observed) for name, value in sums.items()},
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_steps_epoch": global_step - steps_before,
                    "optimizer_steps_cumulative": global_step,
                    "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                    "grad_norm_max": max(grad_norms),
                    "epoch_wall_seconds": time.perf_counter() - epoch_started,
                    "performance_metrics_exposed": False,
                }
            )
            if epoch == 1 or epoch % 5 == 0 or epoch == int(config["training_contract"]["epochs"]):
                base._atomic_json(history_path, history, replace=True)
                base._atomic_json(
                    ARTIFACT_DIR / "progress.json",
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "phase": phase,
                        "epoch": epoch,
                        "epochs": int(config["training_contract"]["epochs"]),
                        "elapsed_seconds": time.perf_counter() - run_started,
                        "historical_fit_count_completed_before_current": ["q2", "q3", "q4"].index(phase),
                        "performance_metrics_exposed": False,
                    },
                    replace=True,
                )
    finally:
        handle.remove()
    training_bundle = base.predict_encoded(
        model,
        training,
        base._all_windows(training, candidate),
        batch_size=int(candidate["training"]["batch_size"]),
        device=device,
    )
    holdout_bundle = base.predict_encoded(
        model,
        holdout,
        base._all_windows(holdout, candidate),
        batch_size=int(candidate["training"]["batch_size"]),
        device=device,
    )
    receipt = {
        "phase": phase,
        "seed": seed,
        "epochs": int(config["training_contract"]["epochs"]),
        "optimizer_steps": global_step,
        "backbone_parameter_count": int(model.trainable_parameter_count),
        "projection_parameter_count": sum(parameter.numel() for parameter in projection.parameters()),
        "event_balance": balance_receipt,
        "event_support_checks": support_checks,
        "history_artifact": {
            "path": history_path.relative_to(ARTIFACT_DIR).as_posix(),
            "bytes": history_path.stat().st_size,
            "sha256": _sha256(history_path),
        },
        "checkpoint_persisted": False,
        "holdout_truth_rows_read": 0,
        **base._cuda_peak_memory_receipt(torch, device),
    }
    del model, projection, optimizer
    torch.cuda.empty_cache()
    return training_bundle, holdout_bundle, receipt


def _type_metrics(base: Any, truth: Any, prediction: Any) -> dict[str, Any]:
    np, _pd, _torch, _model, _data = base._load_scientific()
    label = truth["label"].to_numpy(dtype=np.int8)
    raw_type = truth["anomaly_type"].fillna("").astype(str).str.lower().to_numpy()
    by_type: dict[str, Any] = {}
    for name in base.TYPE_NAMES:
        present = np.asarray(
            [name in {token.strip() for token in value.split("+") if token.strip()} for value in raw_type],
            dtype=bool,
        )
        universe = (label == 0) | present
        metrics = base.binary_metrics(present[universe].astype(np.int8), np.asarray(prediction)[universe])
        metrics["evaluation_rows"] = int(universe.sum())
        metrics["positive_rows"] = int(present.sum())
        by_type[name] = metrics
    return {
        "definition": "normal_rows_plus_current_type_positive_rows",
        "by_type": by_type,
        "macro_f1": sum(float(value["f1"]) for value in by_type.values()) / len(by_type),
    }


def _evaluate_fold(
    base: Any,
    truth: Any,
    holdout: Any,
    candidate: Any,
    proposal: Any,
    control: Any,
) -> dict[str, Any]:
    np, _pd, _torch, _model, _data = base._load_scientific()
    label = truth["label"].to_numpy(dtype=np.int8)
    candidate_metrics = base.binary_metrics(label, candidate)
    control_metrics = base.binary_metrics(label, control)
    type_candidate = _type_metrics(base, truth, candidate)
    type_control = _type_metrics(base, truth, control)
    added = (np.asarray(proposal, dtype=np.int8) == 1) & (
        np.asarray(holdout.surface.anchor, dtype=np.int8) == 0
    )
    proposal_metrics = base.binary_metrics(label, added.astype(np.int8))
    removed = int(
        np.sum(
            (np.asarray(holdout.surface.anchor, dtype=np.int8) == 1)
            & (np.asarray(candidate, dtype=np.int8) == 0)
        )
    )
    return {
        "candidate": candidate_metrics,
        "control": control_metrics,
        "delta_f1": float(candidate_metrics["f1"] - control_metrics["f1"]),
        "candidate_type": type_candidate,
        "control_type": type_control,
        "type_macro_delta_f1": float(type_candidate["macro_f1"] - type_control["macro_f1"]),
        "proposal": proposal_metrics,
        "proposal_precision_floor": 0.5 * float(control_metrics["f1"]),
        "proposal_precision_gate": float(proposal_metrics["precision"])
        > 0.5 * float(control_metrics["f1"]),
        "anchor_positive_removed_rows": removed,
        "changed_row_concentration": changed_row_concentration(
            holdout.surface.keys["station"], candidate, control
        ),
    }


def execute(*, expected_runner_sha256: str) -> dict[str, Any]:
    if expected_runner_sha256.casefold() != _sha256(Path(__file__)):
        raise ContractError("expected runner hash differs")
    if ARTIFACT_DIR.exists() or ATTEMPT_LOCK.exists():
        raise FileExistsError("one-shot namespace already exists")
    preflight = check_only()
    if not preflight["cuda_available"]:
        raise ContractError("P1 low-fidelity screen requires CUDA")
    config = _config()
    base = SOBOL._load_base(root=ROOT)
    source = base._canonical_config()
    np, _pd, torch, _model, _data = base._load_scientific()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": preflight["runner_sha256"],
        "config_sha256": preflight["config_sha256"],
        "one_shot": True,
        "maximum_historical_fits": 3,
        "three_seed_confirmation_authorized": False,
    }
    base._exclusive_json(ATTEMPT_LOCK, lock)
    ARTIFACT_DIR.mkdir(parents=False, exist_ok=False)
    started = datetime.now(UTC)
    run_started = time.perf_counter()
    terminal = ARTIFACT_DIR / "terminal_result.json"
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": preflight["config_sha256"],
        "runner_sha256": preflight["runner_sha256"],
        "helper_sha256": preflight["helper_sha256"],
        "phase_order": config["training_contract"]["phase_order"],
        "fit_budget": 3,
        "seed": config["training_contract"]["seed"],
        "epochs": config["training_contract"]["epochs"],
        "loss_weights": {
            "supcon": config["training_contract"]["supcon_weight"],
            "soft_f1": config["training_contract"]["soft_f1_weight"],
        },
        "proposal_head": config["proposal_head"],
        "screen_gate": config["screen_gate"],
        "three_seed_confirmation_authorized": False,
    }
    commitment_sha = hashlib.sha256(
        json.dumps(commitment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        base._atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
        base._atomic_json(ARTIFACT_DIR / "commitment.json", commitment)
        surfaces = base.load_blind_surfaces(source, root=ROOT)
        sobol_config = json.loads(
            (ROOT / config["source_pins"]["sobol_config"]["path"]).read_text(encoding="utf-8")
        )
        controls = SOBOL._control_candidates(base, sobol_config, surfaces, root=ROOT)
        device = torch.device("cuda")
        receipt_paths: dict[str, Path] = {}
        holdouts: dict[str, Any] = {}
        calibration_receipts: dict[str, Any] = {}
        fit_count = 0
        for phase in config["training_contract"]["phase_order"]:
            encoder, training, holdout, split = base._prepare_phase_surfaces(
                surfaces, source, phase, root=ROOT
            )
            base._atomic_json(ARTIFACT_DIR / f"{phase}_split.json", split)
            base._atomic_json(
                ARTIFACT_DIR / f"{phase}_encoder.json", base._encoder_receipt(encoder)
            )
            training_bundle, holdout_bundle, fit_receipt = _fit_phase(
                base,
                training,
                holdout,
                source=source,
                config=config,
                phase=phase,
                device=device,
                run_started=run_started,
            )
            fit_count += 1
            proposal_config = config["proposal_head"]
            cell_rates, fallback_rate, calibration = calibrate_cell_topk_rates(
                training.surface.keys,
                training_bundle.row_probability,
                training.surface.labels,
                minimum_rows=int(proposal_config["minimum_training_rows_per_cell"]),
                minimum_positives=int(
                    proposal_config["minimum_training_positive_rows_per_cell"]
                ),
                maximum_rate=float(proposal_config["maximum_topk_rate"]),
            )
            proposal, application = apply_cell_topk(
                holdout.surface.keys,
                holdout_bundle.row_probability,
                holdout.surface.anchor,
                cell_rates=cell_rates,
                fallback_rate=fallback_rate,
            )
            candidate = base.anchor_preserving_union(
                holdout.surface.anchor, proposal
            ).astype(np.int8, copy=False)
            calibration_receipts[phase] = {
                "calibration": calibration,
                "application": application,
                "eligible_cell_rates": {
                    "|".join(key): float(value) for key, value in sorted(cell_rates.items())
                },
                "global_fallback_rate": float(fallback_rate),
            }
            base._atomic_json(
                ARTIFACT_DIR / f"{phase}_topk_receipt.json", calibration_receipts[phase]
            )
            fold = {"q2": "2025_q2", "q3": "2025_q3", "q4": "2025_q4"}[phase]
            receipt_paths[phase] = SOBOL._seal_npz(
                base,
                ARTIFACT_DIR,
                name=f"{phase}_event_balanced_supcon_f1_blind",
                arrays={
                    "row_probability": holdout_bundle.row_probability,
                    "boundary_probability": holdout_bundle.boundary_probability,
                    "type_probability": holdout_bundle.type_probability,
                    "proposal": proposal,
                    "candidate": candidate,
                },
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
                key_sha256=surfaces.membership_sha256[fold],
                fit_receipts=[fit_receipt],
            )
            holdouts[phase] = holdout
        if fit_count != 3 or len(receipt_paths) != 3:
            raise ContractError("all three low-fidelity predictions were not sealed")
        for path in receipt_paths.values():
            SOBOL._verify_receipt(
                path,
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
            )
        # First holdout truth read occurs only after all three blind artifacts
        # and receipts have been byte-verified.
        truths = {
            phase: SOBOL._load_truth(
                base,
                source,
                holdouts[phase].surface,
                [receipt_paths[phase]],
                fold={"q2": "2025_q2", "q3": "2025_q3", "q4": "2025_q4"}[phase],
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
                root=ROOT,
            )
            for phase in config["training_contract"]["phase_order"]
        }
        sealed = {
            phase: SOBOL._load_arrays(
                receipt_paths[phase],
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
            )
            for phase in config["training_contract"]["phase_order"]
        }
        fold_results = {
            phase: _evaluate_fold(
                base,
                truths[phase],
                holdouts[phase],
                sealed[phase]["candidate"],
                sealed[phase]["proposal"],
                controls[phase],
            )
            for phase in config["training_contract"]["phase_order"]
        }
        pooled_truth = np.concatenate(
            [truths[phase]["label"].to_numpy(dtype=np.int8) for phase in config["training_contract"]["phase_order"]]
        )
        pooled_candidate = np.concatenate(
            [sealed[phase]["candidate"] for phase in config["training_contract"]["phase_order"]]
        )
        pooled_control = np.concatenate(
            [controls[phase] for phase in config["training_contract"]["phase_order"]]
        )
        pooled_truth_frame = _pd.concat(
            [truths[phase] for phase in config["training_contract"]["phase_order"]],
            ignore_index=True,
        )
        pooled_type_candidate = _type_metrics(base, pooled_truth_frame, pooled_candidate)
        pooled_type_control = _type_metrics(base, pooled_truth_frame, pooled_control)
        pooled_concentration = changed_row_concentration(
            np.concatenate(
                [holdouts[phase].surface.keys["station"].astype(str).to_numpy() for phase in config["training_contract"]["phase_order"]]
            ),
            pooled_candidate,
            pooled_control,
        )
        pooled = {
            "candidate": base.binary_metrics(pooled_truth, pooled_candidate),
            "control": base.binary_metrics(pooled_truth, pooled_control),
            "candidate_type": pooled_type_candidate,
            "control_type": pooled_type_control,
            "type_macro_delta_f1": float(
                pooled_type_candidate["macro_f1"] - pooled_type_control["macro_f1"]
            ),
            "changed_row_concentration": pooled_concentration,
        }
        checks = {
            "all_windows_delta_f1_strictly_positive": all(
                float(fold_results[phase]["delta_f1"]) > 0.0
                for phase in config["training_contract"]["phase_order"]
            ),
            "pooled_anomaly_type_macro_delta_f1_strictly_positive": float(
                pooled["type_macro_delta_f1"]
            )
            > 0.0,
            "proposal_precision_gt_half_incumbent_f1_each_window": all(
                bool(fold_results[phase]["proposal_precision_gate"])
                for phase in config["training_contract"]["phase_order"]
            ),
            "anchor_positive_removed_rows_eq_0": sum(
                int(fold_results[phase]["anchor_positive_removed_rows"])
                for phase in config["training_contract"]["phase_order"]
            )
            == 0,
            "maximum_station_changed_row_share_lte_0_8": float(
                pooled_concentration["maximum_station_share"]
            )
            <= float(config["screen_gate"]["maximum_station_changed_row_share"]),
        }
        passed = all(checks.values())
        result = {
            "schema_version": "p1.event_balanced_supcon_f1_head.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "PASS_LOW_FIDELITY_RESEARCH_ONLY_CONFIRMATION_NOT_AUTHORIZED"
                if passed
                else "NO_GO_LOW_FIDELITY_SCREEN"
            ),
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - run_started,
            "historical_fit_count": fit_count,
            "folds": fold_results,
            "pooled": pooled,
            "checks": checks,
            "pass": passed,
            "calibration_receipts": calibration_receipts,
            "all_three_blind_predictions_sealed_before_truth_read": True,
            "holdout_truth_rows_read_before_all_seals": 0,
            "synthetic_event_count": 0,
            "checkpoint_persisted": False,
            "three_seed_confirmation_authorized": False,
            "three_seed_confirmation_executed": False,
            "result_based_rerun_authorized": False,
            "official_interface_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
        }
        base._atomic_json(ARTIFACT_DIR / "aggregate.json", result)
        base._atomic_json(terminal, result)
        return result
    except BaseException as error:
        if ARTIFACT_DIR.exists() and not terminal.exists():
            base._atomic_json(
                terminal,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_EXECUTION_NO_RETRY_AUTHORIZED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "historical_fit_count_upper_bound": 3,
                    "official_interface_rows_read": 0,
                    "csv_created": False,
                    "upload_performed": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check-only", "execute"), required=True)
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args()
    if args.mode == "check-only":
        result = check_only()
    else:
        if not args.expected_runner_sha256:
            raise SystemExit("--expected-runner-sha256 is required for execute")
        result = execute(expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
