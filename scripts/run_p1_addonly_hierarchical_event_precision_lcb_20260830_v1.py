"""Execute the sealed one-shot P1 add-only hierarchical precision experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc import (  # noqa: E402
    p1_addonly_hierarchical_event_precision_lcb_20260830_v1 as core,
)

EXPERIMENT_ID = "p1_addonly_hierarchical_event_precision_lcb_20260830_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
CONFIG_SHA256 = "989a5d42878ff0062b57c2d76425da8d654161b9fd76150fcd8d9eb01c0692a2"


class RegistrationError(RuntimeError):
    """Raised before fitting when the sealed preregistration no longer matches."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()


def _verify_registered_file(
    record: Mapping[str, Any], forbidden_tokens: Sequence[str]
) -> tuple[Path, dict[str, Any]]:
    relative = _normal_path(str(record["path"]))
    if any(token.lower() in relative for token in forbidden_tokens):
        raise RegistrationError("registered path contains a forbidden interface token")
    try:
        path = (ROOT / str(record["path"])).resolve(strict=True)
        resolved_root = ROOT.resolve(strict=True)
    except OSError as error:
        raise RegistrationError("registered file is missing or inaccessible") from error
    if path != resolved_root and resolved_root not in path.parents:
        raise RegistrationError("registered path escaped the repository root")
    observed = {"path": relative, "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
    if observed["bytes"] != int(record["bytes"]) or observed["sha256"] != str(
        record["sha256"]
    ):
        raise RegistrationError("registered file bytes or sha256 changed")
    return path, observed


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegistrationError("registered JSON is not an object")
    return value


def _load_source_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("sealed_p1_proposal_bank_source_loader", path)
    if spec is None or spec.loader is None:
        raise RegistrationError("sealed proposal-bank loader cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RegistrationError("experiment identity changed")
    if config.get("status") != "PREREGISTERED_ONE_SHOT_EXPOSED_OOF_RESEARCH_ONLY":
        raise RegistrationError("research-only status changed")
    prefixes = [
        (item["name"], tuple(item["fit_folds"]), item["blind_next_fold"])
        for item in config["chronological_prefixes"]
    ]
    if prefixes != [
        ("q2_to_q3", ("2025_q2",), "2025_q3"),
        ("q2_q3_to_q4", ("2025_q2", "2025_q3"), "2025_q4"),
    ]:
        raise RegistrationError("chronological prefix contract changed")
    head = config["head"]
    if (
        head["family"] != "aggregated_binomial_logistic_partial_pooling"
        or float(head["ridge_strength"]) != 4.0
        or int(head["maximum_newton_iterations"]) != 100
        or float(head["newton_tolerance"]) != 1e-10
        or float(head["precision_lcb_one_sided_confidence"]) != 0.95
        or float(head["precision_lcb_z"]) != 1.6448536269514722
        or int(head["hyperparameter_or_feature_search_count"]) != 0
    ):
        raise RegistrationError("fixed partial-pooling head contract changed")
    candidate = config["candidate_rule"]
    if (
        candidate["proposal_bank_rule"]
        != "anchor == 0 AND any frozen heterogeneous source == 1"
        or candidate["event_group"] != ["fold", "station", "year", "layer"]
        or int(candidate["contiguous_cadence_minutes"]) != 10
        or candidate["accept_event_if"]
        != "train_only_predicted_precision_one_sided_95pct_lcb > fit_prefix_anchor_f1 / 2"
        or int(candidate["threshold_search_count"]) != 0
        or candidate["candidate"] != "anchor OR accepted_proposal_event_rows"
        or candidate["anchor_positive_removal_allowed"] is not False
        or candidate["persist_row_predictions_or_keys"] is not False
    ):
        raise RegistrationError("add-only candidate contract changed")
    hierarchy = config["decision_hierarchy"]
    if (
        "directional margin = 0" not in hierarchy["level_1_sole_primary"]
        or "pooled row-level binary micro-F1" not in hierarchy["level_1_sole_primary"]
        or hierarchy["level_3_diagnostics_only"]
        != [
            "prefix F1",
            "accepted event and row support",
            "station",
            "layer",
            "KST month",
            "frozen source signature",
            "proposal precision",
        ]
        or set(hierarchy["forbidden_hard_gates"])
        != {
            "minimum event count",
            "minimum improved window count",
            "all-window improvement",
            "station concentration cap",
            "worst-slice veto",
            "post-result numeric delta",
        }
    ):
        raise RegistrationError("metric-aligned decision hierarchy changed")
    resampling = config["paired_resampling"]
    if (
        resampling["method"] != "circular_moving_block_bootstrap"
        or int(resampling["replicates"]) != 5000
        or int(resampling["seed"]) != 20260830
        or float(resampling["lower_quantile"]) != 0.05
        or float(resampling["upper_quantile"]) != 0.95
    ):
        raise RegistrationError("paired uncertainty contract changed")
    execution = config["execution_contract"]
    expected_execution = {
        "authorized_attempts": 1,
        "model_fit_count": 2,
        "threshold_search_count": 0,
        "retry_or_tuning_count": 0,
        "official_interface_rows_read": 0,
        "raw_training_rows_read": 0,
        "raw_temp_rows_read": 0,
        "auxiliary_psal_depth_rows_read": 0,
        "prediction_csv_count": 0,
        "upload_count": 0,
    }
    if execution != expected_execution:
        raise RegistrationError("one-shot execution contract changed")
    outlier = config["outlier_policy"]
    if (
        outlier["remove_label_1_rows_or_events"] is not False
        or outlier["clip_or_remove_raw_temp_anomaly_signal"] is not False
        or outlier["hard_delete_any_input_row"] is not False
        or int(outlier["target_positive_rows_removed"]) != 0
    ):
        raise RegistrationError("outlier or target-anomaly protection changed")
    outputs = config["outputs"]
    if (
        outputs["attempt_lock"]
        != "reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/attempt_lock.json"
        or outputs["result"]
        != "reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/result.json"
        or outputs["exclusive_create"] is not True
        or outputs["allowed_extensions"] != [".json"]
        or any(
            int(outputs[name]) != 0
            for name in ("raw_prediction_or_key_files", "model_or_checkpoint_files", "csv_files")
        )
    ):
        raise RegistrationError("sealed aggregate-only output contract changed")
    if not all(config["prohibitions"].values()):
        raise RegistrationError("a prohibition was disabled")


def _sum_counts(receipts: Sequence[Mapping[str, Any]], name: str) -> dict[str, int]:
    keys = ("tp", "fp", "fn", "tn", "rows")
    return {
        key: int(sum(int(receipt["next_fold_metrics"][name][key]) for receipt in receipts))
        for key in keys
    }


def _run_registered(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = [str(value).lower() for value in config["forbidden_path_tokens"]]
    policy_path, policy_observed = _verify_registered_file(config["governing_policy"], forbidden)
    source_config_path, source_config_observed = _verify_registered_file(
        config["frozen_inputs"]["proposal_bank_source_preregistration"], forbidden
    )
    source_loader_path, source_loader_observed = _verify_registered_file(
        config["frozen_inputs"]["proposal_bank_source_loader"], forbidden
    )
    bank_receipt_path, bank_receipt_observed = _verify_registered_file(
        config["frozen_inputs"]["proposal_bank_receipt"], forbidden
    )
    policy = _load_json(policy_path)
    if (
        policy.get("status") != config["governing_policy"]["required_status"]
        or policy["decision_hierarchy"]["level_1_primary_utility"]["P1"]
        != "pooled row-level binary micro-F1; benefit = candidate - incumbent"
        or policy["official_and_data_boundary"]["official_upload_authorized"] is not False
        or policy["official_and_data_boundary"]["outlier_hard_deletion_authorized"] is not False
    ):
        raise RegistrationError("governing metric-aligned P1 policy changed")

    source_runner = _load_source_runner(source_loader_path)
    source_config = _load_json(source_config_path)
    source_runner._validate_config(source_config)
    frame, source_names, oof_provenance = source_runner._load_inputs(ROOT, source_config)
    _, rebuilt_bank = source_runner._build_events(frame, source_names, source_config)
    bank_receipt = _load_json(bank_receipt_path)
    bank_registration = config["frozen_inputs"]["proposal_bank_receipt"]
    if (
        bank_receipt.get("experiment_id") != bank_registration["required_experiment_id"]
        or bank_receipt.get("config_sha256")
        != config["frozen_inputs"]["proposal_bank_source_preregistration"]["sha256"]
        or bank_receipt.get("provenance", {}).get("complete")
        is not bank_registration["required_provenance_complete"]
        or bank_receipt.get("proposal_bank", {}).get("ordered_bank_sha256")
        != bank_registration["required_bank_sha256"]
        or int(bank_receipt.get("proposal_bank", {}).get("anchor_positive_removed_rows", -1))
        != int(bank_registration["required_anchor_positive_removed_rows"])
        or rebuilt_bank != bank_receipt.get("proposal_bank")
    ):
        raise RegistrationError("rebuilt frozen proposal bank differs from its sealed receipt")

    events, proposal_mask = core.build_event_bank(
        frame,
        source_names,
        event_group=config["candidate_rule"]["event_group"],
        cadence_minutes=int(config["candidate_rule"]["contiguous_cadence_minutes"]),
    )
    if len(events) != int(rebuilt_bank["proposal_events"]) or int(proposal_mask.sum()) != int(
        rebuilt_bank["proposal_rows"]
    ):
        raise RegistrationError("label-free event geometry differs from the sealed bank")

    prefix_receipts: list[dict[str, Any]] = []
    anchor = frame["anchor"].to_numpy(dtype=np.int8)
    combined_candidate = anchor.copy()
    accepted_indices: list[int] = []
    for prefix in config["chronological_prefixes"]:
        receipt, prefix_candidate, prefix_accepted = core.evaluate_prefix(
            frame, events, prefix, config["head"]
        )
        selected = frame["fold"].astype(str).eq(str(prefix["blind_next_fold"])).to_numpy()
        combined_candidate[selected] = prefix_candidate[selected]
        accepted_indices.extend(int(value) for value in prefix_accepted)
        prefix_receipts.append(receipt)

    evaluation_folds = [str(item["blind_next_fold"]) for item in config["chronological_prefixes"]]
    evaluation_mask = frame["fold"].astype(str).isin(evaluation_folds).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    pooled_anchor_counts = core.binary_counts(labels[evaluation_mask], anchor[evaluation_mask])
    pooled_candidate_counts = core.binary_counts(
        labels[evaluation_mask], combined_candidate[evaluation_mask]
    )
    accepted = np.asarray(accepted_indices, dtype=np.int64)
    accepted_labels = labels[accepted] if accepted.size else np.asarray([], dtype=np.int8)
    pooled_added_tp = int(accepted_labels.sum())
    pooled_added_fp = int(len(accepted_labels) - pooled_added_tp)
    anchor_positive_removed = int(np.sum((anchor == 1) & (combined_candidate == 0)))
    pooled_algebra = core.addonly_algebra_sanity(
        pooled_anchor_counts,
        pooled_candidate_counts,
        added_tp=pooled_added_tp,
        added_fp=pooled_added_fp,
        anchor_positive_removed=anchor_positive_removed,
    )
    if not pooled_algebra["pass"]:
        raise RegistrationError("pooled exact add-only F1/2 sanity failed")
    anchor_f1, anchor_num, anchor_den = core.f1_from_counts(pooled_anchor_counts)
    candidate_f1, candidate_num, candidate_den = core.f1_from_counts(pooled_candidate_counts)
    uncertainty = core.paired_event_preserving_day_bootstrap(
        frame,
        combined_candidate,
        events,
        evaluation_folds,
        replicates=int(config["paired_resampling"]["replicates"]),
        seed=int(config["paired_resampling"]["seed"]),
        lower_quantile=float(config["paired_resampling"]["lower_quantile"]),
        upper_quantile=float(config["paired_resampling"]["upper_quantile"]),
    )
    point = candidate_f1 - anchor_f1
    if uncertainty["point_candidate_minus_anchor_f1"] != point:
        raise RegistrationError("bootstrap point does not equal the sole pooled primary metric")
    state = core.evidence_state(
        point, uncertainty["lower_one_sided_95"], uncertainty["upper_one_sided_95"]
    )
    fit_count = sum(int(receipt["fit"]["model_fit_count"]) for receipt in prefix_receipts)
    level_0 = {
        "registered_provenance_complete": oof_provenance.get("complete") is True,
        "frozen_bank_digest_match": True,
        "chronological_prefixes_disjoint_from_next_fold": all(
            item["blind_next_fold"] not in item["fit_folds"]
            for item in config["chronological_prefixes"]
        ),
        "holdout_truth_used_before_acceptance_zero": all(
            receipt["blind_acceptance"]["holdout_truth_fields_used_before_acceptance"] == 0
            for receipt in prefix_receipts
        ),
        "anchor_positive_removed_rows_zero": anchor_positive_removed == 0,
        "prefix_f1_over_2_algebra_exact": all(
            receipt["f1_over_2_hard_sanity"]["pass"] for receipt in prefix_receipts
        ),
        "pooled_f1_over_2_algebra_exact": pooled_algebra["pass"],
        "model_fit_count_exactly_two": fit_count == 2,
        "search_retry_tuning_count_zero": True,
        "official_raw_csv_upload_count_zero": True,
        "target_or_outlier_hard_deletion_count_zero": True,
    }
    if not all(level_0.values()):
        raise RegistrationError("a Level-0 validity gate failed")

    return {
        "schema_version": config["schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "research_only": True,
        "fresh_holdout_available": False,
        "official_action_authorized": False,
        "governing_policy": {
            "status": policy["status"],
            "sha256": policy_observed["sha256"],
            "numeric_effect_margin": 0,
            "sole_primary": "pooled_row_level_binary_micro_f1",
        },
        "level_0_hard_validity": level_0,
        "frozen_proposal_bank": {
            **rebuilt_bank,
            "label_free_geometry_events": len(events),
            "label_free_geometry_rows": int(proposal_mask.sum()),
        },
        "prefixes": {receipt["name"]: receipt for receipt in prefix_receipts},
        "pooled_primary": {
            "evaluation_folds": evaluation_folds,
            "rows": int(evaluation_mask.sum()),
            "anchor_counts": pooled_anchor_counts,
            "candidate_counts": pooled_candidate_counts,
            "anchor_f1": anchor_f1,
            "anchor_f1_numerator": anchor_num,
            "anchor_f1_denominator": anchor_den,
            "candidate_f1": candidate_f1,
            "candidate_f1_numerator": candidate_num,
            "candidate_f1_denominator": candidate_den,
            "candidate_minus_anchor_f1": point,
            "directional_margin": 0,
            "accepted_proposal_events": int(
                sum(receipt["blind_acceptance"]["events_accepted"] for receipt in prefix_receipts)
            ),
            "accepted_proposal_rows": int(len(accepted)),
            "accepted_proposal_tp": pooled_added_tp,
            "accepted_proposal_fp": pooled_added_fp,
            "accepted_proposal_precision": pooled_algebra["proposal_precision"],
        },
        "paired_uncertainty": uncertainty,
        "pooled_f1_over_2_hard_sanity": pooled_algebra,
        "diagnostic_gate_policy": {
            "support_station_layer_month_type_are_diagnostic_only": True,
            "minimum_event_count_gate_applied": False,
            "all_window_gate_applied": False,
            "station_concentration_gate_applied": False,
            "worst_slice_veto_applied": False,
        },
        "provenance": {
            "complete": True,
            "governing_policy": policy_observed,
            "proposal_bank_source_preregistration": source_config_observed,
            "proposal_bank_source_loader": source_loader_observed,
            "proposal_bank_receipt": bank_receipt_observed,
            "registered_oof": oof_provenance,
        },
        "execution_audit": {
            "authorized_attempts": 1,
            "attempts_executed": 1,
            "model_fit_count": fit_count,
            "threshold_search_count": 0,
            "hyperparameter_or_feature_search_count": 0,
            "retry_or_tuning_count": 0,
            "row_prediction_or_key_files": 0,
            "model_or_checkpoint_files": 0,
            "prediction_csv_count": 0,
            "official_interface_rows_read": 0,
            "raw_training_rows_read": 0,
            "raw_temp_rows_read": 0,
            "auxiliary_psal_depth_rows_read": 0,
            "target_positive_rows_removed": 0,
            "outlier_hard_deleted_rows": 0,
            "anchor_positive_removed_rows": anchor_positive_removed,
            "upload_count": 0,
        },
        "outlier_policy": config["outlier_policy"],
        "code_and_registration_sha256": {
            "config": CONFIG_SHA256,
            "core_module": _sha256(Path(core.__file__).resolve()),
            "runner": _sha256(Path(__file__).resolve()),
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execute_one_shot() -> dict[str, Any]:
    if _sha256(CONFIG_PATH) != CONFIG_SHA256:
        raise RegistrationError("sealed preregistration sha256 changed")
    config = _load_json(CONFIG_PATH)
    _validate_config(config)
    lock_path = (ROOT / config["outputs"]["attempt_lock"]).resolve()
    result_path = (ROOT / config["outputs"]["result"]).resolve()
    started = datetime.now(UTC)
    _atomic_json(
        lock_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "attempt": 1,
            "authorized_attempts": 1,
            "started_at_utc": started.isoformat(),
            "config_sha256": CONFIG_SHA256,
            "exclusive_create": True,
            "retry_authorized": False,
        },
    )
    start_clock = time.perf_counter()
    result = _run_registered(config)
    finished = datetime.now(UTC)
    result["runtime"] = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "wall_seconds": time.perf_counter() - start_clock,
    }
    _atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-one-shot",
        action="store_true",
        help="Consume the sole attempt and exclusively create the aggregate result.",
    )
    args = parser.parse_args()
    if not args.execute_one_shot:
        parser.error("the sealed runner requires --execute-one-shot")
    result = execute_one_shot()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
