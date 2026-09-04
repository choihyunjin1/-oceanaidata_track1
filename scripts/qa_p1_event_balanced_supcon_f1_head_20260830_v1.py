"""Independent recomputation for the terminal P1 SupCon/top-k screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXPERIMENT_ID = "p1_event_balanced_supcon_f1_head_20260830_v1"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
CONFIG_PATH = ROOT / "configs/experiments/p1_event_balanced_supcon_f1_head_20260830_v1.json"
RUNNER_PATH = ROOT / "scripts/run_p1_event_balanced_supcon_f1_head_20260830_v1.py"
HELPER_PATH = ROOT / "src/p1_qc/event_balanced_supcon_f1.py"
SOBOL_PATH = ROOT / "scripts/run_p1_mstcn_sobol_hpo_20260829_v1.py"

SPEC = importlib.util.spec_from_file_location("p1_sobol_for_supcon_independent_qa", SOBOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen Sobol runner")
SOBOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOBOL
SPEC.loader.exec_module(SOBOL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary(truth: Any, prediction: Any) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _type_macro(base: Any, truth: Any, prediction: Any) -> float:
    label = truth["label"].to_numpy(dtype=np.int8)
    raw = truth["anomaly_type"].fillna("").astype(str).str.lower().to_numpy()
    values: list[float] = []
    for name in base.TYPE_NAMES:
        present = np.asarray(
            [name in {token.strip() for token in value.split("+") if token.strip()} for value in raw],
            dtype=bool,
        )
        universe = (label == 0) | present
        values.append(float(_binary(present[universe].astype(np.int8), np.asarray(prediction)[universe])["f1"]))
    return sum(values) / len(values)


def _maximum_station_share(stations: Any, candidate: Any, control: Any) -> float:
    names = np.asarray(stations).astype(str)
    changed = np.asarray(candidate, dtype=np.int8) != np.asarray(control, dtype=np.int8)
    total = int(changed.sum())
    if total == 0:
        return 0.0
    return max(float(np.sum(changed & (names == station))) / total for station in np.unique(names))


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    terminal = json.loads((ARTIFACT_DIR / "terminal_result.json").read_text(encoding="utf-8"))
    preflight = json.loads((ARTIFACT_DIR / "preflight.json").read_text(encoding="utf-8"))
    commitment = json.loads((ARTIFACT_DIR / "commitment.json").read_text(encoding="utf-8"))
    lock = json.loads(
        (ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json").read_text(encoding="utf-8")
    )
    design_sha = hashlib.sha256(
        json.dumps(commitment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    base = SOBOL._load_base(root=ROOT)
    source = base._canonical_config()
    surfaces = base.load_blind_surfaces(source, root=ROOT)
    sobol_config = json.loads(
        (ROOT / config["source_pins"]["sobol_config"]["path"]).read_text(encoding="utf-8")
    )
    controls = SOBOL._control_candidates(base, sobol_config, surfaces, root=ROOT)
    phases = tuple(config["training_contract"]["phase_order"])
    folds = {"q2": "2025_q2", "q3": "2025_q3", "q4": "2025_q4"}
    holdouts: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    truths: dict[str, Any] = {}
    receipt_checks: dict[str, Any] = {}
    for phase in phases:
        _encoder, _training, holdout, _split = base._prepare_phase_surfaces(
            surfaces, source, phase, root=ROOT
        )
        receipt_path = ARTIFACT_DIR / f"{phase}_event_balanced_supcon_f1_blind_receipt.json"
        receipt = SOBOL._verify_receipt(
            receipt_path,
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
        )
        arrays[phase] = SOBOL._load_arrays(
            receipt_path,
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
        )
        truths[phase] = SOBOL._load_truth(
            base,
            source,
            holdout.surface,
            [receipt_path],
            fold=folds[phase],
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
            root=ROOT,
        )
        fit = receipt["fit_receipts"]
        history_path = ARTIFACT_DIR / fit[0]["history_artifact"]["path"]
        history = json.loads(history_path.read_text(encoding="utf-8"))
        receipt_checks[phase] = {
            "one_fit": len(fit) == 1,
            "seed": int(fit[0]["seed"]),
            "epochs": int(fit[0]["epochs"]),
            "history_records": len(history),
            "history_sha256_match": _sha256(history_path) == fit[0]["history_artifact"]["sha256"],
            "history_all_finite": all(
                all(not isinstance(value, float) or np.isfinite(value) for value in row.values())
                for row in history
            ),
            "checkpoint_persisted": bool(fit[0]["checkpoint_persisted"]),
            "truth_before_receipt": int(receipt["same_fold_holdout_truth_columns_opened_before_receipt"]),
            "official_rows": int(receipt["official_interface_rows_read"]),
        }
        holdouts[phase] = holdout

    recomputed_folds: dict[str, Any] = {}
    for phase in phases:
        truth = truths[phase]["label"].to_numpy(dtype=np.int8)
        candidate = arrays[phase]["candidate"]
        proposal = arrays[phase]["proposal"]
        control = controls[phase]
        candidate_metrics = _binary(truth, candidate)
        control_metrics = _binary(truth, control)
        additions = (proposal == 1) & (holdouts[phase].surface.anchor == 0)
        proposal_metrics = _binary(truth, additions.astype(np.int8))
        recomputed_folds[phase] = {
            "delta_f1": float(candidate_metrics["f1"] - control_metrics["f1"]),
            "type_macro_delta_f1": _type_macro(base, truths[phase], candidate)
            - _type_macro(base, truths[phase], control),
            "proposal_precision": float(proposal_metrics["precision"]),
            "proposal_precision_floor": 0.5 * float(control_metrics["f1"]),
            "anchor_positive_removed_rows": int(
                np.sum((holdouts[phase].surface.anchor == 1) & (candidate == 0))
            ),
            "maximum_station_changed_row_share": _maximum_station_share(
                holdouts[phase].surface.keys["station"], candidate, control
            ),
        }
    pooled_candidate = np.concatenate([arrays[phase]["candidate"] for phase in phases])
    pooled_control = np.concatenate([controls[phase] for phase in phases])
    import pandas as pd

    combined_truth = pd.concat([truths[phase] for phase in phases], ignore_index=True)
    pooled_type_delta = _type_macro(base, combined_truth, pooled_candidate) - _type_macro(
        base, combined_truth, pooled_control
    )
    pooled_station = np.concatenate(
        [holdouts[phase].surface.keys["station"].astype(str).to_numpy() for phase in phases]
    )
    recomputed_checks = {
        "all_windows_delta_f1_strictly_positive": all(
            recomputed_folds[phase]["delta_f1"] > 0 for phase in phases
        ),
        "pooled_anomaly_type_macro_delta_f1_strictly_positive": pooled_type_delta > 0,
        "proposal_precision_gt_half_incumbent_f1_each_window": all(
            recomputed_folds[phase]["proposal_precision"]
            > recomputed_folds[phase]["proposal_precision_floor"]
            for phase in phases
        ),
        "anchor_positive_removed_rows_eq_0": sum(
            recomputed_folds[phase]["anchor_positive_removed_rows"] for phase in phases
        )
        == 0,
        "maximum_station_changed_row_share_lte_0_8": _maximum_station_share(
            pooled_station, pooled_candidate, pooled_control
        )
        <= 0.8,
    }
    numeric_match = all(
        np.isclose(
            recomputed_folds[phase][name],
            terminal["folds"][phase][name],
            atol=1e-12,
            rtol=0.0,
        )
        for phase in phases
        for name in ("delta_f1", "type_macro_delta_f1", "proposal_precision_floor")
    )
    forbidden_files = [
        path.relative_to(ARTIFACT_DIR).as_posix()
        for path in ARTIFACT_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".csv", ".pt", ".pth", ".ckpt", ".parquet"}
    ]
    checks = {
        "terminal_status_no_go": terminal["status"] == "NO_GO_LOW_FIDELITY_SCREEN",
        "terminal_fit_count_three": int(terminal["historical_fit_count"]) == 3,
        "config_hash_match": _sha256(CONFIG_PATH) == preflight["config_sha256"] == lock["config_sha256"],
        "runner_hash_match": _sha256(RUNNER_PATH) == preflight["runner_sha256"] == lock["runner_sha256"],
        "helper_hash_match": _sha256(HELPER_PATH) == preflight["helper_sha256"],
        "all_receipts_exact": all(
            value["one_fit"]
            and value["seed"] == 20260830
            and value["epochs"] == 25
            and value["history_records"] == 25
            and value["history_sha256_match"]
            and value["history_all_finite"]
            and not value["checkpoint_persisted"]
            and value["truth_before_receipt"] == 0
            and value["official_rows"] == 0
            for value in receipt_checks.values()
        ),
        "numeric_results_match": bool(numeric_match),
        "gate_vector_match": recomputed_checks == terminal["checks"],
        "pooled_type_delta_match": bool(
            np.isclose(pooled_type_delta, terminal["pooled"]["type_macro_delta_f1"], atol=1e-12, rtol=0.0)
        ),
        "all_three_seals_before_truth_claim": bool(
            terminal["all_three_blind_predictions_sealed_before_truth_read"]
        )
        and int(terminal["holdout_truth_rows_read_before_all_seals"]) == 0,
        "no_forbidden_artifact_files": not forbidden_files,
        "official_csv_upload_zero": int(terminal["official_interface_rows_read"]) == 0
        and not terminal["csv_created"]
        and not terminal["upload_performed"],
        "no_confirmation": not terminal["three_seed_confirmation_authorized"]
        and not terminal["three_seed_confirmation_executed"],
    }
    result = {
        "schema_version": "p1.event_balanced_supcon_f1_head.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "receipt_checks": receipt_checks,
        "recomputed_folds": recomputed_folds,
        "recomputed_pooled_type_macro_delta_f1": pooled_type_delta,
        "recomputed_checks": recomputed_checks,
        "forbidden_artifact_files": forbidden_files,
        "official_interface_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "independent-qa.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
