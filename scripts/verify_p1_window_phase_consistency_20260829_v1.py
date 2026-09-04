"""Independent aggregate verifier for the P1 window-phase one-shot run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_window_phase_consistency_20260829_v1"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
AGGREGATE_PATH = ROOT / "reports" / EXPERIMENT_ID / "aggregate.json"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
TRUTH_PATH = ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
ANCHOR_PATH = ROOT / "artifacts" / "p1_current_router_oof_anchor_v1" / "anchor.parquet"
Q2_GRID_PATH = (
    ROOT
    / "artifacts"
    / "p1_incumbent_preserving_mstcn_asrf_v2"
    / "q2_qualification_grid_blind.npz"
)
KEYS = ("station", "year", "layer", "time")


class VerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_key_sha(frame: Any) -> str:
    digest = hashlib.sha256()
    for column in KEYS:
        digest.update(column.encode("ascii") + b"\0")
        for value in frame[column].tolist():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def _binary_metrics(truth: Any, prediction: Any) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": float(2 * tp / denominator) if denominator else 0.0,
    }


def _load_receipt(phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    name = (
        "q2_alternate_tiling_blind_receipt.json"
        if phase == "q2"
        else f"{phase}_paired_view_blind_receipt.json"
    )
    receipt_path = ARTIFACT_DIR / name
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    score_path = (ARTIFACT_DIR / str(receipt["score_path"])).resolve()
    if score_path.parent != ARTIFACT_DIR.resolve() or not score_path.is_file():
        raise VerificationError(f"{phase} score path escapes or is absent")
    if int(score_path.stat().st_size) != int(receipt["score_bytes"]):
        raise VerificationError(f"{phase} score byte count changed")
    if _sha256(score_path) != receipt["score_sha256"]:
        raise VerificationError(f"{phase} score hash changed")
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    inventory = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    if inventory != receipt["array_inventory"]:
        raise VerificationError(f"{phase} array inventory changed")
    if receipt.get("same_fold_truth_columns_opened_before_receipt") != 0:
        raise VerificationError(f"{phase} truth-firewall attestation changed")
    return receipt, arrays


def _fold_frame(path: Path, fold: str, columns: list[str]) -> Any:
    table = ds.dataset(path, format="parquet").to_table(
        columns=columns, filter=ds.field("fold") == fold
    )
    return table.to_pandas()


def _aligned_truth_and_anchor(phase: str, receipt: dict[str, Any]) -> tuple[Any, Any]:
    fold = {"q2": "2025_q2", "q3": "2025_q3", "q4": "2025_q4"}[phase]
    truth = _fold_frame(TRUTH_PATH, fold, [*KEYS, "label", "fold"])
    anchor = _fold_frame(
        ANCHOR_PATH, fold, [*KEYS, "current_router_prediction", "fold"]
    )
    if _ordered_key_sha(truth) != receipt["ordered_holdout_key_sha256"]:
        raise VerificationError(f"{phase} truth key identity changed")
    if _ordered_key_sha(anchor) != receipt["ordered_holdout_key_sha256"]:
        raise VerificationError(f"{phase} anchor key identity changed")
    if any(not truth[column].astype(str).equals(anchor[column].astype(str)) for column in KEYS):
        raise VerificationError(f"{phase} truth/anchor keys differ")
    return (
        truth["label"].to_numpy(dtype=np.int8),
        anchor["current_router_prediction"].to_numpy(dtype=np.int8),
    )


def verify() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    terminal = json.loads((ARTIFACT_DIR / "terminal_result.json").read_text(encoding="utf-8"))
    if aggregate != terminal:
        raise VerificationError("tracked aggregate differs from append-only terminal")
    if aggregate.get("experiment_id") != EXPERIMENT_ID:
        raise VerificationError("aggregate experiment identity changed")
    q2_receipt, q2_arrays = _load_receipt("q2")
    with np.load(Q2_GRID_PATH, allow_pickle=False) as grid:
        row_id = np.flatnonzero((grid["widths"] == 512) & (grid["epochs"] == 150))
        threshold_id = np.flatnonzero(np.isclose(grid["thresholds"], 0.8, rtol=0.0, atol=0.0))
        if len(row_id) != 1 or len(threshold_id) != 1:
            raise VerificationError("fixed e150 grid cell is absent")
        replay = {
            "decoder_probability_bitwise_equal": bool(
                np.array_equal(q2_arrays["default_probability"], grid["row_probability"][row_id[0]])
            ),
            "boundary_probability_bitwise_equal": bool(
                np.array_equal(
                    q2_arrays["default_boundary_probability"],
                    grid["boundary_probability"][row_id[0]],
                )
            ),
            "proposal_bitwise_equal": bool(
                np.array_equal(
                    q2_arrays["default_proposal"],
                    grid["proposal"][row_id[0], threshold_id[0]],
                )
            ),
        }
    truth, anchor = _aligned_truth_and_anchor("q2", q2_receipt)
    q99 = float(
        np.quantile(
            np.abs(q2_arrays["default_probability"] - q2_arrays["plus256_probability"]),
            0.99,
        )
    )
    xor_rows = int(np.sum(q2_arrays["default_proposal"] != q2_arrays["plus256_proposal"]))
    anchor_score = _binary_metrics(truth, anchor)
    candidate_score = _binary_metrics(truth, q2_arrays["average_candidate"])
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    limits = config["q2_preflight"]["kill_if_any"]
    q2_checks = {
        "default_replay_bitwise_identical": all(replay.values()),
        "q99_absolute_probability_difference": q99
        >= float(limits["q99_absolute_probability_difference_below"]),
        "proposal_xor_rows": xor_rows >= int(limits["proposal_xor_rows_below"]),
        "fixed_average_anchor_union_delta_f1": delta
        >= float(limits["fixed_average_anchor_union_delta_f1_below"]),
    }
    reported = aggregate["q2_preflight"]
    if reported["replay_identity"] != replay or reported["gate_checks"] != q2_checks:
        raise VerificationError("Q2 replay identity or gates do not recompute")
    for name, value in (
        ("q99_absolute_probability_difference", q99),
        ("fixed_average_anchor_union_delta_f1", delta),
    ):
        if not math_isclose(float(reported[name]), value):
            raise VerificationError(f"Q2 metric differs: {name}")
    if int(reported["proposal_xor_rows"]) != xor_rows:
        raise VerificationError("Q2 XOR count differs")
    warm_started = bool(aggregate["paired_view_training_started"])
    if all(q2_checks.values()) != warm_started:
        raise VerificationError("warm-start decision differs from Q2 gate conjunction")
    confirmatory_recomputed: dict[str, Any] | None = None
    if warm_started:
        fold_rows: dict[str, Any] = {}
        truth_parts: list[Any] = []
        anchor_parts: list[Any] = []
        candidate_parts: list[Any] = []
        for phase in ("q3", "q4"):
            receipt, arrays = _load_receipt(phase)
            phase_truth, phase_anchor = _aligned_truth_and_anchor(phase, receipt)
            candidate = arrays["candidate"]
            anchor_metric = _binary_metrics(phase_truth, phase_anchor)
            candidate_metric = _binary_metrics(phase_truth, candidate)
            fold_rows[phase] = {
                "delta_f1": float(candidate_metric["f1"] - anchor_metric["f1"]),
                "anchor": anchor_metric,
                "candidate": candidate_metric,
            }
            truth_parts.append(phase_truth)
            anchor_parts.append(phase_anchor)
            candidate_parts.append(candidate)
        pooled_truth = np.concatenate(truth_parts)
        pooled_anchor = np.concatenate(anchor_parts)
        pooled_candidate = np.concatenate(candidate_parts)
        pooled_anchor_metric = _binary_metrics(pooled_truth, pooled_anchor)
        pooled_candidate_metric = _binary_metrics(pooled_truth, pooled_candidate)
        pooled_delta = float(pooled_candidate_metric["f1"] - pooled_anchor_metric["f1"])
        removed = int(np.sum((pooled_anchor == 1) & (pooled_candidate == 0)))
        gates = {
            "q3_delta_f1_strictly_positive": fold_rows["q3"]["delta_f1"] > 0.0,
            "q4_delta_f1_strictly_positive": fold_rows["q4"]["delta_f1"] > 0.0,
            "pooled_delta_f1_min": pooled_delta
            >= float(config["confirmatory_gate"]["pooled_delta_f1_min"]),
            "anchor_positive_removed_rows": removed == 0,
        }
        if aggregate["confirmatory"]["gate_checks"] != gates:
            raise VerificationError("confirmatory gates do not recompute")
        confirmatory_recomputed = {
            "q3_delta_f1": fold_rows["q3"]["delta_f1"],
            "q4_delta_f1": fold_rows["q4"]["delta_f1"],
            "pooled_delta_f1": pooled_delta,
            "anchor_positive_removed_rows": removed,
            "gate_checks": gates,
        }
    csv_files = [path.name for path in ARTIFACT_DIR.rglob("*.csv")]
    if csv_files:
        raise VerificationError("run namespace contains a forbidden CSV")
    return {
        "schema_version": "p1.window_phase_consistency.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": aggregate["status"],
        "q2": {
            "q99_absolute_probability_difference": q99,
            "proposal_xor_rows": xor_rows,
            "fixed_average_anchor_union_delta_f1": delta,
            "gate_checks": q2_checks,
        },
        "confirmatory": confirmatory_recomputed,
        "aggregate_matches_terminal": True,
        "sealed_hashes_verified": True,
        "forbidden_csv_count": 0,
        "result": "PASS",
    }


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-12


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
