"""Exactly-once dtype-tolerant materializer for a QA-passed v18r1 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18 import (  # noqa: E402
    CANDIDATE,
    KEYS,
    attach_energy,
    canonical,
    ecdf,
    fit_theta,
    load_energy_history,
    load_historical,
    midpoint_prediction,
    official_frame,
    sequential_energy_rank,
    sha256,
    v14b_correction,
    write_new,
)

EXPERIMENT_ID = "p3_kma_v14b_v16_midpoint_materializer_20260831_v18m1"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
RECOVERY_RESULT = ROOT / "artifacts/p3_kma_v14b_v16_midpoint_cycle_20260831_v18r1/result.json"
RECOVERY_QA = ROOT / "reports/p3_kma_v14b_v16_midpoint_cycle_20260831_v18r1/independent-qa.json"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_V14B_V16_MIDPOINT_V18_RECOVERED")


class ContractError(RuntimeError):
    pass


def key_values_and_order_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    return bool(np.array_equal(left.to_numpy(), right.to_numpy()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "MATERIALIZER_READY"}))
        return 0
    if ARTIFACT_DIR.exists() or LOCK.exists() or DELIVERY.exists():
        raise ContractError("materializer artifact, lock, or delivery exists")
    recovered = json.loads(RECOVERY_RESULT.read_text(encoding="utf-8"))
    qa = json.loads(RECOVERY_QA.read_text(encoding="utf-8"))
    if recovered["decision"] != "PASS_INTERNAL_AWAITING_SEPARATE_MATERIALIZER" or recovered["passing_candidate_count"] != 1 or qa["status"] != "PASS":
        raise ContractError("recovery result/QA is not materialization eligible")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runner_hash = sha256(Path(__file__))
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_EXACTLY_ONCE_MATERIALIZER", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "recovery_result_sha256": sha256(RECOVERY_RESULT), "recovery_qa_sha256": sha256(RECOVERY_QA), "allowed_fix": "dtype_independent_key_value_order_equality_only"}))
    frame, _ = load_historical()
    history = load_energy_history()
    frame = attach_energy(frame, history)
    training_rank = sequential_energy_rank(frame, history)
    active = frame["lead_h"].isin([18, 24])
    correction = v14b_correction(frame.loc[active], training_rank[active.to_numpy()])
    theta, _ = fit_theta(frame.loc[active], correction)
    official, champion = official_frame()
    energy = pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", "wave_energy_current"])
    official = official.merge(energy, on=["case_id", "station"], how="left", validate="many_to_one")
    if not key_values_and_order_equal(champion[KEYS], official[KEYS]) or official["wave_energy_current"].isna().any():
        raise ContractError("official key values/order or energy failed")
    prefix = np.sort(history["wave_energy_current"].to_numpy(dtype=np.float64))
    rank = ecdf(prefix, official["wave_energy_current"].to_numpy(dtype=np.float64))
    official_correction = v14b_correction(official, rank)
    prediction, _ = midpoint_prediction(official, official_correction, theta)
    submission = official[KEYS].copy()
    submission["hs_pred"] = prediction
    if len(submission) != 1200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
        raise ContractError("submission structural QA failed")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    path = DELIVERY / "P3_submission.csv"
    payload = submission.to_csv(index=False, lineterminator="\n").encode()
    write_new(path, payload)
    output = {"candidate": CANDIDATE, "path": str(path), "rows": 1200, "sha256": hashlib.sha256(payload).hexdigest(), "uploads": 0}
    result = {"schema_version": "p3.midpoint_materializer.result.v18m1", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_MATERIALIZED_NOT_UPLOADED", "output": output, "data_access_this_materializer": {"official_test_index_rows_read": 1200, "official_test_context_rows_read": 200, "official_feature_rows_read": 200, "official_component_prediction_rows_read": 3600, "hidden_truth_rows_read": 0, "uploads": 0}, "cumulative_official_access_including_failure_diagnostics": {"official_test_index_rows_read": 4800, "official_test_context_rows_read": 800, "official_feature_rows_read": 800, "official_component_prediction_rows_read": 14400, "hidden_truth_rows_read": 0, "uploads": 0}, "provenance": {"runner_sha256": runner_hash, "recovery_result_sha256": sha256(RECOVERY_RESULT), "recovery_qa_sha256": sha256(RECOVERY_QA)}, "prediction_or_weight_changed": False, "dtype_only_validation_fix": True}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    write_new(DELIVERY / "SET_MANIFEST.json", canonical({"experiment_id": EXPERIMENT_ID, "outputs": [output], "uploads": 0}))
    print(json.dumps({"status": "COMPLETE", "output": output}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
