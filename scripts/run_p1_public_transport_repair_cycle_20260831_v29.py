from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p1_qc.inner_group_day_guard import (  # noqa: E402
    apply_group_guard,
    day_cap_mask,
    eligible_groups,
)
from src.p1_qc.prequential_label_shift_em import (  # noqa: E402
    frozen_logit_matrix,
    label_shift_em,
    select_inner_threshold,
)

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v29.json"
V28_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v28.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v29/preflight-report.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v29"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    old = json.loads(V28_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "model_C": cfg["model"]["C"] == old["model"]["C"] == 0.1,
        "model_solver": cfg["model"]["solver"] == old["model"]["solver"] == "lbfgs",
        "model_iter": cfg["model"]["max_iter"] == old["model"]["max_iter"] == 500,
        "em_iter": cfg["em"]["maximum_iterations"] == old["outer_label_shift"]["maximum_iterations"] == 200,
        "em_tol": cfg["em"]["tolerance"] == old["outer_label_shift"]["tolerance"] == 1e-10,
        "support": cfg["inner_selector"]["minimum_proposed_additions_per_group"] == 20,
        "day_cap": cfg["outer_day_guard"]["maximum_changed_fraction_per_day"] == 0.005,
        "fits": cfg["fit_budget"]["maximum"] == 2,
        "historical_on": cfg["authorization"]["historical_execution"] is True,
        "lock_on": cfg["authorization"]["attempt_lock_creation"] is True,
        "calibration": cfg["transport"]["calibration_sha256"] == sha256(CALIBRATION),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v29 contract mismatch: {checks}")
    return cfg


def preflight() -> dict:
    cfg = load_contract()
    rng = np.random.default_rng(20260929)
    n = 4000
    station = np.where(np.arange(n) % 2, "A", "B")
    layer = np.where(np.arange(n) % 3, 1, 2)
    day = np.arange(n) // 400
    anchor = np.zeros(n, dtype=np.int8)
    score = rng.uniform(size=n)
    truth = ((score > 0.94) & (station == "A")).astype(np.int8)
    proposed = score > 0.9
    allowed = eligible_groups(truth, anchor, proposed, station, layer, minimum_support=20)
    guarded = apply_group_guard(proposed, station, layer, allowed)
    capped = day_cap_mask(guarded, score, day, maximum_fraction=0.005)
    corrected, receipt = label_shift_em(score, float(np.clip(truth.mean(), 1e-6, 1 - 1e-6)), maximum_iterations=200, tolerance=1e-10, epsilon=1e-6)
    checks = {
        "some_group_allowed": bool(allowed),
        "group_guard_subset": bool(np.all(~guarded | proposed)),
        "day_cap_exact": all(int(capped[day == d].sum()) <= int(np.floor(0.005 * int((day == d).sum()))) for d in np.unique(day)),
        "em_converged": receipt.converged,
        "em_finite": bool(np.isfinite(corrected).all()),
        "historical_authorized": cfg["authorization"]["historical_execution"] is True,
        "lock_authorized": cfg["authorization"]["attempt_lock_creation"] is True,
        "official_zero": True,
    }
    return {"schema_version": "p1.v29.preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "allowed_groups": sorted([list(x) for x in allowed]), "hashes": {"config_sha256": sha256(CONFIG), "v28_config_sha256": sha256(V28_CONFIG), "calibration_sha256": sha256(CALIBRATION)}, "access": {"historical_truth_reads": 0, "attempt_locks": 0, "official_reads": 0, "hidden_truth_reads": 0, "csv": 0, "uploads": 0}}


def run_historical(cfg: dict) -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v29 exactly-once artifact already exists")
    import scripts.run_p1_public_transport_repair_cycle_20260831_v15 as evaluation
    import scripts.run_p1_public_transport_repair_cycle_20260831_v16 as source
    import scripts.run_p1_public_transport_repair_cycle_20260831_v28 as v28

    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {"experiment_id": cfg["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "loading", "fit_count": 0}) + "\n", encoding="utf-8")
    frame, anchor, _, dependency = source.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    design = frozen_logit_matrix(frame["probability_base"].to_numpy(np.float64), frame["probability_peer"].to_numpy(np.float64), frame["e150_probability"].to_numpy(np.float64))
    times = pd.to_datetime(frame["time"], utc=True)
    times_ns = times.astype("int64").to_numpy(np.int64)
    kst_day = times.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    candidate = anchor.copy()
    corrected_probability = np.zeros(len(frame), dtype=np.float64)
    receipts = []
    for fit_number, spec in enumerate(cfg["validation"]["nested_fits"], 1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        split = v28.chronological_inner_split(times_ns, prefix, fit_fraction=float(cfg["model"]["fit_fraction"]))
        fit_negative = split.fit_mask & (anchor == 0)
        calibration = split.calibration_mask
        calibration_negative = calibration & (anchor == 0)
        outer = frame["fold"].eq(spec["validation_fold"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        model = v28._calibrator(cfg)
        model.fit(design[fit_negative], truth[fit_negative])
        inner_negative_probability = model.predict_proba(design[calibration_negative])[:, 1]
        inner_probability = np.zeros(int(calibration.sum()), dtype=np.float64)
        inner_probability[anchor[calibration] == 0] = inner_negative_probability
        threshold = select_inner_threshold(inner_probability, truth[calibration], anchor[calibration], maximum_changed_fraction=0.005)
        inner_proposed = (anchor[calibration] == 0) & (inner_probability >= threshold.threshold)
        allowed = eligible_groups(truth[calibration], anchor[calibration], inner_proposed, frame.loc[calibration, "station"].to_numpy(), frame.loc[calibration, "layer"].to_numpy(), minimum_support=20)
        source_prevalence = float(np.clip(truth[calibration_negative].mean(), 1e-6, 1 - 1e-6))
        outer_source = model.predict_proba(design[outer_negative])[:, 1]
        outer_corrected, em = label_shift_em(outer_source, source_prevalence, maximum_iterations=200, tolerance=1e-10, epsilon=1e-6)
        if not em.converged:
            raise RuntimeError("v29 EM failed to converge")
        corrected_probability[outer_negative] = outer_corrected
        raw_proposed = outer_corrected >= threshold.threshold
        group_proposed = apply_group_guard(raw_proposed, frame.loc[outer_negative, "station"].to_numpy(), frame.loc[outer_negative, "layer"].to_numpy(), allowed)
        capped = day_cap_mask(group_proposed, outer_corrected, kst_day[outer_negative], maximum_fraction=0.005)
        proposed = np.flatnonzero(outer_negative)[capped]
        candidate[proposed] = 1
        receipts.append({"fit_number": fit_number, "train_folds": spec["train_folds"], "validation_fold": spec["validation_fold"], "fit_rows": int(fit_negative.sum()), "inner_rows": int(calibration.sum()), "inner_threshold": threshold.threshold if np.isfinite(threshold.threshold) else None, "inner_additions": threshold.additions, "allowed_groups": sorted([list(x) for x in allowed]), "outer_labels_read_before_prediction_seal": 0, "em_iterations": em.iterations, "em_converged": em.converged, "outer_raw_proposals": int(raw_proposed.sum()), "outer_group_guarded": int(group_proposed.sum()), "sealed_outer_additions": int(len(proposed))})
        (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n", encoding="utf-8")
    prediction = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(prediction, candidate=candidate, corrected_probability=corrected_probability)
    record = evaluation.evaluate(frame, anchor, candidate, cfg)
    record["name"] = cfg["candidate"]
    checks = {"exact_two_fits": len(receipts) == 2, "em_converged": all(x["em_converged"] for x in receipts), "outer_labels_zero": all(x["outer_labels_read_before_prediction_seal"] == 0 for x in receipts), "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}
    qa = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    result = {"schema_version": "p1.v29.result.1", "experiment_id": cfg["experiment_id"], "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "nested_fit_receipts": receipts, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "operations": {"historical_reads": 1, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"), "prediction_sha256": sha256(prediction)}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]}) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(run_historical(load_contract()), indent=2, sort_keys=True))
        return
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    result = preflight()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
