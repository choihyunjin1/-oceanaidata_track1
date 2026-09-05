"""Independent arithmetic and lineage QA inside a standalone P3 package."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as r  # noqa: E402


def arithmetic(target, prediction):
    if len(target) != len(prediction) or not len(target):
        raise ValueError("paired nonempty population required")
    sse = math.fsum((float(p) - float(y)) ** 2 for y, p in zip(target, prediction, strict=True))
    return sse, math.sqrt(sse / len(target))


def audit(training_only):
    started = time.perf_counter()
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    r.guard(source, official=not training_only)
    config = json.loads(r.CONFIG.read_text(encoding="utf-8"))
    r.verify(config, source, prepared=True)
    training_path = r.REPORT / "training-result.json"
    train = json.loads(training_path.read_text(encoding="utf-8"))
    pre = json.loads((r.REPORT / "prepare.json").read_text(encoding="utf-8"))
    replay = json.loads((r.REPORT / "fresh-process-replay.json").read_text(encoding="utf-8"))
    regenerated = json.loads((r.REPORT / "regenerated-oof.json").read_text(encoding="utf-8"))
    frame = r.pd.read_parquet(r.WORK / "oof.parquet")
    checks = {}
    checks["recovered_fresh_model_training_complete"] = (
        train["status"] == "TRAINING_COMPLETE_RECOVERED_FRESH_MODELS"
        and not train["empty_03_model_verified"]
        and not train["uninterrupted_cold_start_proof"]
    )
    checks["exact_new_fit_budget"] = (
        train["new_backbone_fits"] == 1
        and train["new_full_router_fits"] == 1
        and train["inherited_successful_backbone_fits"] == 7
        and train["inherited_interrupted_backbone_attempts"] == 1
        and train["repeated_historical_backbone_fits"] == 0
        and train["repeated_prequential_router_fits"] == 0
    )
    checks["recovery_contract_hash"] = train["recovery_inputs_sha256"] == r.sha(
        r.ROOT / "RECOVERY_INPUTS.json"
    )
    checks["exact_fit_budget"] = (
        train["backbone_fits"],
        train["historical_backbone_fits"],
        train["full_backbone_fits"],
        train["prequential_router_fits"],
        train["full_router_fits"],
    ) == (8, 6, 2, 2, 1)
    checks["full_training_population"] = (
        train["full_single_rows"] == 146160 and train["full_multi_cases"] == 24360
    )
    checks["fixed_feature_population"] = (
        pre["features"] == 591 and pre["train_anchors"] == 24360 and pre["cases"] == 181
    )
    checks["code_config_hash"] = train["runner_sha256"] == r.sha(r.CODE / "run.py") and train[
        "config_sha256"
    ] == r.sha(r.CONFIG)
    for name, digest in {**train["model_sha256"], **train["files_sha256"]}.items():
        checks["own_file_hash:" + name] = r.sha(r.ROOT / name) == digest
    checks["nine_saved_models"] = len(train["model_sha256"]) == 9
    checks["fresh_pid"] = replay["fresh_pid"] != train["pid"] and os.getpid() != train["pid"]
    checks["exact_saved_prediction_replay"] = (
        replay["status"] == "PASS" and replay["max_abs_prediction_error_m"] == 0
    )
    checks["replay_result_link"] = replay["training_result_sha256"] == r.sha(training_path)
    checks["no_old_asset_or_pretraining"] = (
        train["old_model_cache_oof_answer_reads"] == 0 and train["pretrained_weights"] == 0
    )
    checks["training_official_csv_upload_zero"] = (
        train["official_rows"] == 0 and train["csv_rows"] == 0 and train["uploads"] == 0
    )
    checks["unique_complete_historical_keys"] = (
        len(frame) == 1086
        and frame.anchor_id.nunique() == 181
        and not frame.duplicated(["anchor_id", "station", "lead_h"]).any()
    )
    checks["all_six_leads"] = (
        frame.groupby("anchor_id").lead_h.agg(lambda x: tuple(sorted(x)) == r.LEADS).all()
    )
    sse, rmse = arithmetic(frame.target_hs, frame.final_prediction)
    checks["independent_sse"] = math.isclose(sse, regenerated["sse_m2"], rel_tol=0, abs_tol=1e-9)
    checks["independent_rmse"] = math.isclose(
        rmse, train["regenerated_oof_rmse_m"], rel_tol=0, abs_tol=1e-12
    )
    checks["oof_hash"] = r.sha(r.WORK / "oof.parquet") == regenerated["oof_sha256"]
    wave = r.pd.read_csv(source / "train_wave.csv", usecols=["station", "time", "hs"])
    wave.time = r.pd.to_datetime(wave.time, utc=True)
    source_lookup = wave.set_index(["station", "time"]).hs
    anchors = r.pd.read_parquet(r.WORK / "train_anchors.parquet").set_index("anchor_id")
    expected = [
        source_lookup.loc[
            (str(st), anchors.loc[aid, "anchor_time"] + r.pd.Timedelta(hours=int(lead)))
        ]
        for aid, st, lead in frame[["anchor_id", "station", "lead_h"]].itertuples(
            index=False, name=None
        )
    ]
    checks["all_historical_targets_match_source"] = r.np.array_equal(
        r.np.asarray(expected), frame.target_hs.to_numpy()
    )
    for row in pre["prior_oof_actual_time_audit"]:
        checks["earlier_targets_available:" + row["fold"]] = (
            row["target_ready_fold_start_gap_hours"] > 0
        )
    for fold, item in pre["split_audit"]["folds"].items():
        checks["split_gap_episode:" + fold] = (
            item["minimum_train_validation_anchor_gap_hours"] >= 78
            and item["shared_train_validation_station_episode_count"] == 0
        )
    checks["finite_prediction_range"] = (
        r.np.isfinite(frame.final_prediction).all() and frame.final_prediction.between(0, 30).all()
    )
    checks["compute_under_six_hours"] = train["prepare_train_seconds"] < 21600
    if not training_only:
        answer = json.loads((r.REPORT / "answer-qa.json").read_text(encoding="utf-8"))
        repeat = json.loads((r.REPORT / "answer-replay-qa.json").read_text(encoding="utf-8"))
        prediction = r.pd.read_csv(r.ANSWER / "submission.csv")
        index = r.pd.read_csv(source / "test_index.csv", usecols=r.KEYS)
        checks["answer_schema"] = list(prediction.columns) == [*r.KEYS, "hs_pred"]
        checks["answer_original_key_order"] = prediction[r.KEYS].equals(index[r.KEYS])
        checks["answer_rows_unique_range"] = (
            len(prediction) == 1200
            and not prediction.duplicated(r.KEYS).any()
            and r.np.isfinite(prediction.hs_pred).all()
            and prediction.hs_pred.between(0, 30).all()
        )
        checks["exact_csv_fresh_pid_replay"] = (
            repeat["status"] == "PASS"
            and repeat["exact_csv_bytes"]
            and repeat["pid"] != answer["pid"]
            and repeat["sha256"] == answer["sha256"] == r.sha(r.ANSWER / "submission.csv")
        )
        checks["sample_hidden_upload_zero"] = (
            answer["sample_rows"] == 0 and answer["hidden_rows"] == 0 and answer["uploads"] == 0
        )
        wall = json.loads((r.REPORT / "source-to-answer-wall.json").read_text(encoding="utf-8"))
        checks["source_to_answer_wall_under_six_hours"] = (
            wall["under_six_hours"] and wall["elapsed_wall_seconds_including_process_gaps"] < 21600
        )
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    receipt = {
        "status": "FAIL" if failed else "PASS",
        "pid": os.getpid(),
        "training_result_sha256": r.sha(training_path),
        "checks_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "independent_sse_m2": sse,
        "independent_rmse_m": rmse,
        "historical_reference_rmse_exact": rmse == config["expected_clean_rmse_m_qa_only"],
        "historical_reference_drift_m": rmse - config["expected_clean_rmse_m_qa_only"],
        "historical_equality_is_not_safety_gate": True,
        "seconds": time.perf_counter() - started,
        "training_only": training_only,
        "os_network_isolation_certified": False,
    }
    path = r.REPORT / ("training-independent-qa.json" if training_only else "independent-qa.json")
    r.save(path, receipt)
    print(
        json.dumps(
            {"status": receipt["status"], "checks_count": len(checks), "failed_checks": failed}
        ),
        flush=True,
    )
    if failed:
        raise ValueError("independent QA failed: " + ", ".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-only", action="store_true")
    audit(parser.parse_args().training_only)
