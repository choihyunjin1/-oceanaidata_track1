"""Independent arithmetic/file/empty-model/answer receipts for P3 regeneration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import run_p3_clean_regeneration_20260905_v4 as run


def main(*, training_only=False):
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    run.guard(source, official=not training_only)
    config = json.loads(run.CONFIG.read_text(encoding="utf-8"))
    report, work, out = run.REPORT, run.WORK, run.OUT
    names = ["prepare.json", "training-result.json", "regenerated-oof.json", "fresh-process-replay.json"]
    if not training_only:
        names += ["answer-qa.json", "answer-replay-qa.json"]
    receipts = {name: json.loads((report / name).read_text(encoding="utf-8")) for name in names}
    prepared, trained, evaluated, replay = (receipts[name] for name in names[:4])
    seal = json.loads((out / "seal.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "TRAIN_LOCK.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})

    check("runner_frozen", trained["runner_sha256"] == seal["runner_sha256"] == run.sha(run.ROOT / "scripts/run_p3_clean_regeneration_20260905_v4.py"))
    check("config_frozen", trained["config_sha256"] == seal["config_sha256"] == run.sha(run.CONFIG))
    for relative, expected in seal["snapshot_sha256"].items():
        check(f"snapshot/{relative}", run.sha(run.CODE / relative) == expected == run.sha(run.ROOT / relative))
    for name, expected in config["source_files"].items():
        check(f"source_copy/{name}", run.sha(source / name) == run.sha(run.DATA / name) == expected)
    check("initial_empty_model", seal["empty_03_model_initial"] and prepared["empty_03_model_after_prepare"] and lock["empty_03_model"] and trained["empty_03_model_verified"])
    check("initial_empty_answer", seal["empty_05_answer_initial"])
    check("fit_budget", trained["historical_backbone_fits"] == 6 and trained["full_backbone_fits"] == 2 and trained["backbone_fits"] == 8 and trained["prequential_router_fits"] == 2 and trained["full_router_fits"] == 1)
    check("no_old_assets", trained["old_model_cache_oof_answer_reads"] == 0 and trained["pretrained_weights"] == 0 and prepared["old_cache_reads"] == 0)
    check("no_early_official", prepared["official_rows"] == trained["official_rows"] == trained["csv_rows"] == replay["official_rows"] == 0)
    check("nine_saved_model_files", len(trained["model_sha256"]) == 9)
    for relative, expected in trained["model_sha256"].items():
        check(f"model/{relative}", run.sha(out / relative) == expected)
    for name, expected in prepared["files"].items():
        check(f"new_prepare/{name}", run.sha(work / name) == expected)
    oof = pd.read_parquet(work / "oof.parquet")
    anchors = pd.read_parquet(work / "train_anchors.parquet").set_index("anchor_id")
    keys = pd.read_parquet(work / "validation_keys.parquet")
    check("oof_hash", run.sha(work / "oof.parquet") == evaluated["oof_sha256"])
    check("oof_population", len(oof) == 1086 and oof.anchor_id.nunique() == 181 and len(keys) == 181)
    check("unique_keys", not oof.duplicated(["anchor_id", "station", "lead_h"]).any() and not keys.anchor_id.duplicated().any())
    check("target6_completeness", oof.groupby("anchor_id").lead_h.agg(lambda x: sorted(x) == list(run.LEADS)).all())
    for lead in run.LEADS:
        subset = oof[oof.lead_h.eq(lead)]
        check(f"source_generated_target/{lead}", np.array_equal(subset.target_hs, anchors.loc[subset.anchor_id, f"target_{lead}"]))
    truth, prediction = oof.target_hs.to_numpy(), oof.final_prediction.to_numpy()
    sse = float(np.dot(truth - prediction, truth - prediction))
    rmse = float(np.sqrt(sse / len(oof)))
    check("independent_pooled_sse", np.isclose(sse, evaluated["sse_m2"], rtol=0, atol=1e-9))
    check("independent_pooled_rmse", np.isclose(rmse, evaluated["rmse_m"], rtol=0, atol=1e-12) and np.isclose(rmse, trained["regenerated_oof_rmse_m"], rtol=0, atol=1e-12))
    check("fixed_clean_internal_reproduced", np.isclose(rmse, config["expected_clean_rmse_m_qa_only"], rtol=0, atol=1e-12))
    keys.anchor_time = pd.to_datetime(keys.anchor_time, utc=True)
    order = ["2024_h2_storm", "winter_transition", "2025_h1"]
    starts = ["2024-07-01", "2024-11-01", "2025-03-01"]
    for index, fold in enumerate(order):
        previous = keys[keys.fold.isin(order[:index])]
        current = keys[keys.fold.eq(fold)]
        check(f"fold_count/{fold}", len(current) == [49, 79, 53][index] and len(previous) == [0, 49, 128][index])
        if index:
            check(f"prior_targets_ready/{fold}", previous.anchor_time.max() + pd.Timedelta(hours=24) < pd.Timestamp(starts[index], tz="UTC"))
            overlap = set(zip(previous.station, previous.episode_id, strict=True)) & set(zip(current.station, current.episode_id, strict=True))
            check(f"no_episode_overlap/{fold}", not overlap)
    check("fresh_process", replay["fresh_pid"] != replay["training_pid"] == trained["pid"] and replay["rows"] == 1086 and replay["max_abs_prediction_error_m"] == 0)
    check("replay_training_link", replay["training_result_sha256"] == run.sha(report / "training-result.json"))
    if training_only:
        failed = [x["check"] for x in checks if not x["pass"]]
        receipt = {"status": "PASS" if not failed else "FAIL", "checks_count": len(checks), "failed_checks": failed, "checks": checks, "training_result_sha256": run.sha(report / "training-result.json"), "replay_sha256": run.sha(report / "fresh-process-replay.json"), "qa_runner_sha256": run.sha(Path(__file__)), "official_rows": 0, "submission_csv_rows": 0, "uploads": 0, "scope": "independent full training/model/source/OOF/chronology/fresh replay QA before any official inference"}
        run.save(report / "training-independent-qa.json", receipt)
        print(json.dumps({"status": receipt["status"], "checks": len(checks), "failed_checks": failed, "official_rows": 0}))
        if failed:
            raise SystemExit(1)
        return
    answer, answer_replay = receipts[names[4]], receipts[names[5]]
    frame = pd.read_csv(run.ANSWER / "submission.csv")
    check("answer_schema", list(frame) == [*run.KEYS, "hs_pred"] and len(frame) == 1200 and not frame.duplicated(run.KEYS).any())
    check("answer_range", np.isfinite(frame.hs_pred).all() and frame.hs_pred.between(0, 30).all())
    check("answer_six_leads", frame.groupby(["case_id", "station"]).lead_h.agg(lambda x: sorted(x) == list(run.LEADS)).all())
    check("answer_hash_chain", run.sha(run.ANSWER / "submission.csv") == answer["sha256"] == answer_replay["sha256"])
    check("known_clean_answer_reproduced", answer["exact_prior_clean_sha_match"] and answer["sha256"] == config["expected_clean_csv_sha256_qa_only"])
    check("answer_fresh_pid", answer["pid"] != trained["pid"] and answer_replay["pid"] != answer["pid"])
    check("answer_replay_link", answer_replay["answer_qa_sha256"] == run.sha(report / "answer-qa.json") and answer_replay["exact_csv_bytes"])
    check("answer_training_replay_links", answer["training_result_sha256"] == run.sha(report / "training-result.json") and answer["fresh_replay_sha256"] == run.sha(report / "fresh-process-replay.json"))
    check("authorized_public_inputs_only", answer["official_context_rows"] == 57800 and answer["official_index_rows"] == 1200 and answer["sample_rows"] == answer["hidden_rows"] == answer["old_answer_rows_read"] == answer["uploads"] == 0)
    check("six_hour_arithmetic", np.isclose(answer["source_train_replay_inference_seconds"], trained["prepare_train_seconds"] + replay["seconds"] + answer["seconds"], rtol=0, atol=1) and answer["source_train_replay_inference_seconds"] < 21600)
    failed = [x["check"] for x in checks if not x["pass"]]
    result = {"status": "PASS" if not failed else "FAIL", "checks_count": len(checks), "failed_checks": failed, "checks": checks, "canonical_receipt_sha256": {name: run.sha(report / name) for name in names}, "qa_runner_sha256": run.sha(Path(__file__)), "baseline_empty_model_training_to_answer_regeneration_pass": not failed, "official_local_answer_only": True, "uploads": 0, "source_or_target_values_printed": 0, "clean_machine_relocation_os_network_isolation_organizer_hardware_verified": False}
    run.save(report / "independent-qa.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed_checks": failed, "empty_model_to_answer_pass": not failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    main(training_only=args.training_only)
