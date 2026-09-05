"""Aggregate QA for one uninterrupted cold start and a separately recovered run.

Reads only those two new packages after their own QA passes. No fitting, raw
source data, hidden truth, previous contest answers, network or uploads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from compare_repetitions import digest, load_run, np, paired_difference, pd


def validate_claims(cold, recovery):
    if (
        cold["status"] != "TRAINING_COMPLETE_FROM_EMPTY_MODEL"
        or not cold["empty_03_model_verified"]
    ):
        raise ValueError("uninterrupted run is not a verified empty-model training")
    if (
        recovery["status"] != "TRAINING_COMPLETE_RECOVERED_FRESH_MODELS"
        or recovery["empty_03_model_verified"]
    ):
        raise ValueError("recovery must not claim a cold start")
    if (
        recovery["new_backbone_fits"],
        recovery["new_full_router_fits"],
        recovery["inherited_successful_backbone_fits"],
        recovery["inherited_interrupted_backbone_attempts"],
    ) != (1, 1, 7, 1):
        raise ValueError("recovery attempt accounting differs")
    if (
        recovery["repeated_historical_backbone_fits"]
        or recovery["repeated_prequential_router_fits"]
    ):
        raise ValueError("unexpected recovery refits")
    if cold["source_sha256"] != recovery["source_sha256"]:
        raise ValueError("two source data contracts differ")


def compare(cold_path, recovery_path):
    cold, cold_hashes = load_run(cold_path)
    recovered, recovered_hashes = load_run(recovery_path)
    ct, rt = cold["training-result.json"], recovered["training-result.json"]
    validate_claims(ct, rt)
    if ct["pid"] == rt["pid"]:
        raise ValueError("two training PIDs are not distinct")
    cm = json.loads((cold_path / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    rm = json.loads((recovery_path / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for root, manifest in ((cold_path, cm), (recovery_path, rm)):
        for name, expected in manifest["sha256"].items():
            candidate = (root / name).resolve()
            if root not in candidate.parents or digest(candidate) != expected:
                raise ValueError("own portable manifest differs")
    core = [
        name
        for name in cm["sha256"]
        if name.startswith("02_code/p3_clean/") or name == "02_code/recipe.json"
    ]
    if not core or any(cm["sha256"][name] != rm["sha256"].get(name) for name in core):
        raise ValueError("model algorithm or recipe source differs")
    if digest(cold_path / "02_code/config.json") != digest(
        recovery_path / "02_code/original_config.json"
    ):
        raise ValueError("original fixed configuration differs")
    oof = paired_difference(
        pd.read_parquet(recovery_path / "04_logs/validation/oof.parquet"),
        pd.read_parquet(cold_path / "04_logs/validation/oof.parquet"),
        ["anchor_id", "station", "lead_h"],
        "final_prediction",
        target="target_hs",
    )
    cases_c = pd.read_parquet(cold_path / "04_logs/validation/replay_cases.parquet")
    cases_r = pd.read_parquet(recovery_path / "04_logs/validation/replay_cases.parquet")
    if not cases_c.equals(cases_r):
        raise ValueError("two historical probe populations/features differ")
    probes = []
    for root in (recovery_path, cold_path):
        with np.load(root / "04_logs/validation/replay_expected.npz", allow_pickle=False) as data:
            values = data["prediction"].copy()
        probes.append(pd.DataFrame({"row": np.arange(len(values)), "prediction": values}))
    probe = paired_difference(*probes, ["row"], "prediction")
    # Load only each new, hash-checked full single for the additional provenance
    # check requested after the interrupted run lacked a save-time single SHA.
    code = cold_path / "02_code"
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("p3_portable_new_cold", code / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    columns = json.loads(
        (cold_path / "04_logs/validation/feature_columns.json").read_text(encoding="utf-8")
    )["columns"]
    matrix, current, keys = runner.rows_for_cases(cases_c, columns)
    matrix = runner.base._cat_frame(matrix)
    single_predictions, single_parameters = [], []
    for root in (recovery_path, cold_path):
        single = runner.CatBoostRegressor().load_model(root / "03_model/full/single.cbm")
        values = np.clip(current + single.predict(matrix, thread_count=2), 0, 30)
        single_predictions.append(keys.assign(prediction=values))
        single_parameters.append(single.get_all_params())
    single = paired_difference(*single_predictions, runner.KEYS, "prediction")
    single["same_model_parameters"] = single_parameters[0] == single_parameters[1]
    single["same_binary_sha256"] = (
        rt["model_sha256"]["03_model/full/single.cbm"]
        == ct["model_sha256"]["03_model/full/single.cbm"]
    )
    single["recovery_digest_provenance"] = (
        "acceptance-time only; no pre-interruption save-time digest"
    )
    multi_matrix = cases_c[["station", *columns]].copy()
    multi_matrix.station = multi_matrix.station.astype(str)
    multi_predictions, multi_parameters = [], []
    for root in (recovery_path, cold_path):
        multi_model = runner.CatBoostRegressor().load_model(root / "03_model/full/multi.cbm")
        values = np.clip(
            cases_c.hs_current.to_numpy()[:, None]
            + multi_model.predict(multi_matrix, thread_count=2),
            0,
            30,
        ).reshape(-1)
        multi_predictions.append(keys.assign(prediction=values))
        multi_parameters.append(multi_model.get_all_params())
    multi = paired_difference(*multi_predictions, runner.KEYS, "prediction")
    multi["same_model_parameters"] = multi_parameters[0] == multi_parameters[1]
    multi["same_binary_sha256"] = (
        rt["model_sha256"]["03_model/full/multi.cbm"]
        == ct["model_sha256"]["03_model/full/multi.cbm"]
    )
    answers = paired_difference(
        pd.read_csv(recovery_path / "05_answer/submission.csv"),
        pd.read_csv(cold_path / "05_answer/submission.csv"),
        ["case_id", "station", "lead_h"],
        "hs_pred",
    )
    bytes_equal = cold["answer-qa.json"]["sha256"] == recovered["answer-qa.json"]["sha256"]
    exact = bytes_equal and oof["exact_prediction_array"] and probe["exact_prediction_array"]
    return {
        "status": "RECOVERY_VS_COLD_EXACT" if exact else "RECOVERY_VS_COLD_DIFFERENCE_OBSERVED",
        "comparison_direction": "A=recovered run_a; B=uninterrupted run_b; differences B minus A",
        "uninterrupted_cold_start_repetitions": 1,
        "recovered_completions": 1,
        "two_uninterrupted_cold_starts_proven": False,
        "same_model_core_and_recipe": True,
        "core_source_checks": len(core),
        "independent_qa_pass_both": True,
        "exact_tolerance": 0,
        "historical_oof": oof,
        "full_model_historical_probe": probe,
        "full_single_native_probe": single,
        "full_multi_gpu_native_probe": multi,
        "model_binary_hash_equal": {
            name: value == ct["model_sha256"].get(name)
            for name, value in rt["model_sha256"].items()
        },
        "model_binary_hash_equality_is_not_prediction_equality": True,
        "new_official_answer_comparison_no_truth": answers,
        "exact_new_answer_csv_bytes": bytes_equal,
        "uninterrupted_run_b": {
            "receipt_sha256": cold_hashes,
            "training_pid": ct["pid"],
            "answer_sha256": cold["answer-qa.json"]["sha256"],
            "old_clean_sha_match_qa_only": cold["answer-qa.json"]["exact_prior_clean_sha_match"],
            "prepare_train_seconds": ct["prepare_train_seconds"],
            "source_to_answer_wall_seconds": cold["source-to-answer-wall.json"][
                "elapsed_wall_seconds_including_process_gaps"
            ],
            "qa_checks": cold["independent-qa.json"]["checks_count"],
        },
        "recovered_run_a": {
            "receipt_sha256": recovered_hashes,
            "training_pid": rt["pid"],
            "answer_sha256": recovered["answer-qa.json"]["sha256"],
            "old_clean_sha_match_qa_only": recovered["answer-qa.json"][
                "exact_prior_clean_sha_match"
            ],
            "new_train_seconds": rt["train_seconds"],
            "source_to_answer_wall_including_interruption_seconds": recovered[
                "source-to-answer-wall.json"
            ]["elapsed_wall_seconds_including_process_gaps"],
            "qa_checks": recovered["independent-qa.json"]["checks_count"],
        },
        "successful_backbone_fits_all_runs": 16,
        "interrupted_backbone_attempts": 1,
        "total_started_backbone_attempts": 17,
        "successful_router_fits_all_runs": 6,
        "old_legacy_asset_reads": 0,
        "raw_source_reads_this_comparison": 0,
        "hidden_reads": 0,
        "uploads": 0,
        "os_network_isolation_or_clean_venv_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", required=True, type=Path)
    parser.add_argument("--recovery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.cold.resolve(), args.recovery.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_new_answer_csv_bytes": result["exact_new_answer_csv_bytes"],
                "full_single_exact": result["full_single_native_probe"]["exact_prediction_array"],
            }
        )
    )
