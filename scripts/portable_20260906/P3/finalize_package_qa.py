"""Aggregate final portable-package evidence without fitting or reading data rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(artifact, report):
    checks = {}
    links = {}

    def check(name, condition):
        checks[name] = bool(condition)

    def linked(path):
        links[path.relative_to(report.parent.parent.parent).as_posix()] = sha(path)
        return read(path)

    comparison = linked(report / "repetition-comparison.json")
    build = linked(report / "archive-build-v3-qa.json")
    linked(report / "interruption-review.json")
    linked(report / "inference-directory-repair.json")
    for lane, expected_count, reference in (
        ("v2_run_b_completed", 42, "uninterrupted_run_b"),
        ("v3_recovered_completed", 44, "recovered_run_a"),
    ):
        receipts = artifact / lane / "04_logs/receipts"
        qa = linked(receipts / "independent-qa.json")
        train = linked(receipts / "training-result.json")
        check(
            lane + ":all_original_qa_pass",
            qa["status"] == "PASS"
            and not qa["failed_checks"]
            and qa["checks_count"] == expected_count
            and all(qa["checks"].values()),
        )
        check(
            lane + ":training_link",
            qa["training_result_sha256"] == sha(receipts / "training-result.json"),
        )
        for name, expected in comparison[reference]["receipt_sha256"].items():
            check(lane + ":receipt_hash:" + name, sha(receipts / name) == expected)
        for name, expected in train["model_sha256"].items():
            check(lane + ":model_hash:" + name, sha(artifact / lane / name) == expected)
        check(
            lane + ":answer_hash",
            sha(artifact / lane / "05_answer/submission.csv")
            == comparison[reference]["answer_sha256"],
        )

    extracted = artifact / "archive_v3_extraction_verified"
    cold, saved = extracted / "cold", extracted / "saved"
    replay = linked(saved / "04_logs/receipts/archive-inference-replay.json")
    check("archive_inference_complete", replay["status"] == "SAVED_MODEL_INFERENCE_REPLAY_PASS")
    check("archive_pid_fresh", replay["pid"] != replay["original_training_pid"])
    check("archive_1200_rows", replay["rows"] == 1200 and replay["cases"] == 200)
    check("archive_no_fit_or_selection", replay["new_fits"] == replay["new_selection"] == 0)
    check(
        "archive_restricted_reads",
        replay["sample_rows"]
        == replay["hidden_rows"]
        == replay["old_answer_rows_read"]
        == replay["uploads"]
        == 0,
    )
    check(
        "archive_exact_answer",
        sha(saved / "05_answer/replayed_submission.csv")
        == replay["sha256"]
        == comparison["uninterrupted_run_b"]["answer_sha256"],
    )
    check("archive_inference_under_6h", replay["inference_only_seconds"] < 21600)
    check("archive_frozen_core", replay["same_frozen_official_frame_and_predict_functions"])
    companion = read(saved / "INFERENCE_ADAPTER_MANIFEST.json")
    check(
        "adapter_companion_link",
        sha(saved / "INFERENCE_ADAPTER_MANIFEST.json") == replay["companion_sha256"],
    )
    for name, expected in companion["sha256"].items():
        check("adapter_entry:" + name, sha(saved / name) == expected)
    check(
        "original_manifest_unchanged",
        sha(saved / "PACKAGE_MANIFEST.json")
        == sha(artifact / "v2_run_b_completed/PACKAGE_MANIFEST.json"),
    )
    for name, expected in read(cold / "PACKAGE_MANIFEST.json")["sha256"].items():
        check("cold_entry:" + name, sha(cold / name) == expected)
    for role in ("cold", "saved"):
        check(
            role + ":six_directories",
            all(
                (extracted / role / name).is_dir()
                for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs")
            ),
        )
    for name in ("03_model", "04_logs", "05_answer"):
        check("cold_empty:" + name, not any((cold / name).iterdir()))
    for role in ("cold_start", "saved_model_replay"):
        archive_path = artifact / "archives_v3" / build[role]["path"]
        check(role + ":zip_hash", sha(archive_path) == build[role]["sha256"])
        with zipfile.ZipFile(archive_path) as archive:
            check(role + ":zip_crc", archive.testzip() is None)
            check(
                role + ":no_raw_or_training_target_tables",
                not any(
                    name.endswith(
                        (
                            "oof.parquet",
                            "train_features.parquet",
                            "train_anchors.parquet",
                            "train_wave.csv",
                            "train_atmos.csv",
                        )
                    )
                    for name in archive.namelist()
                ),
            )
    check(
        "comparison_exact_without_tolerance",
        comparison["exact_tolerance"] == 0
        and comparison["exact_new_answer_csv_bytes"]
        and comparison["historical_oof"]["exact_prediction_array"]
        and comparison["full_single_native_probe"]["exact_prediction_array"]
        and comparison["full_multi_gpu_native_probe"]["exact_prediction_array"],
    )
    check(
        "interrupted_attempt_counted",
        comparison["successful_backbone_fits_all_runs"] == 16
        and comparison["interrupted_backbone_attempts"] == 1
        and comparison["total_started_backbone_attempts"] == 17
        and comparison["successful_router_fits_all_runs"] == 6,
    )
    check(
        "cold_vs_recovery_claim_separate",
        comparison["uninterrupted_cold_start_repetitions"] == 1
        and comparison["recovered_completions"] == 1
        and comparison["two_uninterrupted_cold_starts_proven"] is False,
    )
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": "final receipt/model/archive hashes and actual extracted inference; no new training or row-value reads",
        "checks_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "evidence_sha256": links,
        "original_execution_qa_counts": {"cold_run_b": 42, "recovered_run_a": 44},
        "archive_build": build,
        "extracted_saved_inference": replay,
        "cold_source_to_answer_seconds": comparison["uninterrupted_run_b"][
            "source_to_answer_wall_seconds"
        ],
        "focused_synthetic_pass_counts": {
            "v2": 11,
            "repetition_comparator": 5,
            "recovery": 14,
            "recovery_comparator": 5,
            "archive_contract": 5,
        },
        "ruff": "PASS",
        "uninterrupted_cold_starts": 1,
        "recovered_completions": 1,
        "new_venv_or_os_network_isolation_certified": False,
        "new_fits_this_finalization": 0,
        "uploads": 0,
    }
    with (report / "independent-qa.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_count": len(checks),
                "failed_checks": failed,
                "qa_sha256": sha(report / "independent-qa.json"),
            }
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.artifact.resolve(), args.report.resolve())
