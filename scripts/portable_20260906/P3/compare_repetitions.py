"""Compare two completed new P3 cleanroom runs without reading old assets.

Only aggregate differences are published. This script performs no fitting,
inference, source-data reads, model deserialization, or network operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_option] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def paired_difference(left, right, keys, value, *, target=None):
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("duplicate paired keys")
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("population or original key order differs")
    if target and not np.array_equal(left[target].to_numpy(), right[target].to_numpy()):
        raise ValueError("paired training targets differ")
    a, b = left[value].to_numpy(dtype=float), right[value].to_numpy(dtype=float)
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonempty finite paired predictions required")
    diff = b - a
    result = {
        "rows": len(left),
        "exact_keys_and_order": True,
        "exact_prediction_array": bool(np.array_equal(a, b)),
        "different_rows": int(np.count_nonzero(diff)),
        "max_abs_difference_m": float(np.max(np.abs(diff))),
        "difference_rmse_m": math.sqrt(math.fsum(float(x) ** 2 for x in diff) / len(diff)),
        "mean_signed_difference_b_minus_a_m": math.fsum(float(x) for x in diff) / len(diff),
    }
    if target:
        y = left[target].to_numpy(dtype=float)
        for label, prediction in (("a", a), ("b", b)):
            result[f"rmse_{label}_m"] = math.sqrt(
                math.fsum(float(x) ** 2 for x in prediction - y) / len(y)
            )
        result["rmse_delta_b_minus_a_m"] = result["rmse_b_m"] - result["rmse_a_m"]
    return result


def load_run(path):
    receipts = path / "04_logs" / "receipts"
    names = [
        "prepare.json",
        "training-result.json",
        "fresh-process-replay.json",
        "training-independent-qa.json",
        "answer-qa.json",
        "answer-replay-qa.json",
        "independent-qa.json",
        "source-to-answer-wall.json",
        "regenerated-oof.json",
    ]
    docs = {name: json.loads((receipts / name).read_text(encoding="utf-8")) for name in names}
    train = docs["training-result.json"]
    for name in ("training-independent-qa.json", "independent-qa.json"):
        if docs[name]["status"] != "PASS" or docs[name]["failed_checks"]:
            raise ValueError("a run has failed independent safety checks")
        if docs[name]["training_result_sha256"] != digest(receipts / "training-result.json"):
            raise ValueError("training result link changed")
    for name, expected in {**train["model_sha256"], **train["files_sha256"]}.items():
        if digest(path / name) != expected:
            raise ValueError("a run's own sealed model or validation hash differs")
    if (
        digest(path / "04_logs/validation/oof.parquet")
        != docs["regenerated-oof.json"]["oof_sha256"]
    ):
        raise ValueError("own OOF hash differs")
    answer_digest = digest(path / "05_answer/submission.csv")
    if answer_digest != docs["answer-qa.json"]["sha256"]:
        raise ValueError("own answer hash differs")
    if answer_digest != docs["answer-replay-qa.json"]["sha256"]:
        raise ValueError("own fresh answer replay hash differs")
    return docs, {name: digest(receipts / name) for name in names}


def compare(run_a, run_b):
    a, hashes_a = load_run(run_a)
    b, hashes_b = load_run(run_b)
    manifest_a = json.loads((run_a / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((run_b / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest_a != manifest_b:
        raise ValueError("independent repetitions do not share the same sealed package")
    for run in (run_a, run_b):
        for name, expected in manifest_a["sha256"].items():
            if digest(run / name) != expected:
                raise ValueError("portable source manifest differs")
    trains = [a["training-result.json"], b["training-result.json"]]
    if trains[0]["pid"] == trains[1]["pid"]:
        raise ValueError("independent training process IDs are not distinct")
    for field in ("source_sha256", "runner_sha256", "config_sha256"):
        if trains[0][field] != trains[1][field]:
            raise ValueError("independent source/recipe contract differs")
    for train in trains:
        if (
            train["backbone_fits"],
            train["prequential_router_fits"],
            train["full_router_fits"],
        ) != (8, 2, 1):
            raise ValueError("fit contract differs")
        if not train["empty_03_model_verified"] or train["old_model_cache_oof_answer_reads"] != 0:
            raise ValueError("empty-model source-only contract differs")
    oof_a = pd.read_parquet(run_a / "04_logs/validation/oof.parquet")
    oof_b = pd.read_parquet(run_b / "04_logs/validation/oof.parquet")
    oof = paired_difference(
        oof_a,
        oof_b,
        ["anchor_id", "station", "lead_h"],
        "final_prediction",
        target="target_hs",
    )
    case_a = pd.read_parquet(run_a / "04_logs/validation/replay_cases.parquet")
    case_b = pd.read_parquet(run_b / "04_logs/validation/replay_cases.parquet")
    if not case_a.equals(case_b):
        raise ValueError("full-model historical replay inputs differ")
    with np.load(run_a / "04_logs/validation/replay_expected.npz", allow_pickle=False) as data:
        pred_a = data["prediction"].copy()
    with np.load(run_b / "04_logs/validation/replay_expected.npz", allow_pickle=False) as data:
        pred_b = data["prediction"].copy()
    probe = paired_difference(
        pd.DataFrame({"row": np.arange(len(pred_a)), "p": pred_a}),
        pd.DataFrame({"row": np.arange(len(pred_b)), "p": pred_b}),
        ["row"],
        "p",
    )
    official = paired_difference(
        pd.read_csv(run_a / "05_answer/submission.csv"),
        pd.read_csv(run_b / "05_answer/submission.csv"),
        ["case_id", "station", "lead_h"],
        "hs_pred",
    )
    equal_model_hashes = {
        name: value == trains[1]["model_sha256"].get(name)
        for name, value in trains[0]["model_sha256"].items()
    }
    exact_answers = a["answer-qa.json"]["sha256"] == b["answer-qa.json"]["sha256"]
    return {
        "status": "REPETITION_EXACT"
        if exact_answers and oof["exact_prediction_array"] and probe["exact_prediction_array"]
        else "REPETITION_DIFFERENCE_OBSERVED",
        "scope": "two independently retrained new packages only; no old models/OOF/CSV reads",
        "exact_tolerance": 0,
        "same_package_manifest": True,
        "source_to_empty_model_contract_pass": True,
        "saved_model_replay_pass_both": all(
            x["fresh-process-replay.json"]["status"] == "PASS" for x in (a, b)
        ),
        "csv_fresh_process_replay_pass_both": all(
            x["answer-replay-qa.json"]["status"] == "PASS" for x in (a, b)
        ),
        "independent_qa_pass_both": True,
        "historical_oof": oof,
        "full_model_historical_probe": probe,
        "new_official_answer_comparison_no_truth": official,
        "exact_new_answer_csv_bytes": exact_answers,
        "model_binary_hash_equal": equal_model_hashes,
        "model_binary_hash_equality_not_equivalent_to_prediction_equality": True,
        "run_a": {
            "receipt_sha256": hashes_a,
            "training_pid": trains[0]["pid"],
            "answer_sha256": a["answer-qa.json"]["sha256"],
            "old_clean_sha_match_qa_only": a["answer-qa.json"]["exact_prior_clean_sha_match"],
            "prepare_train_seconds": trains[0]["prepare_train_seconds"],
            "source_to_answer_wall_seconds": a["source-to-answer-wall.json"][
                "elapsed_wall_seconds_including_process_gaps"
            ],
            "qa_checks": a["independent-qa.json"]["checks_count"],
        },
        "run_b": {
            "receipt_sha256": hashes_b,
            "training_pid": trains[1]["pid"],
            "answer_sha256": b["answer-qa.json"]["sha256"],
            "old_clean_sha_match_qa_only": b["answer-qa.json"]["exact_prior_clean_sha_match"],
            "prepare_train_seconds": trains[1]["prepare_train_seconds"],
            "source_to_answer_wall_seconds": b["source-to-answer-wall.json"][
                "elapsed_wall_seconds_including_process_gaps"
            ],
            "qa_checks": b["independent-qa.json"]["checks_count"],
        },
        "total_backbone_fits": sum(t["backbone_fits"] for t in trains),
        "total_router_fits": sum(
            t["prequential_router_fits"] + t["full_router_fits"] for t in trains
        ),
        "os_network_isolation_or_clean_venv_verified": False,
        "old_asset_reads": 0,
        "hidden_reads": 0,
        "uploads": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_a.resolve(), args.run_b.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_new_answer_csv_bytes": result["exact_new_answer_csv_bytes"],
            }
        )
    )
