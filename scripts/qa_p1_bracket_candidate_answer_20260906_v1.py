"""Post-seal local answer integrity and baseline-difference audit; never scores or fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

KEYS = ["station", "year", "layer", "time"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    docs = package / "06_docs"
    records = {name: json.loads((docs / filename).read_text(encoding="utf-8")) for name, filename in {
        "model_qa": "model-replay-qa.json", "root_qa": "independent-training-qa.json",
        "inference": "inference-qa.json", "replay": "answer-replay-qa.json"}.items()}
    training = json.loads((package / "train_result.json").read_text(encoding="utf-8"))
    answer_path = package / "05_answer/P1_submission.csv"
    checks = []

    def check(name, passed):
        checks.append({"check": name, "pass": bool(passed)})
        if not passed:
            raise ValueError(name)

    check("root_qa_pass", records["root_qa"]["status"] == "PASS")
    check("model_qa_pass", records["model_qa"]["status"] == "PASS")
    check("distinct_execution_processes", len({training["pid"], *(r["pid"] for r in records.values()), os.getpid()}) == 6)
    check("all_receipts_sealed_train", all(r["train_result_sha256"] == sha(package / "train_result.json")
                                         for r in records.values()))
    check("all_receipts_sealed_recipe", all(r["recipe_sha256"] == sha(package / "03_model/frozen_recipe.json")
                                          for r in records.values()))
    for name in ("inference", "replay"):
        record = records[name]
        check(name + "_root_gate", record["independent_training_qa_sha256"] == sha(docs / "independent-training-qa.json"))
        check(name + "_model_gate", record["model_replay_qa_sha256"] == sha(docs / "model-replay-qa.json"))
        check(name + "_no_fit_hidden_upload", record["new_fits"] == record["hidden_rows"] == record["uploads"]
              == record["sample_prediction_values_read"] == 0)
        check(name + "_within_budget", record["workflow_elapsed_seconds"] < 3600)
    check("terminal_local_answer", records["inference"]["status"] == "LOCAL_CANDIDATE_CREATED"
          and records["replay"]["status"] == "FRESH_PROCESS_ANSWER_REPLAY_PASS")
    check("exact_answer_sha", sha(answer_path) == records["inference"]["sha256"] == records["replay"]["sha256"])
    for fit in training["fits"]:
        check(fit["model_file"] + "_still_sealed", sha(package / "03_model" / fit["model_file"]) == fit["sha256"])
    manifest = json.loads((package / "02_code/source-manifest.json").read_text(encoding="utf-8"))
    check("source_snapshot_unchanged", all(sha(package / "02_code" / path) == item["sha256"]
                                           for path, item in manifest.items()))
    answer = pd.read_csv(answer_path)
    keys = pd.read_csv(args.sample, usecols=KEYS)
    check("schema", list(answer) == KEYS + ["label"])
    check("count_order_unique", len(answer) == len(keys) == 169011 and answer[KEYS].equals(keys)
          and not answer.duplicated(KEYS).any() and not keys.duplicated().any())
    check("finite_binary_no_null", not answer.isna().any().any() and answer.label.isin([0, 1]).all())
    check("positive_count", int(answer.label.sum()) == records["inference"]["positive_rows"])
    # Only after the frozen candidate is complete: compare with the fixed fallback for reporting.
    check("baseline_identity", sha(args.baseline) == "5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a")
    baseline = pd.read_csv(args.baseline)
    check("baseline_aligned", list(baseline) == list(answer) and baseline[KEYS].equals(answer[KEYS]))
    changed = answer.label.ne(baseline.label)
    result = {"status": "PASS", "pid": os.getpid(), "script_sha256": sha(Path(__file__)),
              "checks": checks, "passed": len(checks), "failed": 0,
              "answer_path": str(answer_path), "rows": len(answer), "sha256": sha(answer_path),
              "positive_rows": int(answer.label.sum()), "baseline_sha256": sha(args.baseline),
              "changed_rows": int(changed.sum()),
              "zero_to_one": int(((baseline.label == 0) & (answer.label == 1)).sum()),
              "one_to_zero": int(((baseline.label == 1) & (answer.label == 0)).sum()),
              "changed_rows_by_station": {str(k): int(v) for k, v in changed.groupby(answer.station).sum().items()},
              "sample_key_rows": len(keys), "sample_prediction_values_read": 0,
              "baseline_answer_read_for_postseal_comparison_only": True,
              "hidden_rows": 0, "new_fits": 0, "uploads": 0, "official_score": None,
              "limitations": ["Changed rows are not known corrections; no hidden truth was accessed.",
                              "One cold training workflow plus saved-model replay, not a second cold rebuild."]}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("status", "passed", "rows", "sha256", "positive_rows", "changed_rows", "zero_to_one", "one_to_zero")}))


if __name__ == "__main__":
    main()
