"""Read-only comparison of two completed portable P1 builds; write aggregate QA."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(first, second):
    roots = [first.resolve(), second.resolve()]
    receipts = [{name: read(root / "06_docs" / name) for name in (
        "train-result.json", "inference-qa.json", "replay-qa.json")}
        for root in roots]
    checks = {}
    models = []
    for index, (root, receipt) in enumerate(zip(roots, receipts, strict=True)):
        train, infer, replay = (receipt[name] for name in (
            "train-result.json", "inference-qa.json", "replay-qa.json"))
        prefix = f"run_{index + 1}"
        checks[prefix + "_four_fresh_fits"] = (
            len(train["fits"]) == 4 and all(x["fresh_initialization"] for x in train["fits"])
            and train["empty_03_model_verified_before_training"])
        checks[prefix + "_separate_processes"] = len({
            train["pid"], infer["pid"], replay["pid"]}) == 3
        checks[prefix + "_source_hashes"] = all(
            sha(root / "02_code" / name) == expected
            for name, expected in train["source_hashes"].items())
        checks[prefix + "_model_hashes"] = all(
            sha(root / "03_model" / fit["model_file"]) == fit["sha256"]
            for fit in train["fits"])
        answer = root / "05_answer" / "P1_submission.csv"
        checks[prefix + "_answer_receipts"] = (
            sha(answer) == infer["sha256"] == replay["sha256"])
        frame = pd.read_csv(answer)
        keys = ["station", "year", "layer", "time"]
        checks[prefix + "_answer_schema"] = (
            list(frame.columns) == keys + ["label"] and len(frame) == 169011
            and not frame[keys].duplicated().any() and frame.notna().all().all()
            and frame.label.isin([0, 1]).all())
        checks[prefix + "_six_hour_budget"] = infer["training_plus_inference_seconds"] <= 21600
        checks[prefix + "_prohibited_access_receipt_zero"] = (
            all(train[key] == 0 for key in ("official_rows", "sample_values_read", "hidden_rows",
                "external_observation_rows", "prior_models_read", "prior_answer_values_read", "uploads"))
            and all(item[key] == 0 for item in (infer, replay) for key in (
                "hidden_rows", "sample_prediction_values_read", "prior_models_read",
                "prior_answer_values_read", "new_fits_in_inference", "uploads")))
        models.append({fit["model_file"]: fit["sha256"] for fit in train["fits"]})
    checks["same_code_manifest"] = sha(first / "02_code/source-manifest.json") == sha(
        second / "02_code/source-manifest.json")
    recipes = [read(root / "03_model/frozen_recipe.json") for root in roots]
    recipe_keys = set(recipes[0]) | set(recipes[1])
    recipe_differences = [key for key in sorted(recipe_keys)
                          if recipes[0].get(key) != recipes[1].get(key)]
    checks["same_rederived_recipe_except_process_identity"] = all(
        key == "inner_training_pid" for key in recipe_differences)
    checks["same_four_model_hashes"] = models[0] == models[1]
    checks["same_answer_bytes"] = sha(first / "05_answer/P1_submission.csv") == sha(
        second / "05_answer/P1_submission.csv")
    inventory = []
    for path in sorted((first / "02_code").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                inventory.append({"file": path.relative_to(first).as_posix(),
                                  "line": node.lineno, "value": node.value})
    return {
        "status": "TWO_LOCAL_OUTSIDE_REPO_REBUILDS_PASS" if all(checks.values()) else "QA_REVIEW_REQUIRED",
        "checks": {key: bool(value) for key, value in checks.items()},
        "runs": [{"path": str(root), "training_pid": rec["train-result.json"]["pid"],
                  "train_seconds": rec["train-result.json"]["runtime_seconds"],
                  "train_plus_infer_seconds": rec["inference-qa.json"]["training_plus_inference_seconds"],
                  "replay_seconds": rec["replay-qa.json"]["runtime_seconds"],
                  "answer_sha256": rec["inference-qa.json"]["sha256"]}
                 for root, rec in zip(roots, receipts, strict=True)],
        "python": platform.python_version(), "actual_fits_total": 8,
        "recipe_metadata_differences": recipe_differences,
        "recipe_comparison_note": "Only execution PID excluded; every learned/config value remains exact. Initial byte-level QA is preserved.",
        "model_hashes": models, "numeric_literal_inventory": inventory,
        "numeric_inventory_is_not_provenance_proof": True,
        "boundary": {"outside_repository_working_directory": True,
                     "python_repo_open_and_socket_audit_guards": True,
                     "same_existing_virtualenv_used": True,
                     "os_network_isolation_verified": False,
                     "fresh_machine_or_fresh_dependency_install_verified": False,
                     "organizer_acceptance_or_final_lock": False},
        "earlier_partial_build": "temp suffix a: UTF-8 README build error; zero model fits; preserved",
        "upload_count": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
    raise SystemExit(0 if all(result["checks"].values()) else 1)
